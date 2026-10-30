"""Iterative DiffuserCam reconstruction for Lab 7.

Usage:
    from starter import InverseSolver, adjoint_test

    solver = InverseSolver(psf, algorithm="gradient_descent")
    x_hat = solver(y)
    plt.plot(solver.losses)
"""

import numpy as np


class InverseSolver:
    """Reconstructs DiffuserCam images via iterative optimization.

    Parameters:
        psf: Point spread function array (H, W, C).
        algorithm: 'gradient_descent', 'ista', or 'fista'.
        prox_fn: Proximal operator with signature
            prox_fn(x, lam, step_size) -> x_new.
            If None, no proximal operator is applied.
        num_iters: Number of iterations to run.
        step_size: Step size for gradient updates. If None,
            estimated automatically via power iteration.
        lam: Regularization parameter (passed to prox_fn).
        mask: Optional binary mask (H, W, C) for compressive
            sensing. If None, all pixels are used.
    """

    def __init__(
        self,
        psf,
        algorithm="gradient_descent",
        prox_fn=None,
        num_iters=200,
        step_size=None,
        lam=0.0,
        mask=None,
    ):
        algorithm = algorithm.replace(" ", "_")
        if algorithm not in {"gradient_descent", "ista", "fista"}:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        if not isinstance(num_iters, (int, np.integer)) or num_iters < 0:
            raise ValueError("num_iters must be a nonnegative integer")

        psf = np.asarray(psf, dtype=np.float64)
        if psf.ndim == 2:
            psf = psf[..., np.newaxis]
        if psf.ndim != 3 or not np.all(np.isfinite(psf)):
            raise ValueError("psf must be a finite H x W or H x W x C array")
        psf_norm = np.linalg.norm(psf)
        if np.isclose(psf_norm, 0.0):
            raise ValueError("psf must not be identically zero")

        # Normalize PSF and precompute frequency-domain arrays
        self.h = psf / psf_norm
        self.H = np.fft.fft2(
            np.fft.ifftshift(self.pad(self.h), axes=(0, 1)), axes=(0, 1), norm="ortho"
        )
        self.H_conj = np.conj(self.H)

        # Store mask for compressive sensing (Lab 8)
        self.mask = mask

        # Compute step size
        if step_size is None:
            full_shape = (self.h.shape[0] * 2, self.h.shape[1] * 2, self.h.shape[2])
            L = self.power_iteration(full_shape)
            self.lipschitz_constant = L
            self.step_size = 1.0 / L
        else:
            if not np.isscalar(step_size) or not np.isfinite(step_size) or step_size <= 0:
                raise ValueError("step_size must be positive and finite")
            self.lipschitz_constant = None
            self.step_size = float(step_size)

        self.algorithm = algorithm
        self.prox_fn = prox_fn
        self.num_iters = num_iters
        self.lam = lam

    def __call__(self, y, snapshot_iters=None):
        """Run reconstruction on measurement y.

        Returns the estimated image, cropped to original size.
        Also stores self.losses (list of loss values per iteration).

        If ``snapshot_iters`` is provided, ``self.snapshots`` stores cropped
        copies at those iteration numbers. Iteration zero is the initial guess.
        """
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 2:
            y = y[..., np.newaxis]
        if y.shape != self.h.shape:
            raise ValueError(f"measurement must have shape {self.h.shape}, got {y.shape}")
        if not np.all(np.isfinite(y)):
            raise ValueError("measurement must contain only finite values")

        x = np.zeros((y.shape[0] * 2, y.shape[1] * 2, y.shape[2]))
        v = np.zeros_like(x)  # momentum term (FISTA only)
        t = 1.0  # acceleration param (FISTA only)

        requested = set() if snapshot_iters is None else {int(i) for i in snapshot_iters}
        if any(i < 0 or i > self.num_iters for i in requested):
            raise ValueError("snapshot iterations must lie between 0 and num_iters")
        self.snapshots = {}
        if 0 in requested:
            self.snapshots[0] = self.crop(x).copy()

        self.losses = []
        for i in range(self.num_iters):
            if self.algorithm == "gradient_descent":
                x, loss = self.gd_update(x, y)
            elif self.algorithm == "ista":
                x, loss = self.ista_update(x, y)
            elif self.algorithm == "fista":
                v, t, x, loss = self.fista_update(v, t, x, y)
            self.losses.append(loss)
            iteration = i + 1
            if iteration in requested:
                self.snapshots[iteration] = self.crop(x).copy()

        return self.crop(x)

    # ----------------------------------------------------------------
    # Provided helper methods
    # ----------------------------------------------------------------

    def pad(self, img):
        """Zero-pad image to 2x size. (H,W,C) -> (2H,2W,C)"""
        img = np.asarray(img)
        return np.pad(
            img,
            (
                (img.shape[0] - img.shape[0] // 2, img.shape[0] // 2),
                (img.shape[1] - img.shape[1] // 2, img.shape[1] // 2),
                (0, 0),
            ),
            mode="constant",
        )

    def crop(self, img):
        """Crop padded image back to original size.
        (2H,2W,C) -> (H,W,C)"""
        height = img.shape[0] // 2
        width = img.shape[1] // 2
        row_start = (img.shape[0] - height + 1) // 2
        col_start = (img.shape[1] - width + 1) // 2
        return img[row_start : row_start + height, col_start : col_start + width, ...]

    def prox(self, x):
        """Apply the proximal operator if one was provided."""
        if self.prox_fn is None:
            return x
        return self.prox_fn(x, self.lam, self.step_size)

    # ----------------------------------------------------------------
    # Linear operators and optimization updates
    # ----------------------------------------------------------------

    def A(self, x):
        """Forward model: A(x) = crop(IFFT(H * FFT(x))).

        Steps:
            1. Compute FFT of x (axes 0,1, norm='ortho')
            2. Multiply by self.H in frequency domain
            3. Compute IFFT (axes 0,1, norm='ortho'), take real part
            4. Crop to measurement size using self.crop()
            5. If self.mask is not None, multiply by self.mask

        Args:
            x: Image in padded space (2H, 2W, C)
        Returns:
            Measurement-space result (H, W, C)
        """
        x = np.asarray(x, dtype=np.float64)
        if x.shape != self.H.shape:
            raise ValueError(f"x must have padded shape {self.H.shape}, got {x.shape}")

        X = np.fft.fft2(x, axes=(0, 1), norm="ortho")
        result = np.fft.ifft2(self.H * X, axes=(0, 1), norm="ortho").real
        result = self.crop(result)
        if self.mask is not None:
            result = result * self.mask
        return result

    def A_adjoint(self, y):
        """Adjoint: A*(y) = IFFT(conj(H) * FFT(pad(y))).

        Steps:
            1. If self.mask is not None, multiply y by self.mask
            2. Zero-pad y using self.pad()
            3. Compute FFT of padded y (axes 0,1, norm='ortho')
            4. Multiply by self.H_conj in frequency domain
            5. Compute IFFT (axes 0,1, norm='ortho'), take real part

        Args:
            y: Measurement-space input (H, W, C)
        Returns:
            Image-space result (2H, 2W, C)
        """
        y = np.asarray(y, dtype=np.float64)
        if y.shape != self.h.shape:
            raise ValueError(f"y must have measurement shape {self.h.shape}, got {y.shape}")
        if self.mask is not None:
            y = y * self.mask

        Y = np.fft.fft2(self.pad(y), axes=(0, 1), norm="ortho")
        return np.fft.ifft2(
            self.H_conj * Y,
            axes=(0, 1),
            norm="ortho",
        ).real

    def power_iteration(self, full_shape, num_iters=10):
        """Estimate largest eigenvalue of A*A via power iteration.

        Do NOT include crop/pad — work in the full padded space. The crop can
        only reduce the operator norm, so the full convolution gives a safe
        Lipschitz estimate. Repeatedly apply H*H in frequency domain:
            b_{k+1} = real(IFFT(|H|^2 * FFT(b_k)))
            b_{k+1} = b_{k+1} / ||b_{k+1}||

        After num_iters, estimate the eigenvalue via the Rayleigh
        quotient: eigenvalue = (b^T M b) / (b^T b)
        where M b = real(IFFT(|H|^2 * FFT(b))).

        Args:
            full_shape: (2H, 2W, C) shape of padded space
            num_iters: Number of iterations (default 10)
        Returns:
            Estimated largest eigenvalue (float)
        """
        if tuple(full_shape) != self.H.shape:
            raise ValueError(f"full_shape must be {self.H.shape}")
        if not isinstance(num_iters, (int, np.integer)) or num_iters <= 0:
            raise ValueError("num_iters must be a positive integer")

        rng = np.random.default_rng(4660)
        b = rng.standard_normal(full_shape)
        b /= np.linalg.norm(b)

        def normal_operator(vector):
            spectrum = np.fft.fft2(vector, axes=(0, 1), norm="ortho")
            return np.fft.ifft2(
                np.abs(self.H) ** 2 * spectrum,
                axes=(0, 1),
                norm="ortho",
            ).real

        for _ in range(num_iters):
            Mb = normal_operator(b)
            magnitude = np.linalg.norm(Mb)
            if np.isclose(magnitude, 0.0):
                raise ValueError("PSF normal operator has zero magnitude")
            b = Mb / magnitude

        Mb = normal_operator(b)
        eigenvalue = np.vdot(b, Mb).real / np.vdot(b, b).real
        if not np.isfinite(eigenvalue) or eigenvalue <= 0:
            raise ValueError("power iteration did not find a positive eigenvalue")
        return float(eigenvalue)

    def gd_update(self, x, y):
        """One gradient descent step.

        x_new = x - step_size * A_adjoint(A(x) - y)

        Args:
            x: Current estimate (2H, 2W, C)
            y: Measurement (H, W, C)
        Returns:
            (x_new, loss) where loss = self.loss(error)
        """
        error = self.A(x) - y
        gradient = self.A_adjoint(error)
        x_new = x - self.step_size * gradient
        return x_new, self.loss(error)

    def ista_update(self, x, y):
        """One proximal-gradient (ISTA) update."""
        error = self.A(x) - y
        gradient = self.A_adjoint(error)
        x_new = self.prox(x - self.step_size * gradient)
        return x_new, self.loss(error)

    def fista_update(self, v, t, x, y):
        """One accelerated proximal-gradient (FISTA) update."""
        error = self.A(v) - y
        gradient = self.A_adjoint(error)
        x_new = self.prox(v - self.step_size * gradient)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t**2))
        v_new = x_new + ((t - 1.0) / t_new) * (x_new - x)
        return v_new, t_new, x_new, self.loss(error)

    def loss(self, error):
        """Compute the data fidelity loss.

        loss = 0.5 * ||error||_2^2

        Args:
            error: Residual A(x) - y
        Returns:
            Scalar loss value (float)
        """
        error = np.asarray(error)
        return float(0.5 * np.vdot(error, error).real)


def adjoint_test(solver, y_shape):
    """Verify that A and A_adjoint satisfy <y, Ax> == <A*y, x>.

    Creates random x (in padded space) and y (in measurement space),
    then checks that the two inner products are equal.

    Args:
        solver: An InverseSolver instance (with A and A_adjoint
            implemented)
        y_shape: Shape of measurement, e.g. (H, W, C)
    Returns:
        (lhs, rhs) tuple — these should be nearly equal
    """
    full_shape = (y_shape[0] * 2, y_shape[1] * 2, y_shape[2])
    rng = np.random.default_rng(4660)
    x = rng.standard_normal(full_shape)
    y = rng.standard_normal(y_shape)

    Ax = solver.A(x)
    As_y = solver.A_adjoint(y)

    lhs = np.vdot(y, Ax).real
    rhs = np.vdot(As_y, x).real
    return float(lhs), float(rhs)
