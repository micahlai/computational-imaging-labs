"""InverseSolver for DiffuserCam reconstruction — Lab 7 starter code.

Fill in the methods marked TODO to complete the lab.

Usage:
    from inverse_solver import InverseSolver, adjoint_test

    solver = InverseSolver(psf, algorithm='fista',
                           prox_fn=lambda x, l, s: np.maximum(x, 0))
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
        assert algorithm in ["gradient_descent", "ista", "fista"], (
            f"Unknown algorithm: {algorithm}"
        )

        # Normalize PSF and precompute frequency-domain arrays
        self.h = psf / np.linalg.norm(psf)
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
            self.step_size = 1.0 / L
        else:
            self.step_size = step_size

        self.algorithm = algorithm
        self.prox_fn = prox_fn
        self.num_iters = num_iters
        self.lam = lam

    def __call__(self, y):
        """Run reconstruction on measurement y.

        Returns the estimated image, cropped to original size.
        Also stores self.losses (list of loss values per iteration).
        """
        x = np.zeros((y.shape[0] * 2, y.shape[1] * 2, y.shape[2]))
        v = np.zeros_like(x)  # momentum term (FISTA only)
        t = 1.0  # acceleration param (FISTA only)

        self.losses = []
        for i in range(self.num_iters):
            if self.algorithm == "gradient_descent":
                x, loss = self.gd_update(x, y)
            elif self.algorithm == "ista":
                x, loss = self.ista_update(x, y)
            elif self.algorithm == "fista":
                v, t, x, loss = self.fista_update(v, t, x, y)
            self.losses.append(loss)

        return self.crop(x)

    # ----------------------------------------------------------------
    # Provided helper methods
    # ----------------------------------------------------------------

    def pad(self, img):
        """Zero-pad image to 2x size. (H,W,C) -> (2H,2W,C)"""
        return np.pad(
            img,
            (
                (img.shape[0] // 2, img.shape[0] // 2),
                (img.shape[1] // 2, img.shape[1] // 2),
                (0, 0),
            ),
            mode="constant",
        )

    def crop(self, img):
        """Crop padded image back to original size.
        (2H,2W,C) -> (H,W,C)"""
        return img[
            img.shape[0] // 4 : -img.shape[0] // 4,
            img.shape[1] // 4 : -img.shape[1] // 4,
        ]

    def prox(self, x):
        """Apply the proximal operator if one was provided."""
        if self.prox_fn is None:
            return x
        return self.prox_fn(x, self.lam, self.step_size)

    # ----------------------------------------------------------------
    # TODO: Implement the methods below
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
        # TODO
        X = np.fft.fft2(x, axes=(0, 1), norm="ortho")
        y = self.crop(
            np.fft.ifft2(self.H * X, axes=(0, 1), norm="ortho").real
        )
        if self.mask is not None:
            y = y * self.mask
        return y

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
        # TODO
        if self.mask is not None:
            y = y * self.mask
        Y = np.fft.fft2(self.pad(y), axes=(0, 1), norm="ortho")
        return np.fft.ifft2(self.H_conj * Y, axes=(0, 1), norm="ortho").real

    def power_iteration(self, full_shape, num_iters=10):
        """Estimate largest eigenvalue of A*A via power iteration.

        Do NOT include crop/pad — work in the full padded space.
        Repeatedly apply H in frequency domain to a random vector:
            b_{k+1} = real(IFFT(H * FFT(b_k)))
            b_{k+1} = b_{k+1} / ||b_{k+1}||

        After num_iters, estimate the eigenvalue via the Rayleigh
        quotient: eigenvalue = (b^T M b) / (b^T b)
        where M b = real(IFFT(H * FFT(b))).

        Args:
            full_shape: (2H, 2W, C) shape of padded space
            num_iters: Number of iterations (default 10)
        Returns:
            Estimated largest eigenvalue (float)
        """
        b = np.random.randn(*full_shape)
        b /= np.linalg.norm(b)
        for _ in range(num_iters):
            B = np.fft.fft2(b, axes=(0, 1), norm="ortho")
            b = np.fft.ifft2(self.H * B, axes=(0, 1), norm="ortho").real
            b /= np.linalg.norm(b)

        B = np.fft.fft2(b, axes=(0, 1), norm="ortho")
        Mb = np.fft.ifft2(self.H * B, axes=(0, 1), norm="ortho").real
        return float(np.sum(b * Mb) / np.sum(b * b))

    def gd_update(self, x, y):
        """One gradient descent step.

        x_new = x - step_size * A_adjoint(A(x) - y)

        Args:
            x: Current estimate (2H, 2W, C)
            y: Measurement (H, W, C)
        Returns:
            (x_new, loss) where loss = self.loss(error)
        """
        # TODO
        error = self.A(x) - y
        x_new = x - self.step_size * self.A_adjoint(error)
        return x_new, self.loss(error)

    def ista_update(self, x, y):
        """One proximal gradient step."""
        x_new, loss = self.gd_update(x, y)
        return self.prox(x_new), loss

    def fista_update(self, v, t, x, y):
        """One accelerated proximal gradient step."""
        x_new, _ = self.ista_update(v, y)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        v_new = x_new + ((t - 1.0) / t_new) * (x_new - x)
        return v_new, t_new, x_new, self.loss(self.A(x_new) - y)

    def loss(self, error):
        """Compute the data fidelity loss.

        loss = 0.5 * ||error||_2^2

        Args:
            error: Residual A(x) - y
        Returns:
            Scalar loss value (float)
        """
        # TODO
        return float(0.5 * np.sum(error**2))


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
    x = np.random.randn(*full_shape)
    y = np.random.randn(*y_shape)

    Ax = solver.A(x)
    As_y = solver.A_adjoint(y)

    lhs = np.dot(y.ravel(), Ax.ravel())
    rhs = np.dot(As_y.ravel(), x.ravel())
    return lhs, rhs
