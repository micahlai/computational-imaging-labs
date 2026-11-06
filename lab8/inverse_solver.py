"""InverseSolver for DiffuserCam reconstruction — Lab 8 starter code, build from Lab 7.

Fill in the TODO
"""

import numpy as np


class InverseSolver:
    """Reconstructs DiffuserCam images via iterative optimization.

    Supports gradient descent, ISTA, and FISTA algorithms with a
    pluggable proximal operator.

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

    def __init__(self, psf, algorithm='gradient_descent',
                 prox_fn=None, num_iters=200,
                 step_size=None, lam=0.0, mask=None):
        assert algorithm in [
            'gradient_descent', 'ista', 'fista'
        ], f"Unknown algorithm: {algorithm}"

        self.h = psf / np.linalg.norm(psf)
        self.H = np.fft.fft2(
            np.fft.ifftshift(self.pad(self.h), axes=(0, 1)),
            axes=(0, 1), norm='ortho')
        self.H_conj = np.conj(self.H)

        self.mask = mask

        if step_size is None:
            full_shape = (self.h.shape[0] * 2,
                          self.h.shape[1] * 2,
                          self.h.shape[2])
            L = self.power_iteration(full_shape)
            self.step_size = 1.0 / L
        else:
            self.step_size = step_size

        self.algorithm = algorithm
        self.prox_fn = prox_fn
        self.num_iters = num_iters
        self.lam = lam

    def __call__(self, y):
        """Run reconstruction on measurement y."""
        x = np.zeros((y.shape[0] * 2, y.shape[1] * 2,
                       y.shape[2]))
        v = np.zeros_like(x)
        t = 1.0

        self.losses = []
        for i in range(self.num_iters):
            if self.algorithm == 'gradient_descent':
                x, loss = self.gd_update(x, y)
            elif self.algorithm == 'ista':
                x, loss = self.ista_update(x, y)
            elif self.algorithm == 'fista':
                v, t, x, loss = self.fista_update(
                    v, t, x, y)
            self.losses.append(loss)

        return self.crop(x)

    # ----------------------------------------------------------------
    # Provided helper methods
    # ----------------------------------------------------------------

    def pad(self, img):
        """Zero-pad image to 2x size."""
        return np.pad(img,
            ((img.shape[0] // 2, img.shape[0] // 2),
             (img.shape[1] // 2, img.shape[1] // 2),
             (0, 0)), mode='constant')

    def crop(self, img):
        """Crop padded image back to original size."""
        return img[img.shape[0] // 4:-img.shape[0] // 4,
                   img.shape[1] // 4:-img.shape[1] // 4]

    def prox(self, x):
        """Apply the proximal operator if one was provided."""
        if self.prox_fn is None:
            return x
        return self.prox_fn(x, self.lam, self.step_size)

    # ----------------------------------------------------------------
    # Solutions
    # ----------------------------------------------------------------

    def A(self, x):
        """Forward model: A(x) = crop(IFFT(H * FFT(x)))."""
        X = np.fft.fft2(x, axes=(0, 1), norm='ortho')
        out = np.real(self.crop(
            np.fft.ifft2(X * self.H,
                          axes=(0, 1), norm='ortho')))
        if self.mask is not None:
            out = out * self.mask
        return out

    def A_adjoint(self, y):
        """Adjoint: A*(y) = IFFT(conj(H) * FFT(pad(y)))."""
        if self.mask is not None:
            y = y * self.mask
        Y = np.fft.fft2(self.pad(y),
                         axes=(0, 1), norm='ortho')
        return np.real(
            np.fft.ifft2(Y * self.H_conj,
                          axes=(0, 1), norm='ortho'))

    def power_iteration(self, full_shape, num_iters=10):
        """Estimate largest eigenvalue of A*A."""
        bk = np.random.randn(*full_shape)
        for _ in range(num_iters):
            X = np.fft.fft2(bk, axes=(0, 1), norm='ortho')
            bk1 = np.real(np.fft.ifft2(
                X * self.H, axes=(0, 1), norm='ortho'))
            bk = bk1 / np.linalg.norm(bk1)
        X = np.fft.fft2(bk, axes=(0, 1), norm='ortho')
        Mbk = np.real(np.fft.ifft2(
            X * self.H, axes=(0, 1), norm='ortho'))
        return (np.dot(bk.ravel(), Mbk.ravel()) /
                np.dot(bk.ravel(), bk.ravel()))

    def gd_update(self, x, y):
        """One gradient descent step."""
        error = self.A(x) - y
        x = x - self.step_size * self.A_adjoint(error)
        return x, self.loss(error)

    def ista_update(self, x, y):
        """One ISTA (proximal gradient) step.

        w = x - step_size * A_adjoint(A(x) - y)
        x_new = prox(w)

        Args:
            x: Current estimate (2H, 2W, C)
            y: Measurement (H, W, C)
        Returns:
            (x_new, loss) where loss = self.loss(error)
        """
        # x_new, loss = self.gd_update(x, y)

        error = self.A(x)-y
        w = x - self.step_size * self.A_adjoint(error)
        x_new = self.prox(w)
        return x_new, self.loss(error)

    # def fista_update(self, v, t, x, y):
    #     """One FISTA (accelerated proximal gradient) step.

    #     w = v - step_size * A_adjoint(A(v) - y)
    #     x_new = prox(w)
    #     t_new = (1 + sqrt(1 + 4*t^2)) / 2
    #     beta = (t - 1) / t_new
    #     v_new = x_new + beta * (x_new - x)

    #     Args:
    #         v: Momentum variable (2H, 2W, C)
    #         t: Acceleration parameter (float)
    #         x: Current estimate (2H, 2W, C)
    #         y: Measurement (H, W, C)
    #     Returns:
    #         (v_new, t_new, x_new, loss) tuple
    #     """
    #     x_new, _ = self.ista_update(v, y)
    #     t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
    #     v_new = x_new + ((t - 1.0) / t_new) * (x_new - x)
    #     return v_new, t_new, x_new, self.loss(self.A(x_new) - y)

    def fista_update(self, v, t, x, y):
            """One FISTA (accelerated proximal gradient) step.
    
            w = v - step_size * A_adjoint(A(v) - y)
            x_new = prox(w)
            t_new = (1 + sqrt(1 + 4*t^2)) / 2
            beta = (t - 1) / t_new
            v_new = x_new + beta * (x_new - x)
    
            Args:
                v: Momentum variable (2H, 2W, C)
                t: Acceleration parameter (float)
                x: Current estimate (2H, 2W, C)
                y: Measurement (H, W, C)
            Returns:
                (v_new, t_new, x_new, loss) tuple
            """
            error = self.A(v) - y
            x_new = self.prox(v - self.step_size * self.A_adjoint(error))
            t_new = (1+np.sqrt(1+4*t**2))/2
            beta = (t-1) / t_new
            v_new = x_new + beta * (x_new - x)
            return v_new, t_new, x_new, self.loss(error)

    def loss(self, error):
        """Compute 0.5 * ||error||^2."""
        return 0.5 * np.linalg.norm(error) ** 2


def adjoint_test(solver, y_shape):
    """Verify <y, Ax> == <A*y, x> for random x, y."""
    full_shape = (y_shape[0] * 2, y_shape[1] * 2,
                  y_shape[2])
    x = np.random.randn(*full_shape)
    y = np.random.randn(*y_shape)

    Ax = solver.A(x)
    As_y = solver.A_adjoint(y)

    lhs = np.dot(y.ravel(), Ax.ravel())
    rhs = np.dot(As_y.ravel(), x.ravel())
    return lhs, rhs