"""Prior / proximal operator functions — Lab 8 starter code.

These functions are proximal operators that can be passed to
InverseSolver as the prox_fn argument. Each has the signature:

    prox_fn(x, lam, step_size) -> x_new

Fill in the functions marked TODO to complete the lab.

Usage:
    from priors import prox_nonneg, prox_tv
    from inverse_solver import InverseSolver

    solver = InverseSolver(psf, algorithm='fista',
                           prox_fn=prox_tv, lam=1e-4)
    x_hat = solver(y)
"""

import numpy as np
# import pywt


# ----------------------------------------------------------------
# Soft thresholding
# ----------------------------------------------------------------

def soft_thresh(z, threshold):
    """Soft-thresholding operator.

    Shrinks all values toward zero by `threshold`. Values with
    magnitude less than `threshold` become exactly zero:

        soft(z, t) = sign(z) * max(|z| - t, 0)

    Args:
        z: Input array
        threshold: Threshold value (>= 0)
    Returns:
        Soft-thresholded array (same shape as z)
    """
    return np.sign(z) * np.maximum(np.abs(z) - threshold, 0)


# ----------------------------------------------------------------
# Proximal operators
# ----------------------------------------------------------------


def prox_native(x, lam, step_size):
    """Native (pixel-domain) sparsity proximal operator.

    Combines non-negativity and L1 sparsity using a proximal
    average:

        prox(x) = 0.5 * (max(x, 0) + soft_thresh(x, lam * step))

    Args:
        x: Input image
        lam: Regularization parameter
        step_size: Step size (1/L)
    Returns:
        Result of applying the proximal operator
    """

    # return 0.5 * (np.maximum(x, 0) + soft_thresh(x, lam * step_size))
    return np.maximum(x - lam * step_size, 0)



def prox_nonneg(x, lam, step_size):
    """Non-negativity proximal operator: max(x, 0).

    This is provided as a reference — it was already used in Lab 7.
    """
    return np.maximum(x, 0)
