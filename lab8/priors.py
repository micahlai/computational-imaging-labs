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
import pywt


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

# ----------------------------------------------------------------
# Wavelet transform (provided)
# ----------------------------------------------------------------

def W(z):
    """Forward wavelet transform (image -> wavelet coefficients).

    Uses Daubechies-4 wavelets with 4 levels of decomposition
    and periodization boundary mode.

    Args:
        z: Image array (H, W, C)
    Returns:
        (coeff_array, slices_list) — coeff_array has the same
        shape as z but contains wavelet coefficients; slices_list
        is needed by WT() for the inverse transform.
    """

    coeffs_list, slices_list = [], []
    for c in range(z.shape[2]):
        coeffs = pywt.wavedec2(z[:, :, c], wavelet='db4',
                                mode='periodization', level=4)
        arr, slices = pywt.coeffs_to_array(coeffs)
        coeffs_list.append(arr)
        slices_list.append(slices)
    return np.stack(coeffs_list, axis=-1), slices_list


def WT(z, slices_list):
    """Inverse wavelet transform (wavelet coefficients -> image).

    Args:
        z: Wavelet coefficient array (H, W, C)
        slices_list: Slice info returned by W()
    Returns:
        Reconstructed image (H, W, C)
    """

    recon = []
    for c in range(z.shape[2]):
        coeffs = pywt.array_to_coeffs(
            z[:, :, c], slices_list[c],
            output_format='wavedec2')
        recon.append(pywt.waverec2(
            coeffs, wavelet='db4', mode='periodization'))
    return np.stack(recon, axis=-1)


def prox_wavelet(x, lam, step_size):
    """Wavelet sparsity proximal operator.

    Transform to wavelet domain, soft-threshold the coefficients,
    then transform back:

        prox(x) = WT( soft_thresh( W(x), lam * step_size ) )

    Args:
        x: Input image
        lam: Regularization parameter
        step_size: Step size (1/L)
    Returns:
        Result of applying the proximal operator
    """
    coeffs, slices_list = W(x)
    thresholded = soft_thresh(coeffs, lam * step_size)
    return WT(thresholded, slices_list)


# ----------------------------------------------------------------
# Total variation helpers (provided)
# ----------------------------------------------------------------

def ht3(x, ax, shift, thresh):
    """Haar transform + soft-thresholding (forward step).

    Splits the input along axis `ax` into even/odd samples,
    computes Haar averages and differences, and soft-thresholds
    the differences.

    Args:
        x: Input array (H, W, C)
        ax: Axis to operate on (0 = rows, 1 = columns)
        shift: If True, roll the input by one pixel first
        thresh: Soft-thresholding value
    Returns:
        (w1, w2) — Haar averages and thresholded differences
    """
    C = 1.0 / np.sqrt(2.0)
    if shift:
        x = np.roll(x, -1, axis=ax)
    if ax == 0:
        w1 = C * (x[1::2, :, :] + x[0::2, :, :])
        w2 = soft_thresh(
            C * (x[1::2, :, :] - x[0::2, :, :]), thresh)
    else:
        w1 = C * (x[:, 1::2, :] + x[:, 0::2, :])
        w2 = soft_thresh(
            C * (x[:, 1::2, :] - x[:, 0::2, :]), thresh)
    return w1, w2


def iht3(w1, w2, ax, shift, shape):
    """Inverse Haar transform (inverse of ht3).

    Reconstructs the signal from Haar averages and differences.

    Args:
        w1: Haar averages
        w2: (Thresholded) Haar differences
        ax: Axis (0 or 1)
        shift: Whether ht3 used a shift
        shape: Original shape (H, W, C)
    Returns:
        Reconstructed array with shape `shape`
    """
    C = 1.0 / np.sqrt(2.0)
    y = np.zeros(shape)
    x1 = C * (w1 - w2)
    x2 = C * (w1 + w2)
    if ax == 0:
        y[0::2, :, :] = x1
        y[1::2, :, :] = x2
    else:
        y[:, 0::2, :] = x1
        y[:, 1::2, :] = x2
    if shift:
        y = np.roll(y, 1, axis=ax)
    return y


def tv3dApproxHaar(x, tau):
    """Approximate TV proximal operator via Haar wavelets (provided).

    Applies soft-thresholding in the Haar wavelet domain to
    approximate the TV proximal operator without an inner loop.

    Args:
        x: Input image (H, W, C)
        tau: Regularization strength
    Returns:
        TV-denoised image (H, W, C)
    """
    D = 2
    thresh = D * tau * np.sqrt(2) * 2
    y = np.zeros_like(x)
    for ax in range(2):
        w0, w1 = ht3(x, ax, False, thresh)
        w2, w3 = ht3(x, ax, True, thresh)
        y += iht3(w0, w1, ax, False, x.shape)
        y += iht3(w2, w3, ax, True, x.shape)
    return y / (2 * D)


def prox_tv(x, lam, step_size):
    """Total variation proximal operator.

    Combines non-negativity and TV via proximal average:

        prox(x) = 0.5 * (max(x, 0) + tv3dApproxHaar(x, lam * step))

    Args:
        x: Input image
        lam: Regularization parameter
        step_size: Step size (1/L)
    Returns:
        Result of applying the proximal operator
    """
    return 0.5 * (np.maximum(x,0) + tv3dApproxHaar(x, lam * step_size))
