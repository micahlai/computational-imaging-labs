from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def psf_to_otf(psf: ArrayLike, output_shape: tuple[int, int]) -> NDArray[np.complex128]:
    """Convert a centered PSF to an optical transfer function (OTF).

    Parameters
    ----------
    psf:
        A point spread function with shape ``H x W`` or ``H x W x C`` whose
        optical center is at ``(height // 2, width // 2)``.  A multichannel
        PSF is transformed independently per channel.
    output_shape:
        Spatial shape ``(height, width)`` of the detector measurement.

    Returns
    -------
    numpy.ndarray
        The complex Fourier response with shape ``output_shape``.
    """
    kernel = np.asarray(psf, dtype=np.float64)

    if kernel.ndim not in (2, 3):
        raise ValueError("psf must have shape H x W or H x W x C")
    if len(output_shape) != 2 or any(int(size) != size or size <= 0 for size in output_shape):
        raise ValueError("output_shape must contain two positive integers")
    if kernel.shape[0] > output_shape[0] or kernel.shape[1] > output_shape[1]:
        raise ValueError("psf cannot be larger than the requested output shape")
    if not np.all(np.isfinite(kernel)):
        raise ValueError("psf must contain only finite values")

    padded = np.zeros(output_shape + kernel.shape[2:], dtype=np.float64)
    padded[: kernel.shape[0], : kernel.shape[1], ...] = kernel

    # Move the supplied PSF center to the FFT origin (0, 0).  Rolling by the
    # kernel size, rather than the padded size, also handles a smaller PSF.
    padded = np.roll(padded, -(kernel.shape[0] // 2), axis=0)
    padded = np.roll(padded, -(kernel.shape[1] // 2), axis=1)
    return np.fft.fft2(padded, axes=(0, 1))


def _prepare_psf(
    psf: ArrayLike,
    output_shape: tuple[int, int],
    normalize_psf: bool,
    channels: int | None,
) -> NDArray[np.float64]:
    """Validate a PSF and optionally normalize its DC response to one."""
    kernel = np.asarray(psf, dtype=np.float64)
    if kernel.ndim not in (2, 3):
        raise ValueError("psf must have shape H x W or H x W x C")
    if not np.all(np.isfinite(kernel)):
        raise ValueError("psf must contain only finite values")
    if kernel.shape[0] > output_shape[0] or kernel.shape[1] > output_shape[1]:
        raise ValueError("psf cannot be larger than the measurement")

    if channels is None and kernel.ndim == 3:
        if kernel.shape[2] != 1:
            raise ValueError("a multichannel psf requires a multichannel measurement")
        kernel = kernel[..., 0]
    elif channels is not None and kernel.ndim == 3 and kernel.shape[2] not in (1, channels):
        raise ValueError("psf channels must be one or match the measurement channels")

    if normalize_psf:
        total = kernel.sum(axis=(0, 1), keepdims=kernel.ndim == 3)
        if np.any(np.isclose(total, 0.0)):
            raise ValueError("every psf channel must have a nonzero sum")
        kernel = kernel / total
    elif not np.any(kernel):
        raise ValueError("psf must not be identically zero")

    return kernel


def convolve_with_psf(
    image: ArrayLike,
    psf: ArrayLike,
    *,
    normalize_psf: bool = True,
) -> NDArray[np.float64]:
    """Apply the diffuser-camera forward model using circular convolution.

    ``image`` may be either ``H x W`` or ``H x W x C``.  For a multichannel
    image, the same two-dimensional PSF is applied independently to every
    channel.
    """
    scene = np.asarray(image, dtype=np.float64)
    if scene.ndim not in (2, 3):
        raise ValueError("image must have shape H x W or H x W x C")
    if not np.all(np.isfinite(scene)):
        raise ValueError("image must contain only finite values")

    spatial_shape = scene.shape[:2]
    channels = scene.shape[2] if scene.ndim == 3 else None
    kernel = _prepare_psf(psf, spatial_shape, normalize_psf, channels)
    otf = psf_to_otf(kernel, spatial_shape)
    if scene.ndim == 3 and otf.ndim == 2:
        otf = otf[..., np.newaxis]

    blurred = np.fft.ifft2(np.fft.fft2(scene, axes=(0, 1)) * otf, axes=(0, 1))
    return np.real_if_close(blurred, tol=1000).real


def inverse_convolution(
    measurement: ArrayLike,
    psf: ArrayLike,
    *,
    regularization: float = 1e-3,
    normalize_psf: bool = True,
    clip_range: tuple[float, float] | None = None,
) -> NDArray[np.float64]:
    """Recover a scene from a diffuser-camera measurement.

    This computes the regularized inverse

    ``X_hat = conj(H) * Y / (abs(H)**2 + regularization)``.

    Parameters
    ----------
    measurement:
        The captured ``H x W`` or ``H x W x C`` image.
    psf:
        A centered point spread function with shape ``H x W`` or ``H x W x C``.
        A 2-D PSF is shared by all measurement channels; a multichannel PSF is
        applied channel by channel.  A smaller PSF is zero-padded.
    regularization:
        Nonnegative Tikhonov parameter.  Larger values suppress more noise but
        also smooth the reconstruction.  Set it to zero for an unregularized
        inverse; Fourier coefficients at exactly zero response are then set to
        zero rather than divided by zero.
    normalize_psf:
        If true, divide the PSF by its sum before inversion.  This preserves
        the measurement's constant (DC) brightness for a physical PSF.
    clip_range:
        Optional ``(minimum, maximum)`` applied to the reconstruction.  Use
        ``(0, 1)`` for normalized images or ``(0, 255)`` for 8-bit-scale data.

    Returns
    -------
    numpy.ndarray
        A floating-point reconstruction with the same shape as measurement.
    """
    captured = np.asarray(measurement, dtype=np.float64)
    if captured.ndim not in (2, 3):
        raise ValueError("measurement must have shape H x W or H x W x C")
    if not np.all(np.isfinite(captured)):
        raise ValueError("measurement must contain only finite values")
    if not np.isscalar(regularization) or not np.isfinite(regularization):
        raise ValueError("regularization must be a finite scalar")
    if regularization < 0:
        raise ValueError("regularization must be nonnegative")

    spatial_shape = captured.shape[:2]
    channels = captured.shape[2] if captured.ndim == 3 else None
    kernel = _prepare_psf(psf, spatial_shape, normalize_psf, channels)
    otf = psf_to_otf(kernel, spatial_shape)
    if captured.ndim == 3 and otf.ndim == 2:
        otf = otf[..., np.newaxis]

    spectrum = np.fft.fft2(captured, axes=(0, 1))
    numerator = np.conjugate(otf) * spectrum
    denominator = np.abs(otf) ** 2 + regularization

    if regularization == 0:
        reconstructed_spectrum = np.zeros_like(numerator)
        np.divide(
            numerator,
            denominator,
            out=reconstructed_spectrum,
            where=denominator > np.finfo(np.float64).eps,
        )
    else:
        reconstructed_spectrum = numerator / denominator

    reconstruction = np.fft.ifft2(reconstructed_spectrum, axes=(0, 1))
    reconstruction = np.real_if_close(reconstruction, tol=1000).real

    if clip_range is not None:
        if len(clip_range) != 2 or clip_range[0] > clip_range[1]:
            raise ValueError("clip_range must be an ordered (minimum, maximum) pair")
        reconstruction = np.clip(reconstruction, clip_range[0], clip_range[1])

    return reconstruction


# A descriptive alias for callers that prefer the camera-specific name.
reconstruct_diffuser_image = inverse_convolution


__all__ = [
    "convolve_with_psf",
    "inverse_convolution",
    "psf_to_otf",
    "reconstruct_diffuser_image",
]
