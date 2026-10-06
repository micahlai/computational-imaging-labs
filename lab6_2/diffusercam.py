"""Padded DiffuserCam forward model and Wiener deconvolution.

The implementation follows the model

    y = crop(h * x)

where ``*`` is linear convolution.  Images are centered in arrays with twice
their spatial size before FFT convolution so circular wraparound does not enter
the active sensor region.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatImage = NDArray[np.float64]


def _as_image(image: ArrayLike, name: str) -> FloatImage:
    """Return a validated H x W or H x W x C floating-point image."""
    result = np.asarray(image, dtype=np.float64)
    if result.ndim not in (2, 3):
        raise ValueError(f"{name} must have shape H x W or H x W x C")
    if result.shape[0] == 0 or result.shape[1] == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _validate_pair(image: FloatImage, psf: FloatImage) -> None:
    if image.shape[:2] != psf.shape[:2]:
        raise ValueError("image and psf must have the same spatial shape")

    if image.ndim == 2 and psf.ndim == 3 and psf.shape[2] != 1:
        raise ValueError("a grayscale image needs a grayscale or one-channel psf")
    if image.ndim == 3 and psf.ndim == 3 and psf.shape[2] not in (1, image.shape[2]):
        raise ValueError("psf channels must be one or match the image channels")


def _broadcast_otf(otf: NDArray[np.complex128], image: FloatImage) -> NDArray[np.complex128]:
    """Make a grayscale OTF broadcast over an H x W x C image."""
    if image.ndim == 3 and otf.ndim == 2:
        return otf[..., np.newaxis]
    if image.ndim == 2 and otf.ndim == 3:
        return otf[..., 0]
    return otf


def pad(image: ArrayLike) -> FloatImage:
    """Center an image in a zero array with twice its height and width.

    For even dimensions this is exactly the half-size padding specified in the
    lab handout.  The asymmetric one-pixel case keeps the result exactly 2x for
    odd dimensions as well.
    """
    image_array = _as_image(image, "image")
    height, width = image_array.shape[:2]
    pad_width = [
        (height // 2, height - height // 2),
        (width // 2, width - width // 2),
    ]
    if image_array.ndim == 3:
        pad_width.append((0, 0))
    return np.pad(image_array, pad_width, mode="constant")


def crop(image: ArrayLike, output_shape: tuple[int, int] | None = None) -> FloatImage:
    """Return a centered sensor-sized crop from a padded image.

    If ``output_shape`` is omitted, half of each spatial dimension is retained.
    """
    # Preserve complex values until callers explicitly take abs() or real().
    image_array = np.asarray(image)
    if image_array.ndim not in (2, 3):
        raise ValueError("image must have shape H x W or H x W x C")
    if image_array.shape[0] == 0 or image_array.shape[1] == 0:
        raise ValueError("image cannot be empty")
    if not np.all(np.isfinite(image_array)):
        raise ValueError("image must contain only finite values")
    padded_height, padded_width = image_array.shape[:2]

    if output_shape is None:
        output_shape = (padded_height // 2, padded_width // 2)
    if len(output_shape) != 2 or any(int(size) != size or size <= 0 for size in output_shape):
        raise ValueError("output_shape must contain two positive integers")

    height, width = int(output_shape[0]), int(output_shape[1])
    if height > padded_height or width > padded_width:
        raise ValueError("output_shape cannot be larger than the image")

    row_start = (padded_height - height) // 2
    col_start = (padded_width - width) // 2
    return image_array[row_start : row_start + height, col_start : col_start + width, ...]


def forward(scene: ArrayLike, psf: ArrayLike) -> FloatImage:
    """Simulate a cropped DiffuserCam measurement with padded FFT convolution.

    Both inputs must have the same spatial shape.  A 2-D PSF is shared by all
    color channels; an H x W x C PSF is applied channel by channel.
    """
    scene_array = _as_image(scene, "scene")
    psf_array = _as_image(psf, "psf")
    _validate_pair(scene_array, psf_array)

    scene_spectrum = np.fft.fft2(pad(scene_array), axes=(0, 1))
    psf_spectrum = np.fft.fft2(pad(psf_array), axes=(0, 1))
    psf_spectrum = _broadcast_otf(psf_spectrum, scene_array)

    padded_measurement = np.fft.ifft2(
        scene_spectrum * psf_spectrum,
        axes=(0, 1),
    )
    padded_measurement = np.fft.ifftshift(padded_measurement, axes=(0, 1))
    return np.abs(crop(padded_measurement, scene_array.shape[:2]))


def naive_deconvolution(measurement: ArrayLike, psf: ArrayLike) -> FloatImage:
    """Apply unregularized frequency-domain deconvolution.

    This intentionally implements Y/H.  Near-zero PSF frequencies can therefore
    produce extremely large or non-finite values, illustrating why Wiener
    regularization is needed for real measurements.
    """
    measurement_array = _as_image(measurement, "measurement")
    psf_array = _as_image(psf, "psf")
    _validate_pair(measurement_array, psf_array)

    measurement_spectrum = np.fft.fft2(pad(measurement_array), axes=(0, 1))
    psf_spectrum = np.fft.fft2(pad(psf_array), axes=(0, 1))
    psf_spectrum = _broadcast_otf(psf_spectrum, measurement_array)

    with np.errstate(divide="ignore", invalid="ignore"):
        reconstruction_spectrum = measurement_spectrum / psf_spectrum

    padded_reconstruction = np.fft.ifft2(reconstruction_spectrum, axes=(0, 1))
    padded_reconstruction = np.fft.ifftshift(padded_reconstruction, axes=(0, 1))
    return np.real(crop(padded_reconstruction, measurement_array.shape[:2]))


def wiener(measurement: ArrayLike, psf: ArrayLike, tau: float) -> FloatImage:
    """Reconstruct an image with a scalar-regularized Wiener filter.

    The filter is ``conj(H) / (abs(H)**2 + tau)``.  ``tau`` must be positive;
    use :func:`naive_deconvolution` to demonstrate the unregularized case.
    """
    measurement_array = _as_image(measurement, "measurement")
    psf_array = _as_image(psf, "psf")
    _validate_pair(measurement_array, psf_array)

    if not np.isscalar(tau) or not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be a positive finite scalar")

    measurement_spectrum = np.fft.fft2(pad(measurement_array), axes=(0, 1))
    psf_spectrum = np.fft.fft2(pad(psf_array), axes=(0, 1))
    psf_spectrum = _broadcast_otf(psf_spectrum, measurement_array)

    wiener_filter = np.conjugate(psf_spectrum) / (np.abs(psf_spectrum) ** 2 + tau)
    padded_reconstruction = np.fft.ifft2(
        wiener_filter * measurement_spectrum,
        axes=(0, 1),
    )
    padded_reconstruction = np.fft.ifftshift(padded_reconstruction, axes=(0, 1))
    return np.real(crop(padded_reconstruction, measurement_array.shape[:2]))


def mean_downsample(image: ArrayLike, factor: int) -> FloatImage:
    """Downsample spatial dimensions with non-overlapping block averaging."""
    image_array = _as_image(image, "image")
    if not isinstance(factor, (int, np.integer)) or factor <= 0:
        raise ValueError("factor must be a positive integer")
    height, width = image_array.shape[:2]
    if height % factor or width % factor:
        raise ValueError("image dimensions must be divisible by factor")

    trailing_shape = image_array.shape[2:]
    reshaped = image_array.reshape(
        height // factor,
        factor,
        width // factor,
        factor,
        *trailing_shape,
    )
    return reshaped.mean(axis=(1, 3))


def match_psf_to_measurement(psf: ArrayLike, measurement: ArrayLike) -> FloatImage:
    """Block-average a higher-resolution PSF to the measurement resolution."""
    psf_array = _as_image(psf, "psf")
    measurement_array = _as_image(measurement, "measurement")
    psf_height, psf_width = psf_array.shape[:2]
    height, width = measurement_array.shape[:2]

    if (psf_height, psf_width) == (height, width):
        return psf_array.copy()
    if psf_height % height or psf_width % width:
        raise ValueError("psf dimensions must be integer multiples of the measurement")

    row_factor = psf_height // height
    col_factor = psf_width // width
    if row_factor != col_factor:
        raise ValueError("psf must have the same integer scale factor on both axes")
    return mean_downsample(psf_array, row_factor)


def normalize_unit_interval(image: ArrayLike) -> FloatImage:
    """Map an image's global finite range to [0, 1] without color wrapping."""
    image_array = _as_image(image, "image")
    minimum = float(image_array.min())
    maximum = float(image_array.max())
    if np.isclose(minimum, maximum):
        return np.zeros_like(image_array)
    return (image_array - minimum) / (maximum - minimum)


__all__ = [
    "crop",
    "forward",
    "match_psf_to_measurement",
    "mean_downsample",
    "naive_deconvolution",
    "normalize_unit_interval",
    "pad",
    "wiener",
]
