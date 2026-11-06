"""Display a Wiener-parameter sweep for the public DiffuserCam example."""

from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np

from lab6_2 import (
    forward,
    match_psf_to_measurement,
    mean_downsample,
    normalize_psf,
    normalize_unit_interval,
    wiener,
)


data_dir = Path(__file__).resolve().parent / "lab6/test_data/shamus-sample"
measurement = np.load(data_dir / "im65.npy")
psf_full = iio.imread(data_dir / "psf.tiff")
reference = np.load(data_dir / "im65_gt.npy")

# The public PSF is 4x the measurement resolution. Match it using block
# averaging, then reduce every image once more to keep the FFT sweep fast.
psf = match_psf_to_measurement(psf_full, measurement)
downsample = 2
measurement = mean_downsample(measurement, downsample)
psf = mean_downsample(psf, downsample)
reference = mean_downsample(reference, downsample)

# The handout specifies normalized inputs and a Wiener sweep in powers of ten.
measurement = normalize_unit_interval(measurement)
# A convolution kernel is normalized by its sum, not its display range. This
# makes H(0) = 1 in every channel and gives tau a meaningful, stable scale.
psf = normalize_psf(psf)
reference = normalize_unit_interval(reference)
synthetic_measurement = forward(reference, psf)
taus = 10.0 ** np.arange(-6, 6)
reconstructions = [
    # Remove the smooth boundary mismatch, then filter at the native size.
    # Neither the measurement nor the PSF is zero-padded in this mode.
    wiener(measurement, psf, tau, boundary="periodic_smooth")
    for tau in taus
]

figure, axes = plt.subplots(3, 4, figsize=(16, 10))
for axis, reconstruction, tau in zip(axes.flat, reconstructions, taus):
    # Keep display data as floats in [0, 1]. Casting the signed Wiener output
    # directly to uint8 wraps negative values and corrupts RGB colors.
    axis.imshow(normalize_unit_interval(reconstruction))
    axis.set_title(f"tau = {tau:g}")
    axis.axis("off")
figure.suptitle("Real DiffuserCam measurement: boundary-conditioned Wiener sweep")
figure.tight_layout()

overview, overview_axes = plt.subplots(1, 4, figsize=(15, 4))
overview_axes[0].imshow(normalize_unit_interval(psf))
overview_axes[0].set_title("PSF")
overview_axes[1].imshow(measurement)
overview_axes[1].set_title("DiffuserCam measurement")
overview_axes[2].imshow(reference)
overview_axes[2].set_title("Lensed reference")
overview_axes[3].imshow(normalize_unit_interval(synthetic_measurement))
overview_axes[3].set_title("Synthetic forward measurement")
for axis in overview_axes:
    axis.axis("off")
overview.tight_layout()

plt.show()
