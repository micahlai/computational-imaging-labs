"""Display a Wiener-parameter sweep for the public DiffuserCam example."""

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np

from lab6_2 import (
    match_psf_to_measurement,
    mean_downsample,
    normalize_unit_interval,
    wiener,
)


measurement = iio.imread("lab6/test_data/dlmd/lensless_measurement.png")
psf_full = iio.imread("lab6/test_data/dlmd/psf.png")
reference = iio.imread("lab6/test_data/dlmd/lensed_ground_truth.png")

# The public PSF is 4x the measurement resolution. Match it using block
# averaging, then reduce every image once more to keep the FFT sweep fast.
psf = match_psf_to_measurement(psf_full, measurement)
downsample = 2
measurement = mean_downsample(measurement, downsample)
psf = mean_downsample(psf, downsample)
reference = mean_downsample(reference, downsample)

# The handout specifies normalized inputs and a Wiener sweep in powers of ten.
measurement = normalize_unit_interval(measurement)
psf = normalize_unit_interval(psf)
reference = normalize_unit_interval(reference)
taus = 10.0 ** np.arange(-6, 6)
reconstructions = [wiener(measurement, psf, tau) for tau in taus]

figure, axes = plt.subplots(3, 4, figsize=(16, 10))
for axis, reconstruction, tau in zip(axes.flat, reconstructions, taus):
    # Keep display data as floats in [0, 1]. Casting the signed Wiener output
    # directly to uint8 wraps negative values and corrupts RGB colors.
    axis.imshow(normalize_unit_interval(reconstruction))
    axis.set_title(f"tau = {tau:g}")
    axis.axis("off")
figure.suptitle("Real DiffuserCam measurement: Wiener parameter sweep")
figure.tight_layout()

overview, overview_axes = plt.subplots(1, 3, figsize=(15, 4))
overview_axes[0].imshow(psf)
overview_axes[0].set_title("PSF")
overview_axes[1].imshow(measurement)
overview_axes[1].set_title("DiffuserCam measurement")
overview_axes[2].imshow(reference)
overview_axes[2].set_title("Lensed reference")
for axis in overview_axes:
    axis.axis("off")
overview.tight_layout()

plt.show()
