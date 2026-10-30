# Lab 6: inverse-convolution diffuser camera

`lab6.py` implements the circular-convolution forward model and a regularized
Fourier-domain inverse using only NumPy.

```python
import numpy as np

from lab6 import inverse_convolution

# `measurement` is H x W or H x W x C. `psf` is centered and may be
# a shared H x W array or a channel-specific H x W x C array.
reconstruction = inverse_convolution(
    measurement,
    psf,
    regularization=1e-3,
    clip_range=(0.0, 1.0),
)
```

Use a smaller `regularization` for a sharper, noisier result and a larger value
for a smoother, more stable result. Set it to zero only for clean simulated
data whose PSF has no missing Fourier frequencies.

## Public test data

`test_data/dlmd` contains one matched example from the DiffuserCam Lensless
Mirflickr Dataset:

- `lensless_measurement.png` - input to `inverse_convolution`
- `psf.png` - measured calibration PSF
- `lensed_ground_truth.png` - reference image for evaluating the result

The PSF is four times the height and width of the measurement. Downsample it
with 4 x 4 block averaging before reconstruction:

```python
import imageio.v3 as iio

from lab6 import inverse_convolution

measurement = iio.imread("lab6/test_data/dlmd/lensless_measurement.png")
psf_full = iio.imread("lab6/test_data/dlmd/psf.png")

height, width = measurement.shape[:2]
psf = psf_full.reshape(height, 4, width, 4, 3).mean(axis=(1, 3))

reconstruction = inverse_convolution(measurement, psf, regularization=1e-3)
```

See [`test_data/dlmd/README.md`](test_data/dlmd/README.md) for provenance,
license, and checksums.
