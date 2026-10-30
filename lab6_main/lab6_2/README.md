# Lab 6 reimplementation: padded DiffuserCam and Wiener filtering

This version implements the model specified in `lab06_handout.pdf` and the
DiffuserCam lecture:

```text
measurement = crop(psf * scene)
```

Unlike the first `lab6` implementation, this version does not assume circular
convolution at the native sensor size. It centers the scene and PSF in arrays
with twice the height and width, performs FFT convolution, applies
`ifftshift`, and crops the active sensor region. The inverse uses

```text
conj(H) / (abs(H)^2 + tau)
```

and follows the same padding, shifting, and cropping convention.

## Run the public DLMD example

From the repository root, with `conda activate cs4660`:

```bash
python -m lab6_2.run_example \
  --measurement lab6/test_data/dlmd/lensless_measurement.png \
  --psf lab6/test_data/dlmd/psf.png \
  --reference lab6/test_data/dlmd/lensed_ground_truth.png \
  --downsample 2 \
  --save lab6_2/wiener_sweep.png
```

The script automatically block-averages the 4x-resolution DLMD PSF to match
the measurement before applying the optional extra downsampling. By default it
sweeps `tau` from `1e-6` through `1e5` in powers of ten.

## API

```python
from lab6_2 import forward, wiener

synthetic_measurement = forward(reference, psf)
reconstruction = wiener(real_measurement, psf, tau=1e-2)
```

Inputs should be floating-point normalized images with identical spatial
dimensions. Both grayscale and RGB arrays are supported.

## Test

```bash
python -m unittest discover -s lab6_2 -v
```
