# Lab 6: DiffuserCam and Wiener filtering

The forward model follows the lab handout:

```text
measurement = crop(psf * scene)
```

It uses doubled arrays for linear convolution and crops to the sensor region.
The Wiener inverse uses `conj(H) / (abs(H)**2 + tau)`, with four choices for
handling the measurement boundaries.

## Reduce lines at low tau without padding

```python
from lab6_main.lab6_2 import normalize_psf, wiener

reconstruction = wiener(
    measurement, normalize_psf(psf), tau=1e-6, boundary="periodic_smooth"
)
```

`periodic_smooth` subtracts a smooth, zero-mean component determined by the
measurement's opposite-border differences. It then applies the Wiener filter
at the original sensor size. **Neither the measurement nor the PSF is padded.**
The PSF is shifted so its center is the FFT origin; its values are unchanged.

This uses [Moisan's periodic plus smooth decomposition](https://helios2.mi.parisdescartes.fr/~moisan/p+s/)
(L. Moisan, *Journal of Mathematical Imaging and Vision*, 39, 161–179, 2011).
Simply removing padding is insufficient: the FFT assumes periodic repetition,
so unequal top/bottom or left/right borders still create a discontinuity.
Nearest-edge extension moves that seam but does not generally eliminate it.

The examples now default to `periodic_smooth`. The API retains `boundary="zero"`
as its default for compatibility with the original handout implementation;
select the new mode explicitly in your own calls.

| Boundary | Measurement treatment | PSF treatment |
| --- | --- | --- |
| `zero` | Zero padding to double size | Zero padding |
| `edge` | Nearest-edge extension to double size | Zero padding |
| `circular` | Native-size FFT | Native-size FFT |
| `periodic_smooth` | Smooth boundary correction, native-size FFT | Native-size FFT |

On both included captures at 2x downsampling, the new mode visibly suppresses
the horizontal bands at `tau=1e-8` and `1e-6`. As a diagnostic, at `tau=1e-6`
the fraction of high-frequency energy near the vertical Fourier axis changes:

| Capture | Zero | Edge | Circular | Periodic + smooth |
| --- | ---: | ---: | ---: | ---: |
| Shamus | 80.6% | 37.5% | 61.3% | 3.5% |
| DLMD | 82.1% | 24.7% | 36.2% | 4.0% |

Here the fraction is `sum(abs(FFT(x))**2)` over `abs(f_y) > 0.1` and
`abs(f_x) < 0.02`, divided by energy over all `abs(f_y) > 0.1`, summed across
RGB channels. Frequencies are cycles/pixel and `x` is the raw reconstruction.
This measures directional concentration, not reconstruction accuracy or Gaussianity.

Limitations: low tau still amplifies poorly measured frequencies. The remaining
noise can look more grain-like but is not guaranteed to be white or independent;
linear filtering of Gaussian noise generally produces correlated Gaussian noise.
The boundary correction may remove genuine slow scene variation. Native-size
filtering assumes circular convolution and is an approximation to the cropped
linear forward model, not an exact inverse of it.

## Run and compare

From the repository root, with `conda activate cs4660`:

```bash
python -m lab6_main.lab6_2.run_example \
  --measurement lab6_main/lab6/test_data/shamus-sample/im65.npy \
  --psf lab6_main/lab6/test_data/shamus-sample/psf.tiff \
  --downsample 2 --taus 1e-8 1e-6 1e-4 \
  --compare-boundaries --no-show \
  --save lab6_main/lab6_2/results/shamus_boundaries.png
```

Omit `--compare-boundaries` for just the new mode, or use `--boundary edge`
(or `zero`, `circular`) to select an older treatment. Each panel uses its own
min/max for display; numerical comparisons should use the raw arrays.

For the public DLMD example:

```bash
python -m lab6_main.lab6_2.run_example \
  --measurement lab6_main/lab6/test_data/dlmd/lensless_measurement.png \
  --psf lab6_main/lab6/test_data/dlmd/psf.png \
  --reference lab6_main/lab6/test_data/dlmd/lensed_ground_truth.png \
  --downsample 2 --taus 1e-8 1e-6 1e-4
```

The script accepts image files and `.npy` arrays. It block-averages the PSF to
match the measurement before optional extra downsampling. Without `--taus`,
it sweeps from `1e-6` through `1e5`. With `--reference`, it also reconstructs a
synthetic measurement from the padded linear forward model.

Inputs to the API must have matching spatial dimensions. Grayscale, RGB,
shared grayscale PSFs, and per-channel PSFs are supported. Normalize PSFs by
sum for a consistent tau scale; normalize reconstructions only for display.

## Test

```bash
python -m unittest discover -s lab6_main/lab6_2 -v
```

## Diagnose color: RGB versus YUV

Run the reproducible color experiment (both included captures):

```bash
python -m lab6_main.lab6_2.diagnose_color --no-show
```

This writes `results/color_diagnosis.png` and `results/color_diagnosis.json`.
It leaves the reconstruction defaults unchanged. Use `--tau` for the detailed
reconstruction and `--chroma-tau` for the smoother color reconstruction.

Observed contributors to discoloring:

- **Color noise:** the RGB PSFs differ. Even a common tau produces different
  transfer functions and noise amplification per channel. At `tau=1e-6`, the
  Wiener filters alone have white-noise RMS gains of approximately R=322,
  G=294, B=328. These are filter gains, not measured sensor-noise estimates;
  the boundary correction also changes the noise statistics.
- **Gray wash from display:** about 42–45% of Shamus reconstruction values in
  each channel are negative at `tau=1e-6`. Global min/max display normalization
  raises physical zero to gray, reducing apparent saturation and contrast.
  This common affine transform does not itself swap RGB channels. Clipping
  negative values and applying a shared exposure is another display choice,
  but it cannot recover the missing scene or remove color noise.
- **Possible color-gain calibration issue:** before per-channel normalization,
  the PSF sums relative to their average are `[1.013, 0.824, 1.164]` for RGB.
  Normalizing each sum to one discards these relative gains. If they represent
  calibrated throughput, that changes recovered color balance. If they instead
  include calibration-source color or camera processing, preserving them could
  also be wrong. The available captures do not establish a white-balance or
  radiometric calibration, so this experiment does not automatically correct it.
- **Boundary approximation:** subtracting the smooth component separately in
  RGB can change broad color variation as well as brightness. The prior line
  fix was not a color-calibration fix.

RGB contains red, green and blue intensities. YUV re-expresses them as a
weighted brightness component Y and two color-difference components U/V
([conversion documentation](https://scikit-image.org/docs/stable/api/skimage.color.html#skimage.color.rgb2yuv)).
For gamma-encoded RGB, Y is technically luma, not physical luminance. The
capture transfer function is not established here. Merely converting RGB to
YUV and back changes these reconstructions by less than `3e-15`; it fixes
nothing by itself.

The experiment takes Y from the RGB Wiener reconstruction at `tau=1e-6` and
U/V from a second RGB Wiener reconstruction at `tau=1e-4`, then converts back.
Brightness is unchanged to numerical precision before display. High-frequency
chroma power (radial frequency > 0.1 cycles/pixel) falls by about 98.6% on Shamus
and 98.8% on DLMD. This includes true fine color detail as well as noise; it is
not a color-accuracy score. The comparison applies exactly the same display
scale to both reconstructions. The lensed reference has a separate scale.

Do not independently deconvolve Y, U and V using a color-converted PSF.
If `A` maps RGB to YUV, the correct frequency-domain operator is
`A @ diag(H_R, H_G, H_B) @ inv(A)`. It generally couples the channels; it is
only diagonal with a shared PSF. A proper joint YUV inverse must account for
that coupling and the transformed noise covariance. The simpler experiment
here keeps both physical inversions in RGB and combines their outputs.

## Real versus synthetic forward-measurement color

The RGB/YUV experiment above addresses reconstruction noise. A separate issue
is the bluer real Shamus measurement versus the greener synthetic measurement,
which is present **before** Wiener inversion and is independent of tau.

Reproduce this diagnosis with:

```bash
python -m lab6_main.lab6_2.diagnose_forward_color --no-show
```

This writes `results/forward_color_diagnosis.png` and its JSON measurements.
After the example's preprocessing, the mean B/G ratio is 1.094 in the real
measurement versus 0.938 in the synthetic one. Corresponding RGB spatial
correlations are 0.994–0.995, consistent with a gain mismatch rather than a
wrong spatial convolution. Fitting one shared exposure gives an RMSE of 0.0459;
fitting separate RGB gains gives 0.0238 (about 48% lower). The fitted gains,
relative to green, are `[0.9124, 1, 1.1670]`. Including overall exposure, they
are `[2.4175, 2.6496, 3.0922]` for these preprocessed arrays.

Per-channel PSF normalization removes throughput ratios, but keeping the raw
ratios alone does not fix this pair: with a fitted shared exposure its RMSE is
0.0509, slightly worse. Other possible contributors include different camera
white balance, calibration illumination and reference-image processing. These
cannot be distinguished from this one pair. RGB/YUV conversion does not
supply the missing calibration.

The plotted adjusted synthetic image is an **in-sample gain fit**, not an
independently calibrated prediction. Use a neutral calibration target or
multiple calibration pairs to estimate reusable RGB gains and black levels.
Keep the same calibrated gains in the forward PSF and inverse model if applying
them later. The diagnosis does not change the current model defaults.
