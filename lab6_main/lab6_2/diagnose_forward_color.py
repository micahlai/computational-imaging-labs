"""Diagnose RGB gain mismatch between the Shamus capture and forward model.

The fitted gains use this measured/reference pair. They demonstrate a color
calibration mismatch; they are not a calibration validated on other scenes.
No reconstruction defaults or input images are changed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .diffusercam import (
    forward, match_psf_to_measurement, mean_downsample,
    normalize_psf, normalize_unit_interval,
)
from .run_example import _load


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    base = Path(__file__).resolve().parents[1] / 'lab6/test_data/shamus-sample'
    measurement = _load(base / 'im65.npy')
    reference = _load(base / 'im65_gt.npy')
    psf = match_psf_to_measurement(_load(base / 'psf.tiff'), measurement)
    measurement = normalize_unit_interval(mean_downsample(measurement, 2))
    reference = normalize_unit_interval(mean_downsample(reference, 2))
    psf = mean_downsample(psf, 2)
    synthetic = forward(reference, normalize_psf(psf))
    # Least-squares multiplicative fits, without black-level offsets or gamma.
    exposure = float((synthetic * measurement).sum() / (synthetic**2).sum())
    gains = (synthetic * measurement).sum(axis=(0, 1)) / (synthetic**2).sum(axis=(0, 1))
    exposed = exposure * synthetic
    adjusted = gains * synthetic
    # Control: preserving the raw PSF channel gains does not solve this pair.
    common_psf = psf / psf.sum(axis=(0, 1)).mean()
    common_synthetic = forward(reference, common_psf)
    common_exposure = (common_synthetic * measurement).sum() / (common_synthetic**2).sum()

    def rmse(image):
        return float(np.sqrt(np.mean((image - measurement)**2)))

    report = {
        'scope': 'Shamus pair, 2x downsampling, same input preprocessing as testlab6.py',
        'real_mean_rgb': measurement.mean(axis=(0, 1)).tolist(),
        'synthetic_mean_rgb': synthetic.mean(axis=(0, 1)).tolist(),
        'real_mean_blue_over_green': float(measurement[..., 2].mean()/measurement[..., 1].mean()),
        'synthetic_mean_blue_over_green': float(synthetic[..., 2].mean()/synthetic[..., 1].mean()),
        'shared_exposure_fit': exposure,
        'rgb_gain_fit_including_exposure': gains.tolist(),
        'rgb_gain_fit_relative_to_green': (gains/gains[1]).tolist(),
        'channel_spatial_correlations': [float(np.corrcoef(measurement[..., c].ravel(), synthetic[..., c].ravel())[0, 1]) for c in range(3)],
        'rmse_after_shared_exposure': rmse(exposed),
        'rmse_after_rgb_gain_fit': rmse(adjusted),
        'rmse_preserving_raw_psf_gains_with_fitted_shared_exposure': rmse(common_exposure*common_synthetic),
        'limitation': 'Fits this pair only; cannot distinguish PSF throughput, camera white balance, transfer functions, or reference processing.',
    }
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for ax, image, title in zip(axes, [measurement, exposed, adjusted], [
        'Real measurement', 'Synthetic: one exposure factor', 'Synthetic: fitted R/G/B gains',
    ]):
        ax.imshow(np.clip(image, 0, 1))
        ax.set_title(title)
        ax.axis('off')
    fig.suptitle('Forward-model color comparison — same 0–1 display scale')
    fig.tight_layout()
    output = Path(__file__).parent / 'results'
    output.mkdir(exist_ok=True)
    fig.savefig(output / 'forward_color_diagnosis.png', dpi=150, bbox_inches='tight')
    (output / 'forward_color_diagnosis.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
