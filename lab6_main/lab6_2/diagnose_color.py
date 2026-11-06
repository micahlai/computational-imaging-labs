"""Compare RGB reconstruction with optional luma/chroma regularization.

Run from the repository root:
    python -m lab6_main.lab6_2.diagnose_color --no-show

This is a diagnostic experiment; it does not change the reconstruction defaults.
The YUV transform is a floating-point, zero-offset BT.601-style transform.
Both inversions use the original RGB PSFs. Transforming an RGB PSF into three
independent Y/U/V kernels would be incorrect for channel-dependent PSFs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .diffusercam import (
    match_psf_to_measurement, mean_downsample, normalize_psf,
    normalize_unit_interval, wiener,
)
from .run_example import _load

RGB_TO_YUV = np.array([
    [0.299, 0.587, 0.114],
    [-0.14713, -0.28886, 0.436],
    [0.615, -0.51499, -0.10001],
])
YUV_TO_RGB = np.linalg.inv(RGB_TO_YUV)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tau', type=float, default=1e-6)
    parser.add_argument('--chroma-tau', type=float, default=1e-4)
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'results')
    args = parser.parse_args()
    if not (np.isfinite(args.tau) and np.isfinite(args.chroma_tau)
            and 0 < args.tau <= args.chroma_tau):
        parser.error('require finite 0 < tau <= chroma-tau')

    data_dir = Path(__file__).resolve().parents[1] / 'lab6/test_data'
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    report = {}
    for row, dataset in enumerate(['shamus-sample', 'dlmd']):
        paths = (('im65.npy', 'psf.tiff', 'im65_gt.npy') if row == 0 else
                 ('lensless_measurement.png', 'psf.png', 'lensed_ground_truth.png'))
        measurement, psf, reference = [_load(data_dir / dataset / p) for p in paths]
        psf = mean_downsample(match_psf_to_measurement(psf, measurement), 2)
        psf_sums = psf.sum(axis=(0, 1))
        measurement = normalize_unit_interval(mean_downsample(measurement, 2))
        reference = normalize_unit_interval(mean_downsample(reference, 2))
        psf = normalize_psf(psf)
        low = wiener(measurement, psf, args.tau, boundary='periodic_smooth')
        high = wiener(measurement, psf, args.chroma_tau, boundary='periodic_smooth')

        low_yuv = low @ RGB_TO_YUV.T
        roundtrip = low_yuv @ YUV_TO_RGB.T
        # Use luminance from the detailed inverse and chrominance from the
        # stronger inverse. This is a heuristic, not a joint YUV Wiener solve.
        mixed_yuv = high @ RGB_TO_YUV.T
        mixed_yuv[..., 0] = low_yuv[..., 0]
        mixed = mixed_yuv @ YUV_TO_RGB.T
        np.testing.assert_allclose(roundtrip, low, atol=1e-12)
        np.testing.assert_allclose((mixed @ RGB_TO_YUV.T)[..., 0], low_yuv[..., 0], atol=1e-12)
        if args.chroma_tau == args.tau:
            np.testing.assert_allclose(mixed, low, atol=1e-12)

        # Use exactly the same affine display transform for both reconstructions.
        # This controls exposure/contrast so changes come from chroma processing.
        minimum, scale = low.min(), np.ptp(low)
        displays = [np.clip((a - minimum) / scale, 0, 1) for a in [low, mixed]]
        for col, (display, title) in enumerate(zip(
            displays + [reference],
            [f'RGB Wiener, tau={args.tau:g}',
             f'Same Y; U/V from tau={args.chroma_tau:g}',
             'Lensed reference (separate display scale)'],
        )):
            axes[row, col].imshow(display)
            axes[row, col].set_title(f'{dataset}\n{title}', fontsize=10)
            axes[row, col].axis('off')

        fy = np.fft.fftfreq(low.shape[0])[:, None]
        fx = np.fft.fftfreq(low.shape[1])[None, :]
        high_frequency = np.sqrt(fy**2 + fx**2) > 0.1
        def chroma_power(yuv):
            return float((abs(np.fft.fft2(yuv[..., 1:], axes=(0, 1)))**2)[high_frequency].sum())
        H = np.fft.fft2(psf, axes=(0, 1))
        gain = abs(H) / (abs(H)**2 + args.tau)
        report[dataset] = {
            'tau': args.tau,
            'chroma_tau': args.chroma_tau,
            'psf_channel_sums_relative_to_mean_before_normalization': (psf_sums/psf_sums.mean()).tolist(),
            'raw_reconstruction_mean_rgb': low.mean(axis=(0, 1)).tolist(),
            'negative_fraction_rgb': (low < 0).mean(axis=(0, 1)).tolist(),
            'display_value_of_physical_zero_after_minmax': float(-minimum / scale),
            'wiener_only_white_noise_rms_gain_rgb': np.sqrt((gain**2).mean(axis=(0, 1))).tolist(),
            'rgb_yuv_roundtrip_max_error': float(abs(roundtrip - low).max()),
            'luma_max_change_before_display': float(abs((mixed @ RGB_TO_YUV.T)[..., 0] - low_yuv[..., 0]).max()),
            'high_frequency_chroma_power_ratio_mixed_over_rgb': chroma_power(mixed_yuv)/chroma_power(low_yuv),
            'mixed_display_clipped_fraction': float(np.mean((mixed < minimum) | (mixed > minimum+scale))),
        }
    fig.suptitle('Color-noise experiment: identical display scale within each reconstruction pair')
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.28)
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / 'color_diagnosis.png', dpi=150, bbox_inches='tight')
    (args.output / 'color_diagnosis.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
