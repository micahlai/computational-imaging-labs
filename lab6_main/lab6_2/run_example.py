"""Run the DiffuserCam model and a boundary-aware Wiener-parameter sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np

try:
    from .diffusercam import (
        forward,
        match_psf_to_measurement,
        mean_downsample,
        normalize_psf,
        normalize_unit_interval,
        wiener,
    )
except ImportError:  # Allow `python lab6_2/run_example.py ...`.
    from lab6_main.lab6_2.diffusercam import (
        forward,
        match_psf_to_measurement,
        mean_downsample,
        normalize_psf,
        normalize_unit_interval,
        wiener,
    )


DEFAULT_TAUS = tuple(10.0**power for power in range(-6, 6))
BOUNDARIES = ("zero", "edge", "circular", "periodic_smooth")


def _load(path: Path) -> np.ndarray:
    image = np.load(path, allow_pickle=False) if path.suffix.lower() == ".npy" else iio.imread(path)
    return np.asarray(image, dtype=np.float64)


def _show_image(axis: plt.Axes, image: np.ndarray, title: str) -> None:
    display = normalize_unit_interval(image)
    if display.ndim == 2 or display.shape[2] == 1:
        axis.imshow(np.squeeze(display), cmap="gray", vmin=0.0, vmax=1.0)
    else:
        axis.imshow(display)
    axis.set_title(title)
    axis.axis("off")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement", required=True, type=Path)
    parser.add_argument("--psf", required=True, type=Path)
    parser.add_argument("--reference", type=Path, help="optional lensed ground truth")
    parser.add_argument(
        "--boundary", choices=BOUNDARIES, default="periodic_smooth",
        help="inverse boundary handling (default: periodic_smooth, no padding)",
    )
    parser.add_argument(
        "--compare-boundaries", action="store_true",
        help="show zero, edge, circular, and periodic_smooth for every tau",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="additional block-average factor applied after matching PSF resolution",
    )
    parser.add_argument(
        "--taus",
        nargs="+",
        type=float,
        default=DEFAULT_TAUS,
        help="positive Wiener regularization values",
    )
    parser.add_argument("--save", type=Path, help="save the reconstruction sweep figure")
    parser.add_argument("--no-show", action="store_true", help="do not open plot windows")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.downsample <= 0:
        raise ValueError("--downsample must be positive")
    if any(not np.isfinite(tau) or tau <= 0 for tau in args.taus):
        raise ValueError("all --taus values must be positive and finite")

    measurement = _load(args.measurement)
    psf = match_psf_to_measurement(_load(args.psf), measurement)
    reference = _load(args.reference) if args.reference else None

    if reference is not None and reference.shape[:2] != measurement.shape[:2]:
        raise ValueError("reference and measurement must have the same spatial shape")

    if args.downsample > 1:
        measurement = mean_downsample(measurement, args.downsample)
        psf = mean_downsample(psf, args.downsample)
        if reference is not None:
            reference = mean_downsample(reference, args.downsample)

    measurement = normalize_unit_interval(measurement)
    psf = normalize_psf(psf)
    if reference is not None:
        reference = normalize_unit_interval(reference)

    synthetic = forward(reference, psf) if reference is not None else None
    boundaries = BOUNDARIES if args.compare_boundaries else (args.boundary,)
    sources = [(measurement, "Real")]
    if synthetic is not None:
        sources.append((synthetic, "Synthetic"))
    reconstruction_rows = [
        (f"{label}, {boundary}", [wiener(source, psf, tau, boundary=boundary) for tau in args.taus])
        for boundary in boundaries
        for source, label in sources
    ]

    overview_images = [(psf, "PSF"), (measurement, "Real measurement")]
    if reference is not None and synthetic is not None:
        overview_images.extend(
            [(reference, "Lensed reference"), (synthetic, "Synthetic measurement")]
        )
    overview, overview_axes = plt.subplots(1, len(overview_images), figsize=(4 * len(overview_images), 4))
    for axis, (image, title) in zip(np.atleast_1d(overview_axes), overview_images):
        _show_image(axis, image, title)
    overview.tight_layout()

    row_count = len(reconstruction_rows)
    column_count = min(4, len(args.taus))
    group_count = int(np.ceil(len(args.taus) / column_count))
    figure, axes = plt.subplots(
        row_count * group_count,
        column_count,
        figsize=(4 * column_count, 3.5 * row_count * group_count),
        squeeze=False,
    )

    used_axes: set[tuple[int, int]] = set()
    for index, tau in enumerate(args.taus):
        group, column = divmod(index, column_count)
        for row_offset, (label, reconstructions) in enumerate(reconstruction_rows):
            row = group * row_count + row_offset
            _show_image(
                axes[row, column], reconstructions[index], f"{label}, tau={tau:g}",
            )
            used_axes.add((row, column))

    for index in np.ndindex(axes.shape):
        if index not in used_axes:
            axes[index].axis("off")
    figure.tight_layout()

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.save, dpi=160, bbox_inches="tight")
    if not args.no_show:
        plt.show()
    plt.close(overview)
    plt.close(figure)


if __name__ == "__main__":
    main()
