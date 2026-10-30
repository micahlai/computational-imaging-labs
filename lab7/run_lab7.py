"""Run Lab 7 on the same public DiffuserCam sample used by Lab 6.

From the repository root:

    conda activate cs4660
    python lab7/run_lab7.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np

try:
    from .starter import InverseSolver, adjoint_test
except ImportError:  # Allow ``python lab7/run_lab7.py``.
    from starter import InverseSolver, adjoint_test


SAMPLE_DIR = (
    Path(__file__).resolve().parents[1]
    / "lab6_main"
    / "lab6"
    / "test_data"
    / "shamus-sample"
)


def mean_downsample(image: np.ndarray, factor: int) -> np.ndarray:
    """Block-average the first two dimensions by an integer factor."""
    image = np.asarray(image, dtype=np.float64)
    if factor <= 0 or image.shape[0] % factor or image.shape[1] % factor:
        raise ValueError("factor must evenly divide both image dimensions")
    trailing = image.shape[2:]
    reshaped = image.reshape(
        image.shape[0] // factor,
        factor,
        image.shape[1] // factor,
        factor,
        *trailing,
    )
    return reshaped.mean(axis=(1, 3))


def normalize_unit_interval(image: np.ndarray) -> np.ndarray:
    """Map a finite image to [0, 1] for reconstruction and display."""
    image = np.asarray(image, dtype=np.float64)
    minimum = float(image.min())
    maximum = float(image.max())
    if np.isclose(minimum, maximum):
        return np.zeros_like(image)
    return (image - minimum) / (maximum - minimum)


def load_lab6_sample(downsample: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess the Lab 6 measurement, PSF, and reference."""
    measurement = np.load(SAMPLE_DIR / "im65.npy")
    reference = np.load(SAMPLE_DIR / "im65_gt.npy")
    psf = iio.imread(SAMPLE_DIR / "psf.tiff")

    row_factor = psf.shape[0] // measurement.shape[0]
    col_factor = psf.shape[1] // measurement.shape[1]
    if row_factor != col_factor or row_factor <= 0:
        raise ValueError("PSF and measurement resolutions are incompatible")
    psf = mean_downsample(psf, row_factor)

    if downsample > 1:
        measurement = mean_downsample(measurement, downsample)
        psf = mean_downsample(psf, downsample)
        reference = mean_downsample(reference, downsample)

    return (
        normalize_unit_interval(measurement),
        np.asarray(psf, dtype=np.float64),
        normalize_unit_interval(reference),
    )


def display_image(image: np.ndarray) -> np.ndarray:
    """Create a robust display image without integer color wrapping."""
    image = np.asarray(image, dtype=np.float64)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image)
    low, high = np.percentile(finite, (0.5, 99.5))
    if np.isclose(low, high):
        return np.zeros_like(image)
    return np.clip((image - low) / (high - low), 0.0, 1.0)


def log10_losses(losses: list[float]) -> np.ndarray:
    """Return bounded log-loss values, retaining visibly divergent runs."""
    values = np.asarray(losses, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        logged = np.log10(values)
    return np.nan_to_num(logged, nan=300.0, posinf=300.0, neginf=-300.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--manual-step-size", type=float, default=1000.0)
    parser.add_argument("--save", type=Path, help="save figures using this filename prefix")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.downsample <= 0:
        raise ValueError("--downsample must be positive")

    measurement, psf, reference = load_lab6_sample(args.downsample)
    snapshot_iters = sorted(
        {0, args.iterations // 4, args.iterations // 2, 3 * args.iterations // 4, args.iterations}
    )

    # Problem 1: verify the hand-written adjoint before optimizing.
    test_solver = InverseSolver(psf, num_iters=0, step_size=1.0)
    lhs, rhs = adjoint_test(test_solver, measurement.shape)
    relative_error = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0)
    print(f"Adjoint test: <y, Ax>={lhs:.12g}, <A*y, x>={rhs:.12g}")
    print(f"Adjoint relative error: {relative_error:.3e}")

    # Problem 2(c): the handout-requested fixed step size.
    manual_solver = InverseSolver(
        psf,
        algorithm="gradient_descent",
        num_iters=args.iterations,
        step_size=args.manual_step_size,
    )
    manual_solver(measurement, snapshot_iters=snapshot_iters)
    manual_losses = np.asarray(manual_solver.losses)
    divergence = np.flatnonzero(
        ~np.isfinite(manual_losses)
        | (manual_losses > max(manual_losses[0], 1.0) * 1.0e6)
    )
    if divergence.size:
        print(f"Fixed step {args.manual_step_size:g} diverges by iteration {divergence[0]}")

    # Problem 2(d): step size estimated from power iteration.
    automatic_solver = InverseSolver(
        psf,
        algorithm="gradient_descent",
        num_iters=args.iterations,
    )
    reconstruction = automatic_solver(measurement, snapshot_iters=snapshot_iters)
    print(
        "Automatic step size: "
        f"{automatic_solver.step_size:.6g} "
        f"(L={automatic_solver.lipschitz_constant:.6g})"
    )
    print(
        "Automatic loss: "
        f"{automatic_solver.losses[0]:.6g} -> {automatic_solver.losses[-1]:.6g}"
    )

    figure, axes = plt.subplots(2, len(snapshot_iters), figsize=(4 * len(snapshot_iters), 7))
    for column, iteration in enumerate(snapshot_iters):
        axes[0, column].imshow(display_image(manual_solver.snapshots[iteration]))
        axes[0, column].set_title(f"step={args.manual_step_size:g}, k={iteration}")
        axes[1, column].imshow(display_image(automatic_solver.snapshots[iteration]))
        axes[1, column].set_title(f"power iteration, k={iteration}")
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    figure.suptitle("Lab 7 gradient-descent reconstructions")
    figure.tight_layout()

    loss_figure, loss_axes = plt.subplots(1, 2, figsize=(11, 4))
    loss_axes[0].plot(log10_losses(manual_solver.losses), color="tab:blue")
    loss_axes[0].set_title(f"Fixed step = {args.manual_step_size:g} (diverges)")
    loss_axes[1].plot(log10_losses(automatic_solver.losses), color="tab:orange")
    loss_axes[1].set_title("Power-iteration step (converges)")
    for axis in loss_axes:
        axis.set_xlabel("Iteration")
        axis.set_ylabel(r"$\log_{10}\!\left(\frac{1}{2}\|A(x)-y\|_2^2\right)$")
        axis.grid(True, alpha=0.25)
    loss_figure.suptitle("Data-fidelity loss")
    loss_figure.tight_layout()

    overview, overview_axes = plt.subplots(1, 4, figsize=(15, 4))
    for axis, image, title in zip(
        overview_axes,
        (measurement, psf, reference, reconstruction),
        ("Measurement", "PSF", "Lensed reference", "Iterative reconstruction"),
    ):
        axis.imshow(display_image(image))
        axis.set_title(title)
        axis.axis("off")
    overview.tight_layout()

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.save.with_name(args.save.stem + "_iterations.png"), dpi=160)
        loss_figure.savefig(args.save.with_name(args.save.stem + "_loss.png"), dpi=160)
        overview.savefig(args.save.with_name(args.save.stem + "_overview.png"), dpi=160)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
