from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np

from inverse_solver import InverseSolver, adjoint_test
from priors import prox_nonneg, prox_native


master_iter: int = 2500


def show_image(axis, image, title):
    low, high = np.percentile(image, (0.5, 99.5))
    display = np.clip((image - low) / max(high - low, np.finfo(float).eps), 0, 1)
    axis.imshow(display)
    axis.set_title(title)
    axis.axis("off")


data_dir = (
    Path(__file__).resolve().parents[1]
    / "lab6_main/lab6/test_data/shamus-sample"
)
y = np.load(data_dir / "im65.npy")
reference = np.load(data_dir / "im65_gt.npy")
psf = iio.imread(data_dir / "psf.tiff").astype(float)
# Average each 4x4 PSF block to match the measurement resolution.
psf = psf.reshape(y.shape[0], 4, y.shape[1], 4, y.shape[2]).mean(axis=(1, 3))

algorithms = (
    ("ista", "ISTA"),
    ("fista", "FISTA"),
)
# Each (label, proximal operator, lambda) entry adds a comparison row.
prox_configs = (
    ("Nonnegativity", prox_nonneg, 0.0),
    ("prox_native, λ=1e-5", prox_native, 1e-5),
    ("prox_native, λ=1e-4", prox_native, 1e-4),
    ("prox_native, λ=1e-3", prox_native, 1e-3),
)
step_size = None

overview, axes = plt.subplots(1, 3, figsize=(14, 5))
overview.canvas.manager.set_window_title("Measurement, PSF, and lensed reference")
for axis, image, title in zip(
    axes, (y, psf, reference), ("Measurement", "PSF", "Lensed reference")
):
    show_image(axis, image, title)
overview.tight_layout()

figure, axes = plt.subplots(
    len(prox_configs), len(algorithms),
    figsize=(12, 3 * len(prox_configs)), squeeze=False,
)
figure.canvas.manager.set_window_title("ISTA and FISTA comparison")
figure.suptitle(f"Final reconstructions: {master_iter} iterations")
for row, (label, prox_fn, lam) in enumerate(prox_configs):
    for column, (algorithm, title) in enumerate(algorithms):
        solver = InverseSolver(
            psf,
            algorithm=algorithm,
            num_iters=master_iter,
            prox_fn=prox_fn,
            lam=lam,
            step_size=step_size,
        )
        if step_size is None:
            step_size = solver.step_size
            lhs, rhs = adjoint_test(solver, y.shape)
            print(f"Adjoint test: {lhs:.12g} == {rhs:.12g}")

        x_hat = solver(y)
        show_image(
            axes[row, column], x_hat, f"{title}: {label}",
        )
        print(
            f"{title} ({label}): "
            f"loss {solver.losses[0]:.6g} -> {solver.losses[-1]:.6g}"
        )
figure.tight_layout()

plt.show()
