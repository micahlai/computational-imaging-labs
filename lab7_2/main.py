from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np

from inverse_solver import InverseSolver, adjoint_test


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
    ("gradient_descent", "Gradient descent"),
    ("ista", "ISTA"),
    ("fista", "FISTA"),
)
snapshot_iters = (0, 100, 250, 500, 750, 1000)
solvers = {}
snapshots = {}
step_size = None

for algorithm, title in algorithms:
    solver = InverseSolver(
        psf,
        algorithm=algorithm,
        prox_fn=None if algorithm == "gradient_descent" else lambda x, l, s: np.maximum(x, 0),
        step_size=step_size,
    )
    if step_size is None:
        step_size = solver.step_size
        lhs, rhs = adjoint_test(solver, y.shape)
        print(f"Adjoint test: {lhs:.12g} == {rhs:.12g}")

    snapshots[algorithm] = {0: np.zeros_like(y)}
    # Each call starts at zero; rerun to each checkpoint without changing the class.
    for iteration in snapshot_iters[1:]:
        solver.num_iters = iteration
        x_hat = solver(y)
        snapshots[algorithm][iteration] = x_hat
    solvers[algorithm] = solver
    print(f"{title}: loss {solver.losses[0]:.6g} -> {solver.losses[-1]:.6g}")

overview, axes = plt.subplots(2, 3, figsize=(14, 7))
for axis, image, title in zip(
    axes[0], (y, psf, reference), ("Measurement", "PSF", "Lensed reference")
):
    show_image(axis, image, title)
for axis, (algorithm, title) in zip(axes[1], algorithms):
    show_image(
        axis, snapshots[algorithm][snapshot_iters[-1]],
        f"{title}: {snapshot_iters[-1]} iterations",
    )
overview.tight_layout()

figure, axes = plt.subplots(3, len(snapshot_iters), figsize=(18, 9))
for row, (algorithm, title) in enumerate(algorithms):
    for column, iteration in enumerate(snapshot_iters):
        show_image(
            axes[row, column], snapshots[algorithm][iteration],
            f"{title}: k={iteration}",
        )
figure.suptitle("Reconstruction progress")
figure.tight_layout()

loss_figure, loss_axes = plt.subplots(1, 2, figsize=(12, 4))
for algorithm, title in algorithms:
    solver = solvers[algorithm]
    for axis in loss_axes:
        axis.plot(solver.losses, label=title)
loss_axes[0].set_title("Data-fidelity loss")
loss_axes[1].set_title("Data-fidelity loss (log scale)")
loss_axes[1].set_yscale("log")
for axis in loss_axes:
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Loss")
    axis.grid(True, alpha=0.25)
    axis.legend()
loss_figure.tight_layout()
plt.show()
