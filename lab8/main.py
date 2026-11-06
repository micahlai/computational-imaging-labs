from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from inverse_solver import InverseSolver, adjoint_test
from priors import prox_nonneg, prox_native, prox_wavelet, prox_tv


master_iter: int = 2500


def reconstruct_with_progress(solver, measurement, label):
    """Track completed solver updates without changing the solver's iteration loop."""
    update_name = f"{solver.algorithm}_update"
    update = getattr(solver, update_name)
    with tqdm(total=solver.num_iters, desc=label, unit="iter", dynamic_ncols=True) as progress:
        def tracked_update(*args, **kwargs):
            result = update(*args, **kwargs)
            progress.update(1)
            return result

        setattr(solver, update_name, tracked_update)
        try:
            return solver(measurement)
        finally:
            setattr(solver, update_name, update)


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

# Each prior adds a row; each lambda adds a column.
prox_configs = (
    ("prox_nonneg", prox_nonneg),
    ("prox_native", prox_native),
    ("prox_wavelet", prox_wavelet),
    ("prox_tv", prox_tv),
)
lambda_values = (1e-3, 1e-4, 1e-5)
step_size = None

overview, axes = plt.subplots(1, 3, figsize=(14, 5))
overview.canvas.manager.set_window_title("Measurement, PSF, and lensed reference")
for axis, image, title in zip(
    axes, (y, psf, reference), ("Measurement", "PSF", "Lensed reference")
):
    show_image(axis, image, title)
overview.tight_layout()

figure, axes = plt.subplots(
    len(prox_configs), len(lambda_values),
    figsize=(5 * len(lambda_values), 3 * len(prox_configs)), squeeze=False,
)
figure.canvas.manager.set_window_title("FISTA: priors and lambda comparison")
figure.suptitle(f"Final FISTA reconstructions: {master_iter} iterations")
total_runs = sum(
    1 if prox_fn is prox_nonneg else len(lambda_values)
    for _, prox_fn in prox_configs
)
run_number = 0
for row, (label, prox_fn) in enumerate(prox_configs):
    # Nonnegativity is independent of lambda; solve once and reuse the image.
    row_lambdas = (0.0,) if prox_fn is prox_nonneg else lambda_values
    for column, lam in enumerate(row_lambdas):
        solver = InverseSolver(
            psf,
            algorithm="fista",
            num_iters=master_iter,
            prox_fn=prox_fn,
            lam=lam,
            step_size=step_size,
        )
        if step_size is None:
            step_size = solver.step_size
            lhs, rhs = adjoint_test(solver, y.shape)
            print(f"Adjoint test: {lhs:.12g} == {rhs:.12g}")

        run_number += 1
        result_label = (
            f"{label} (λ-independent)"
            if prox_fn is prox_nonneg else f"{label}, λ={lam:.0e}"
        )
        x_hat = reconstruct_with_progress(
            solver, y, f"[{run_number}/{total_runs}] FISTA: {result_label}",
        )
        display_axes = axes[row] if prox_fn is prox_nonneg else (axes[row, column],)
        for axis in display_axes:
            show_image(axis, x_hat, result_label)
        print(
            f"FISTA ({result_label}): "
            f"loss {solver.losses[0]:.6g} -> {solver.losses[-1]:.6g}"
        )
figure.tight_layout()

plt.show()
