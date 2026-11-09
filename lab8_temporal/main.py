"""Reconstruct an entire video jointly with FISTA and two selectable priors."""

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .inverse_solver import InverseSolver, adjoint_test
    from .video_io import load_psf, read_video, write_video
except ImportError:
    from inverse_solver import InverseSolver, adjoint_test
    from video_io import load_psf, read_video, write_video


def lambda_from_exponent(value):
    """Parse a base-10 exponent into a finite, positive solver weight."""
    try:
        exponent = float(value)
        weight = 10.0 ** exponent
    except (ValueError, OverflowError) as error:
        raise argparse.ArgumentTypeError("Expected a base-10 exponent, e.g. -5 for 1e-5") from error
    if not np.isfinite(exponent) or not np.isfinite(weight) or weight <= 0:
        raise argparse.ArgumentTypeError("Exponent must produce a finite, positive lambda")
    return weight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="prep.py dataset.npz, or a real blurred MP4")
    parser.add_argument("--psf", type=Path, help="TIFF PSF, required only for MP4 input")
    parser.add_argument("--psf-scale", type=float, default=1.0,
                        help="For MP4 input: PSF pixel size relative to input video pixels")
    parser.add_argument("--spatial-prior", choices=("wavelet", "tv"), default="wavelet")
    parser.add_argument("--temporal-prior", choices=("wavelet", "tv"), default="tv")
    parser.add_argument("--lambda-spatial", type=lambda_from_exponent, default="-5",
                        metavar="EXPONENT", help="Base-10 exponent: -5 means 1e-5 (default: -5)")
    parser.add_argument("--lambda-temporal", type=lambda_from_exponent, default="-5",
                        metavar="EXPONENT", help="Base-10 exponent: -5 means 1e-5 (default: -5)")
    parser.add_argument("--num-iters", type=int, default=200)
    parser.add_argument("--prox-iters", type=int, default=200)
    parser.add_argument("--prox-tol", type=float, default=1e-5)
    parser.add_argument("--wavelet", default="db4")
    parser.add_argument("--wavelet-level", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("reconstructed"))
    args = parser.parse_args()
    reference = None
    if args.input.suffix.lower() == ".npz":
        if args.psf is not None or args.psf_scale != 1.0:
            parser.error("Synthetic datasets already contain the exact scaled PSF; omit --psf/--psf-scale")
        with np.load(args.input, allow_pickle=False) as dataset:
            measurement, psf, fps = dataset["measurement"], dataset["psf"], float(dataset["fps"])
            if "reference" in dataset:
                reference = dataset["reference"]
    elif args.input.suffix.lower() == ".mp4":
        if args.psf is None:
            parser.error("--psf is required for an MP4 input")
        measurement, fps, source_size = read_video(args.input)
        psf = load_psf(args.psf, source_size, source_size, args.psf_scale)
    else:
        parser.error("Input must be a dataset.npz or an .mp4")
    solver = InverseSolver(psf, num_iters=args.num_iters,
                           lam_spatial=args.lambda_spatial, lam_temporal=args.lambda_temporal,
                           spatial_prior=args.spatial_prior, temporal_prior=args.temporal_prior,
                           wavelet=args.wavelet, wavelet_level=args.wavelet_level,
                           prox_iters=args.prox_iters, prox_tol=args.prox_tol)
    # A small probe checks the operator without allocating another full video.
    lhs, rhs = adjoint_test(solver, measurement.shape[:2] + (1, 1))
    if not np.isclose(lhs, rhs, rtol=1e-10, atol=1e-10):
        raise RuntimeError(f"Adjoint test failed: {lhs} != {rhs}")

    def progress(iteration, loss):
        if iteration == 1 or iteration % 10 == 0 or iteration == args.num_iters:
            print(f"FISTA {iteration}/{args.num_iters}: data loss {loss:.6g}", flush=True)

    result = solver(measurement, callback=progress)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "reconstructed.npy", result)
    np.save(args.output_dir / "data_losses.npy", solver.losses)
    write_video(args.output_dir / "reconstructed.mp4", result, fps)
    report = dict(input=str(args.input.resolve()), shape=list(result.shape), fps=fps,
                  spatial_prior=args.spatial_prior, temporal_prior=args.temporal_prior,
                  lambda_spatial=args.lambda_spatial, lambda_temporal=args.lambda_temporal,
                  iterations=args.num_iters, wavelet=args.wavelet, wavelet_level=args.wavelet_level,
                  prox_iters=args.prox_iters, prox_tol=args.prox_tol,
                  final_data_loss=solver.losses[-1],
                  unconverged_prox_steps=sum(not d["converged"] for d in solver.prox_diagnostics))
    if reference is not None:
        if reference.shape != result.shape:
            raise ValueError("Reference and measurement dimensions differ")
        report["blurred_mse"] = float(np.mean((measurement - reference)**2))
        report["reconstructed_mse"] = float(np.mean((result - reference)**2))
        print(f"MSE: blurred={report['blurred_mse']:.6g}, reconstructed={report['reconstructed_mse']:.6g}")
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved reconstruction: {args.output_dir / 'reconstructed.mp4'}")


if __name__ == "__main__":
    main()
