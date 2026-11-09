"""Resize an MP4, then convolve each frame spatially with a scaled TIFF PSF."""

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .inverse_solver import InverseSolver
    from .video_io import load_psf, read_video, write_video
except ImportError:
    from inverse_solver import InverseSolver
    from video_io import load_psf, read_video, write_video


def prepare(video_path, psf_path, resolution, output_dir, psf_scale=1.0, max_frames=None):
    """Save exact synthetic data plus viewable MP4s; return dataset.npz path."""
    if Path(video_path).suffix.lower() != ".mp4":
        raise ValueError("Input video must be an .mp4")
    if Path(psf_path).suffix.lower() not in (".tif", ".tiff"):
        raise ValueError("PSF must be a .tif or .tiff")
    if len(resolution) != 2:
        raise ValueError("resolution must be (width, height)")
    reference, fps, source_size = read_video(video_path, resolution, max_frames)
    psf = load_psf(psf_path, source_size, resolution, psf_scale)
    # Precisely the same normalization, centering, and boundaries as inversion.
    model = InverseSolver(psf)
    measurement = model.A(reference)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(source_video=str(Path(video_path).resolve()),
                    source_psf=str(Path(psf_path).resolve()),
                    source_size=list(source_size), target_size=list(resolution),
                    psf_scale=psf_scale, fps=fps, frames=reference.shape[3],
                    layout="H,W,C,T", boundary="zero", psf_origin="kh//2,kw//2",
                    convolution="spatial only, shared unit-sum PSF",
                    preview="lossy clipped RGB MP4; no audio")
    dataset = output_dir / "dataset.npz"
    np.savez_compressed(dataset, measurement=measurement, reference=reference,
                        psf=model.h, fps=fps, metadata=json.dumps(metadata))
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    write_video(output_dir / "blurred.mp4", measurement, fps)
    write_video(output_dir / "reference.mp4", reference, fps)
    return dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("psf", type=Path)
    parser.add_argument("--resolution", type=int, nargs=2, required=True, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--psf-scale", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("prepared"))
    parser.add_argument("--max-frames", type=int, help="Use just the first N frames")
    args = parser.parse_args()
    try:
        dataset = prepare(args.video, args.psf, args.resolution, args.output_dir,
                          args.psf_scale, args.max_frames)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Saved exact synthetic data: {dataset}")
    print(f"MP4 preview: {args.output_dir / 'blurred.mp4'}")


if __name__ == "__main__":
    main()
