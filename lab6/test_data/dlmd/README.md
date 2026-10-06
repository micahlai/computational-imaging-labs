# DiffuserCam Lensless Mirflickr example

This directory contains a matched PSF, lensless measurement, and lensed ground
truth from the public DiffuserCam Lensless Mirflickr Dataset (DLMD). Only the
small example files are included; the full dataset is several gigabytes.

Source: <https://huggingface.co/datasets/bezzam/DiffuserCam-Lensless-Mirflickr-Dataset>

Original dataset: <https://waller-lab.github.io/LenslessLearning/dataset.html>

Original license: BSD 3-Clause, Copyright (c) 2019 Waller Lab:
<https://github.com/Waller-Lab/LenslessLearning/blob/master/LICENSE>

| Local file | Source file | Shape | SHA-256 |
| --- | --- | --- | --- |
| `lensed_ground_truth.png` | `lensed_example.png` | 270 x 480 x 3 | `aff66bfe925e9324a28abcf32f3235b85b49c33a59ffcdb80d80f03555dd0fca` |
| `lensless_measurement.png` | `lensless_example.png` | 270 x 480 x 3 | `b04402174b3f77a368c9f410f315cf05ad5d017cc028eff81a2ff38dd09c413c` |
| `psf.png` | `psf.png` | 1080 x 1920 x 3 | `c050eb7652505a44e751a6035381b952245d2714fb5c89e60da665831cabaed2` |

The PSF has four times the spatial resolution of the measurement. Reduce it to
270 x 480 with 4 x 4 block averaging before passing it to the reconstruction.
