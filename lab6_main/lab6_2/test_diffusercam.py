import unittest
from unittest.mock import patch

import numpy as np

from lab6_main.lab6_2 import (
    crop,
    forward,
    match_psf_to_measurement,
    mean_downsample,
    normalize_psf,
    normalize_unit_interval,
    pad,
    wiener,
)


class DiffuserCamTests(unittest.TestCase):
    def test_pad_doubles_size_and_crop_restores_image(self):
        image = np.arange(5 * 7 * 3, dtype=float).reshape(5, 7, 3)
        padded = pad(image)

        self.assertEqual(padded.shape, (10, 14, 3))
        np.testing.assert_array_equal(crop(padded, image.shape[:2]), image)

    def test_centered_impulse_psf_is_identity_forward_model(self):
        rng = np.random.default_rng(4660)
        scene = rng.random((11, 15, 3))
        psf = np.zeros_like(scene)
        psf[scene.shape[0] // 2, scene.shape[1] // 2, :] = 1.0

        measurement = forward(scene, psf)

        np.testing.assert_allclose(measurement, scene, atol=1e-12)

    def test_wiener_identity_has_expected_regularization_scale(self):
        rng = np.random.default_rng(6)
        measurement = rng.random((10, 8))
        psf = np.zeros_like(measurement)
        psf[measurement.shape[0] // 2, measurement.shape[1] // 2] = 1.0
        tau = 0.25

        reconstruction = wiener(measurement, psf, tau)

        np.testing.assert_allclose(reconstruction, measurement / (1.0 + tau), atol=1e-12)

    def test_mean_downsample_uses_block_averages(self):
        image = np.arange(16, dtype=float).reshape(4, 4)
        expected = np.array([[2.5, 4.5], [10.5, 12.5]])

        np.testing.assert_allclose(mean_downsample(image, 2), expected)

    def test_match_psf_to_measurement(self):
        measurement = np.zeros((2, 3, 3))
        psf = np.arange(2 * 4 * 3 * 4 * 3, dtype=float).reshape(8, 12, 3)

        matched = match_psf_to_measurement(psf, measurement)

        self.assertEqual(matched.shape, measurement.shape)
        np.testing.assert_allclose(matched, mean_downsample(psf, 4))

    def test_display_normalization_does_not_wrap_colors(self):
        image = np.array([[[-10.0, 20.0, 300.0], [0.0, 30.0, 100.0]]])

        display = normalize_unit_interval(image)

        self.assertEqual(display.dtype, np.float64)
        self.assertGreaterEqual(float(display.min()), 0.0)
        self.assertLessEqual(float(display.max()), 1.0)
        self.assertEqual(float(display[0, 0, 0]), 0.0)
        self.assertEqual(float(display[0, 0, 2]), 1.0)

    def test_psf_normalization_sets_each_channel_sum_to_one(self):
        psf = np.arange(1, 4 * 6 * 3 + 1, dtype=float).reshape(4, 6, 3)

        normalized = normalize_psf(psf)

        np.testing.assert_allclose(normalized.sum(axis=(0, 1)), np.ones(3))

    def test_edge_boundary_preserves_identity_wiener_result(self):
        rng = np.random.default_rng(60)
        measurement = rng.random((9, 13, 3))
        psf = np.zeros_like(measurement)
        psf[measurement.shape[0] // 2, measurement.shape[1] // 2, :] = 1.0
        tau = 0.1

        reconstruction = wiener(measurement, psf, tau, boundary="edge")

        np.testing.assert_allclose(reconstruction, measurement / (1.0 + tau), atol=1e-12)

    def test_wiener_rejects_nonpositive_tau(self):
        image = np.ones((4, 4))
        with self.assertRaisesRegex(ValueError, "positive"):
            wiener(image, image, 0.0)

    def test_circular_identity_preserves_alignment_and_channel_broadcasting(self):
        rng = np.random.default_rng(61)
        for shape in [(9, 13), (10, 8), (9, 13, 3)]:
            for psf_channels in [(), (1,)]:
                with self.subTest(shape=shape, psf_channels=psf_channels):
                    measurement = rng.normal(size=shape)
                    psf = np.zeros(shape[:2] + psf_channels)
                    psf[shape[0] // 2, shape[1] // 2] = 1
                    result = wiener(measurement, psf, 0.25, boundary="circular")
                    np.testing.assert_allclose(result, measurement / 1.25, atol=1e-12)

    def test_circular_recovers_asymmetric_blur_at_low_tau(self):
        rng = np.random.default_rng(62)
        scene = rng.normal(size=(9, 12, 3))
        psf = np.zeros((9, 12))
        psf[4, 6], psf[5, 6], psf[4, 8] = 0.7, 0.2, 0.1
        # Independent spatial circular convolution checks kernel phase/alignment.
        measurement = (0.7 * scene + 0.2 * np.roll(scene, 1, axis=0)
                       + 0.1 * np.roll(scene, 2, axis=1))
        result = wiener(measurement, psf, 1e-10, boundary="circular")
        np.testing.assert_allclose(result, scene, atol=3e-9)

    def test_periodic_smooth_removes_ramp_seam_preserves_mean_without_padding(self):
        # For an affine ramp, the periodic component has 1/N of its slope.
        # This checks the Poisson sign, corner contributions and RGB independence.
        for height, width in [(9, 12), (10, 7), (1, 8), (8, 1), (1, 1)]:
            with self.subTest(shape=(height, width)):
                rows = np.arange(height)[:, None] - (height - 1) / 2
                cols = np.arange(width)[None, :] - (width - 1) / 2
                ramp = 2 * rows + 3 * cols
                image = 5 + ramp[..., None] * np.array([1, 2, -1])
                expected = 5 + (2 * rows / height + 3 * cols / width)[..., None] * np.array([1, 2, -1])
                psf = np.zeros((height, width))
                psf[height // 2, width // 2] = 1
                with patch("numpy.pad", side_effect=AssertionError("must not pad")):
                    result = wiener(image, psf, 1e-6, boundary="periodic_smooth")
                np.testing.assert_allclose(result, expected / (1 + 1e-6), atol=1e-11)
                np.testing.assert_allclose(result.mean(axis=(0, 1)), image.mean(axis=(0, 1)) / (1 + 1e-6))

    def test_periodic_smooth_preserves_image_when_opposite_borders_match(self):
        rng = np.random.default_rng(63)
        measurement = rng.normal(size=(11, 14))
        measurement[-1] = measurement[0]
        measurement[:, -1] = measurement[:, 0]
        psf = np.zeros_like(measurement)
        psf[5, 7] = 1
        result = wiener(measurement, psf, 0.1, boundary="periodic_smooth")
        np.testing.assert_allclose(result, measurement / 1.1, atol=1e-12)

    def test_wiener_rejects_unknown_boundary(self):
        image = np.ones((4, 4))
        with self.assertRaisesRegex(ValueError, "boundary"):
            wiener(image, image, 1.0, boundary="periodic")


if __name__ == "__main__":
    unittest.main()
