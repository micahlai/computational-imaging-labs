import unittest

import numpy as np

from lab6_2 import (
    crop,
    forward,
    match_psf_to_measurement,
    mean_downsample,
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
        scene = rng.random((12, 14, 3))
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

    def test_wiener_rejects_nonpositive_tau(self):
        image = np.ones((4, 4))
        with self.assertRaisesRegex(ValueError, "positive"):
            wiener(image, image, 0.0)


if __name__ == "__main__":
    unittest.main()
