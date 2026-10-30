import unittest

import numpy as np

from lab6 import convolve_with_psf, inverse_convolution, psf_to_otf


class DiffuserCameraTests(unittest.TestCase):
    def test_centered_impulse_has_flat_fourier_response(self):
        psf = np.zeros((3, 3))
        psf[1, 1] = 1.0

        np.testing.assert_allclose(psf_to_otf(psf, (8, 10)), 1.0)

    def test_unregularized_inverse_recovers_noise_free_image(self):
        rng = np.random.default_rng(4660)
        expected = rng.random((16, 18))
        psf = np.zeros((3, 3))
        psf[1, 1] = 1.0

        measurement = convolve_with_psf(expected, psf)
        actual = inverse_convolution(measurement, psf, regularization=0)

        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_rgb_channels_are_processed_independently(self):
        rng = np.random.default_rng(6)
        image = rng.random((12, 14, 3))
        psf = np.array(
            [
                [1.0, 2.0, 1.0],
                [2.0, 4.0, 2.0],
                [1.0, 2.0, 1.0],
            ]
        )

        measurement = convolve_with_psf(image, psf)
        reconstructed = inverse_convolution(measurement, psf, regularization=1e-5)

        self.assertEqual(reconstructed.shape, image.shape)
        self.assertLess(np.mean((reconstructed - image) ** 2), 0.03)

    def test_rgb_psf_is_normalized_channel_by_channel(self):
        rng = np.random.default_rng(60)
        expected = rng.random((8, 10, 3))
        psf = np.zeros((3, 3, 3))
        psf[1, 1, :] = (1.0, 2.0, 4.0)

        measurement = convolve_with_psf(expected, psf)
        actual = inverse_convolution(measurement, psf, regularization=0)

        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_rejects_invalid_regularization(self):
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            inverse_convolution(np.ones((4, 4)), np.ones((3, 3)), regularization=-1)


if __name__ == "__main__":
    unittest.main()
