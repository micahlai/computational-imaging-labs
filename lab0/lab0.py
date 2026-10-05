import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import kernels

path = 'lab0/example.png'
image = iio.imread(path)
kernel = kernels.gaussian_kernel_2d(10,10)
img_height, img_width, channels = image[..., :3].shape
kernel_height, kernel_width = kernel.shape
rgb_image = image[..., :3]
output = np.stack(
    [convolve2d(rgb_image[..., channel], kernel, mode='same', boundary='symm')
    for channel in range(rgb_image.shape[-1])],
    axis=-1,
)
output = np.clip(output, 0, 255).astype(np.uint8)


plt.imshow(output)
plt.axis('off')
plt.show()