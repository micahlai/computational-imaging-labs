from pathlib import Path

import numpy as np
import rawpy
import imageio as iio
import matplotlib.pyplot as plt
import scipy
from skimage.color import rgb2gray

image_path = Path(__file__).with_name("image.CR3")

with rawpy.imread(str(image_path)) as raw:
    img = raw.raw_image_visible.copy()
    color_desc = raw.color_desc
    raw_pattern = raw.raw_pattern.copy()
    black_level_per_channel = raw.black_level_per_channel
    black_levels = np.array(raw.black_level_per_channel)
    white_level = raw.white_level
    M_cam_to_xyz = raw.rgb_xyz_matrix[:3, :3]
    compare = raw.postprocess()

print(color_desc)

img = img.astype(np.float64)

black_map = np.zeros_like(img)

for row in range(2):
    for col in range(2):
        channel = raw_pattern[row, col]
        black_map[row::2, col::2] = black_levels[channel]

linear = (img - black_map) / (white_level - black_map)
linear = np.clip(linear, 0, 1)

R  = linear[0::2, 0::2]
G1 = linear[0::2, 1::2]
B = linear[1::2, 1::2]
G2  = linear[1::2, 0::2]

r_mean = R.mean()
g_mean = (G1.mean() + G2.mean()) / 2
b_mean = B.mean()

r_gain = g_mean / r_mean
g_gain = 1.0
b_gain = g_mean / b_mean

wb = linear.copy()

wb[0::2, 0::2] *= r_gain
wb[0::2, 1::2] *= g_gain
wb[1::2, 0::2] *= g_gain
wb[1::2, 1::2] *= b_gain

h, w = wb[0::2, 0::2].shape

spline_R = scipy.interpolate.RectBivariateSpline(
    np.arange(h),
    np.arange(w),
    wb[0::2, 0::2],
    kx=1,
    ky=1
)

R_full = spline_R(
    np.arange(2*h) / 2,
    np.arange(2*w) / 2
)

h, w = wb[1::2, 1::2].shape

spline_B = scipy.interpolate.RectBivariateSpline(
    np.arange(h),
    np.arange(w),
    wb[1::2, 1::2],
    kx=1,
    ky=1
)
B_full = spline_B(
    np.arange(2*h) / 2,
    np.arange(2*w) / 2
)


G1 = wb[0::2, 1::2]
G2 = wb[1::2, 0::2]

h, w = G1.shape

# G1 is shifted half a step horizontally
spline_G1 = scipy.interpolate.RectBivariateSpline(
    np.arange(h),
    np.arange(w) + 0.5,
    G1,
    kx=1,
    ky=1
)

# G2 is shifted half a step vertically
spline_G2 = scipy.interpolate.RectBivariateSpline(
    np.arange(h) + 0.5,
    np.arange(w),
    G2,
    kx=1,
    ky=1
)

# Coordinates of every full-resolution pixel
y_full = np.arange(2*h) / 2
x_full = np.arange(2*w) / 2

G1_full = spline_G1(y_full, x_full)
G2_full = spline_G2(y_full, x_full)

# Combine the two green estimates
G_full = (G1_full + G2_full) / 2

# Put the ACTUAL measured green values back
G_full[0::2, 1::2] = G1
G_full[1::2, 0::2] = G2


rgb = np.stack(
    [R_full, G_full, B_full],
    axis=-1
)

M_srgb_to_xyz = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]
])

M_srgb_to_cam = M_cam_to_xyz @ M_srgb_to_xyz
M_srgb_to_cam = M_srgb_to_cam / M_srgb_to_cam.sum(axis=1, keepdims=True)
M_cam_to_srgb = np.linalg.inv(M_srgb_to_cam)
rgb_corrected = rgb @ M_cam_to_srgb.T


gray = rgb2gray(rgb_corrected)

current_mean = gray.mean()

target_mean = 0.25
scale = target_mean / current_mean

bright = rgb_corrected * scale
bright = np.clip(bright, 0, 1)

final = np.where(
    bright <= 0.0031308,
        12.92 * bright,
        1.055 * np.power(bright, 1/2.4) - 0.055
)

print("M_cam_to_xyz:")
print(M_cam_to_xyz)

print("\ndeterminant:")
print(np.linalg.det(M_cam_to_xyz))

print("\nM_srgb_to_cam before normalization:")
M = np.linalg.inv(M_cam_to_xyz) @ M_srgb_to_xyz
print(M)

print("\nrow sums:")
print(M.sum(axis=1))

M /= M.sum(axis=1, keepdims=True)

print("\nnormalized:")
print(M)

print("\ncam -> sRGB:")
print(np.linalg.inv(M))

plt.figure(1)               # Creates the first separate window
plt.imshow(final)
plt.title("mine")

# --- Window 2 ---
plt.figure(2)               # Creates the second separate window
plt.imshow(compare)
plt.title("library postprocess")
plt.show()