
import numpy as np

def gaussian_kernel_2d(kernel_size, sigma=1.0):
    """
    Generates a 2D Gaussian kernel array.
    
    Parameters:
    - kernel_size (int): The width and height of the square kernel (ideally an odd number).
    - sigma (float): The standard deviation of the Gaussian distribution.
    """
    # 1. Create a coordinate grid centered at zero
    ax = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    
    # 2. Compute the 2D Gaussian exponent
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    
    # 3. Normalize the kernel so all elements sum to 1
    return kernel / np.sum(kernel)