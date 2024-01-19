
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GaussianBlur3D(nn.Module):
    """
    Gaussian 3D convolution using a real-space approach.
    if the ratio of max_kernel_size / sigma > 6, the maximum discrepancy between these results
    on a cube test image of 24^3 in centered in a 64^3 image is around or lower than 1%, and
    has a mean error of about a factor 2 lower than that.

    """
    def __init__(self, initial_sigma, max_kernel_size=11):
        super(GaussianBlur3D, self).__init__()
        self.sigma = nn.Parameter(torch.tensor([initial_sigma], dtype=torch.float32))
        self.max_kernel_size = max_kernel_size
        self.kernel = self.create_gaussian_kernel(self.max_kernel_size, self.sigma)

    def create_gaussian_kernel(self, kernel_size, sigma):
        # Grids for x, y, z
        range = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
        x = range.view(-1, 1, 1).repeat(1, kernel_size, kernel_size)
        y = range.view(1, -1, 1).repeat(kernel_size, 1, kernel_size)
        z = range.view(1, 1, -1).repeat(kernel_size, kernel_size, 1)

        # 3D Gaussian kernel
        gaussian_kernel = torch.exp(-(x**2 + y**2 + z**2) / (2 * sigma**2))
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        return gaussian_kernel.view(1, 1, kernel_size, kernel_size, kernel_size)

    def forward(self, x):
        # Ensure sigma is positive
        sigma = self.sigma.abs() + 1e-6

        # Update kernel based on current sigma
        self.kernel = self.create_gaussian_kernel(self.max_kernel_size, sigma)

        # Calculate padding and apply reflective padding
        padding = int((self.max_kernel_size - 1) / 2)
        x_padded = F.pad(x, [padding, padding, padding, padding, padding, padding], mode='reflect')

        # Apply the Gaussian kernel
        blurred = F.conv3d(x_padded, self.kernel, padding=0)
        return blurred
