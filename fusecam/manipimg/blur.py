
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GaussianBlur3D(nn.Module):
    """
    A module that applies 3D Gaussian blurring to an input tensor.

    This class creates a Gaussian 3D convolution kernel that is applied to the input tensor.
    The Gaussian kernel is defined by a sigma value, which controls the spread of the blur.
    The size of the kernel can be adjusted with the `max_kernel_size` parameter.

    Note: if the ratio of max_kernel_size / sigma > 6, the maximum discrepancy between these results
    on a cube test image of 24^3 in centered in a 64^3 image is around or lower than 1%, and
    has a mean error of about 0.5% in areas where there was signal in the unblurred image.

    Attributes:
        sigma (nn.Parameter): The standard deviation of the Gaussian kernel.
        max_kernel_size (int): The maximum size of the Gaussian kernel.
        kernel (Tensor): The Gaussian kernel.

    Args:
        initial_sigma (float): The initial standard deviation of the Gaussian kernel.
        max_kernel_size (int, optional): The maximum size of the Gaussian kernel. Default is 11.

    """
    def __init__(self, initial_sigma, max_kernel_size=11):
        super(GaussianBlur3D, self).__init__()
        self.sigma = nn.Parameter(torch.tensor([initial_sigma], dtype=torch.float32))
        self.max_kernel_size = max_kernel_size
        self.kernel = self.create_gaussian_kernel(self.max_kernel_size, self.sigma)

    def create_gaussian_kernel(self, kernel_size, sigma):
        """
        Creates a 3D Gaussian kernel.

        This function generates a 3D Gaussian kernel using the specified kernel size and sigma.
        The kernel is normalized so that its sum is 1.

        Args:
            kernel_size (int): The size of the kernel.
            sigma (Tensor): The standard deviation of the Gaussian kernel.

        Returns:
            Tensor: A normalized 3D Gaussian kernel.
        """
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
        """
        Apply Gaussian blurring to the input tensor.

        This method updates the Gaussian kernel based on the current value of sigma,
        applies padding to the input tensor, and then convolves it with the kernel to produce a blurred output.

        Args:
            x (Tensor): The input tensor to be blurred.

        Returns:
            Tensor: The blurred output tensor.
        """
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


class GaussianBlur2D(nn.Module):
    """
    A module that applies 2D Gaussian blurring to an input tensor.

    This class creates a Gaussian 2D convolution kernel that is applied to the input tensor.
    The Gaussian kernel is defined by a sigma value, which controls the spread of the blur.
    The size of the kernel can be adjusted with the `max_kernel_size` parameter.

    Attributes:
        sigma (nn.Parameter): The standard deviation of the Gaussian kernel.
        max_kernel_size (int): The maximum size of the Gaussian kernel.
        kernel (Tensor): The Gaussian kernel.

    Args:
        initial_sigma (float): The initial standard deviation of the Gaussian kernel.
        max_kernel_size (int, optional): The maximum size of the Gaussian kernel. Default is 11.
    """
    def __init__(self, initial_sigma, max_kernel_size=11):
        super(GaussianBlur2D, self).__init__()
        self.sigma = nn.Parameter(torch.tensor([initial_sigma], dtype=torch.float32))
        self.max_kernel_size = max_kernel_size
        self.kernel = self.create_gaussian_kernel(self.max_kernel_size, self.sigma)

    def create_gaussian_kernel(self, kernel_size, sigma):
        """
        Creates a 2D Gaussian kernel.

        This function generates a 2D Gaussian kernel using the specified kernel size and sigma.
        The kernel is normalized so that its sum is 1.

        Args:
            kernel_size (int): The size of the kernel.
            sigma (Tensor): The standard deviation of the Gaussian kernel.

        Returns:
            Tensor: A normalized 2D Gaussian kernel.
        """
        # Grids for x, y
        range = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
        x = range.view(-1, 1).repeat(1, kernel_size)
        y = range.view(1, -1).repeat(kernel_size, 1)

        # 2D Gaussian kernel
        gaussian_kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        return gaussian_kernel.view(1, 1, kernel_size, kernel_size)

    def forward(self, x):
        """
        Apply Gaussian blurring to the input tensor.

        This method updates the Gaussian kernel based on the current value of sigma,
        applies padding to the input tensor, and then convolves it with the kernel to produce a blurred output.

        Args:
            x (Tensor): The input tensor to be blurred.

        Returns:
            Tensor: The blurred output tensor.
        """
        # Ensure sigma is positive
        sigma = self.sigma.abs() + 1e-6

        # Update kernel based on current sigma
        self.kernel = self.create_gaussian_kernel(self.max_kernel_size, sigma)

        # Calculate padding and apply reflective padding
        padding = int((self.max_kernel_size - 1) / 2)
        x_padded = F.pad(x, [padding, padding, padding, padding], mode='reflect')

        # Apply the Gaussian kernel
        blurred = F.conv2d(x_padded, self.kernel, padding=0)
        return blurred
