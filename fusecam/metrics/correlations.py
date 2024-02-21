import torch
from fusecam.manipimg import blur
import numpy as np
import einops

def pearson_correlation(tensor1, tensor2):
    """
    Computes the Pearson correlation coefficient between two tensors channel-wise.

    This function calculates the Pearson correlation coefficient for each channel
    across potentially batched (N), X, Y, Z dimensions, averaging the values over
    N, X, Y, Z if those dimensions are present.

    Parameters:
    - tensor1 (torch.Tensor): A tensor of shape (N,C,X,Y,Z) or (C,X,Y,Z).
    - tensor2 (torch.Tensor): A tensor of shape (N,C,X,Y,Z) or (C,X,Y,Z), must have the
      same shape as tensor1.

    Returns:
    - torch.Tensor: A 1D tensor containing the Pearson correlation coefficient for each
      channel.
    """
    # Ensure both tensors have the same shape
    assert tensor1.shape == tensor2.shape, "Tensors do not have the same shape"

    # Add a batch dimension if not present
    if len(tensor1.shape) == 4:
        tensor1 = tensor1.unsqueeze(0)
        tensor2 = tensor2.unsqueeze(0)

    # Ensure tensors are at least 5D now
    assert tensor1.dim() == 5, "Dimension should be 5. Check Input"

    # Compute means and standard deviations over N, X, Y, Z, keeping the channel dimension

    tensor1 = einops.rearrange(tensor1, "N C X Y Z -> (N X Y Z) C ")
    tensor2 = einops.rearrange(tensor2, "N C X Y Z -> (N X Y Z) C ")
    mean1 = tensor1.mean(dim=0)
    mean2 = tensor2.mean(dim=0)
    std1 = tensor1.std(dim=0)
    std2 = tensor2.std(dim=0)
    # Compute Pearson correlation coefficient per channel
    covariance = ((tensor1 - mean1.unsqueeze(0)) * (tensor2 - mean2.unsqueeze(0))).mean(dim=0)
    correlation = covariance / (std1 * std2)
    return correlation

def blur_and_correlate(to_blur, not_to_blur, blur_range=np.arange(0, 5.25, 0.25)):
    """
    Applies Gaussian blur with varying sigma to a tensor and calculates the Pearson
    correlation coefficient with another non-blurred tensor for each sigma value.

    Parameters:
    - to_blur (torch.Tensor): The tensor to apply Gaussian blur to. Expected shape (N,C,X,Y,Z) or (C,X,Y,Z).
    - not_to_blur (torch.Tensor): The tensor to compare against, without applying blur. Same shape requirements.
    - blur_range (np.ndarray): An array of sigma values to use for the Gaussian blur. Default range is 0 to 5 with 0.25 steps.

    Returns:
    - (np.ndarray, list): A tuple containing the array of sigma values used for blurring and a list of 1D tensors representing
      the Pearson correlation coefficient for each channel and blur level.
    """
    with torch.no_grad():
        results = []
        for sigma in blur_range:
            blurrer = blur.GaussianBlur3D(initial_sigma=sigma)
            blurred = torch.stack([blurrer(m.unsqueeze(0))[0] for m in to_blur])
            ccs = pearson_correlation(blurred, not_to_blur)
            results.append(ccs)
        return blur_range, results








