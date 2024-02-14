import pytest
import torch
import numpy as np
from fusecam.metrics.correlations import pearson_correlation, blur_and_correlate


def create_test_tensor():
    """
    Creates a test tensor of shape (2, 32, 32, 32) with a 9x9x9 patch filled with 1s in the center.
    """
    tensor = torch.zeros((2, 32, 32, 32))
    center = [slice(32//2 - 4, 32//2 + 5)] * 3  # Center patch
    tensor[(slice(None),) + tuple(center)] = 1
    return tensor

def test_pearson_correlation():
    """
    Test the pearson_correlation function with identical tensors to ensure correlation is 1.
    """
    tensor = create_test_tensor()
    correlation = pearson_correlation(tensor, tensor)
    # Expect the correlation to be close to 1 for each channel
    assert torch.allclose(correlation, torch.tensor([1.0, 1.0]), atol=1e-6), "Correlation should be close to 1"

def test_blur_and_correlate():
    """
    Test the blur_and_correlate function with a simple blur range and ensure it runs without errors.
    """
    with torch.no_grad():
        to_blur = create_test_tensor()
        not_to_blur = create_test_tensor()
        blur_range, results = blur_and_correlate(to_blur, not_to_blur, blur_range=np.arange(0, 1.0, 0.5))

        # Check if results are returned for each blur level
        assert len(results) == len(blur_range), "Should return results for each blur level"
        assert torch.all(torch.concat(results) > 0.98)
    # Optionally, check the type or specific values within results, depending on expected behavior
