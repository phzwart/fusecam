import torch
from einops import rearrange

def rotate_90(tensor, axis):
    """
    Rotate a 3D tensor by 90 degrees around a specified axis.

    Parameters:
        tensor (torch.Tensor): A 3D tensor (cube).
        axis (str): The axis to rotate around ('x', 'y', or 'z').

    Returns:
        torch.Tensor: The rotated tensor.
    """
    if axis == 'x':
        # Rotate 90 degrees around X-axis
        return rearrange(tensor, 'i j k -> i k j').flip(dims=[2])
    elif axis == 'y':
        # Rotate 90 degrees around Y-axis
        return rearrange(tensor, 'i j k -> k j i').flip(dims=[0])
    elif axis == 'z':
        # Rotate 90 degrees around Z-axis
        return rearrange(tensor, 'i j k -> j i k').flip(dims=[1])
    else:
        raise ValueError("Invalid axis. Choose 'x', 'y', or 'z'.")
