import torch
import torch.nn.functional as F
from einops import rearrange

def extract_and_linearize_neighborhoods_with_channels(input_tensor, M):
    """
    Extracts neighborhoods around each pixel within a cube of size (2M+1)^3, including channel data,
    and linearizes them into vectors.

    Parameters:
    - input_tensor: A torch.Tensor of shape (C, X, Y, Z).
    - M: The neighborhood radius.

    Returns:
    - A torch.Tensor where each row is the linearized neighborhood of a pixel, including channel information.
    """
    C, X, Y, Z = input_tensor.shape
    # Pad the tensor to handle edge cases, padding applied to spatial dimensions only
    padded_input = F.pad(input_tensor, (M, M, M, M, M, M), mode='constant', value=0)

    # Use unfold to create the sliding window effect across X, Y, and Z dimensions
    # Unfolding spatial dimensions to capture (2M+1) neighborhoods
    windows = padded_input.unfold(1, 2*M+1, 1).unfold(2, 2*M+1, 1).unfold(3, 2*M+1, 1)

    # Now windows have shape (C, X', Y', Z', (2M+1), (2M+1), (2M+1))
    # where X', Y', and Z' are the new dimensions after unfolding
    # Linearize including the channel dimension
    # Note: We adjust the einops rearrange pattern to account for the correct dimensions
    linearized_windows = rearrange(windows, 'c x y z d1 d2 d3 -> (x y z) (c d1 d2 d3)')

    return linearized_windows


def circular_shift(tensor, shift, dims):
    """
    Apply circular shift to a tensor along specified dimensions.

    Args:
    - tensor (torch.Tensor): The tensor to shift.
    - shift (tuple): The shift amount for each dimension.
    - dims (tuple): The dimensions to apply the shift.

    Returns:
    - torch.Tensor: The shifted tensor.
    """
    shifted_tensor = tensor.clone()
    for dim, s in zip(dims, shift):
        shifted_tensor = shifted_tensor.roll(s, dims=dim)
    return shifted_tensor

def produce_individual_tensors(input_tensor, d1, d2, d3, C, X, Y, Z):
    """
    Produce d1*d2*d3 individual tensors from reshaped slices, applying circular shifts.

    Args:
    - reshaped_tensor (torch.Tensor): Tensor of shape (x, y, z, c, d1, d2, d3).
    - d1, d2, d3 (int): Dimensions of the local neighborhood.
    - C, X, Y, Z (int): Original tensor dimensions.

    Returns:
    - List[torch.Tensor]: A list of shifted tensors of shape (C, X, Y, Z) each.
    """

    reshaped_tensor = rearrange(input_tensor,
                                "(X Y Z) (C d1 d2 d3) -> X Y Z C d1 d2 d3",
                                X=X,Y=Y,Z=Z,C=C,d1=d1,d2=d2,d3=d3)
    individual_tensors = []

    # Calculate center of the local neighborhood
    center = (d1 // 2, d2 // 2, d3 // 2)

    for i in range(d1):
        for j in range(d2):
            for k in range(d3):
                # Determine shift needed to align current slice with original position
                shift = (i - center[0], j - center[1], k - center[2])

                # Initialize an empty tensor for each shift
                shifted_tensor = torch.zeros((C, X, Y, Z))

                # Apply circular shift for each channel and the current slice
                for c in range(C):
                    # Extract and shift the slice
                    slice_to_shift = reshaped_tensor[:, :, :, c, i, j, k]
                    shifted_slice = circular_shift(slice_to_shift, shift, dims=(0, 1, 2))

                    # Assign the shifted slice back to the corresponding channel
                    shifted_tensor[c] = shifted_slice

                # Store the shifted tensor in the list
                individual_tensors.append(shifted_tensor.unsqueeze(0))

    return torch.concat(individual_tensors)

