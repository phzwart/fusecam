import torch
import faiss
import numpy as np

def find_nearest(template_tensor, target_tensor, N):
    # Convert PyTorch tensors to NumPy arrays and ensure they are C-contiguous
    template_np = np.ascontiguousarray(template_tensor.detach().numpy())
    target_np = np.ascontiguousarray(target_tensor.detach().numpy())

    # Create a FAISS index
    dimension = template_np.shape[1]  # Assuming 3D points
    index = faiss.IndexFlatL2(dimension)

    # Add the template tensor to the index
    index.add(template_np)

    # Search for the nearest neighbors
    distances, indices = index.search(target_np, N)

    return torch.tensor(indices), torch.tensor(distances)

def compute_weights(nearest_distances, power=2.0, cutoff=None):
    """
    Computes the weights for inverse distance weighting using a given power parameter and applies a distance cutoff.

    Args: nearest_distances (torch.Tensor): A tensor of distances to the nearest neighbors. power (float, optional):
    The power parameter for the inverse distance weighting. Default is 2.0. cutoff (float, optional): The cutoff
    distance value. Points with no neighbors closer than this value will have NaN weights. Default is None,
    which means no cutoff.

    Returns: torch.Tensor: A tensor of normalized weights. Weights are NaN for points with no neighbors within the
    cutoff distance.
    """
    # Initialize weights tensor
    weights = torch.zeros_like(nearest_distances)

    # Handle zero distances to avoid division by zero
    nearest_distances[nearest_distances == 0] = 1e-5

    # Check if cutoff is applied
    if cutoff is not None:
        # Mask for neighbors within the cutoff
        within_cutoff = nearest_distances < cutoff

        # Check if there are any neighbors within the cutoff for each target point
        has_neighbors_within_cutoff = torch.any(within_cutoff, dim=1)

        # Inverse of the distances raised to the given power
        inv_distances = 1.0 / nearest_distances ** power

        # Apply cutoff
        inv_distances[~within_cutoff] = 0

        # Normalize weights
        sum_inv_distances = torch.sum(inv_distances, axis=1, keepdim=True)
        weights[has_neighbors_within_cutoff] = inv_distances[has_neighbors_within_cutoff] / sum_inv_distances[
            has_neighbors_within_cutoff]

        # Set weights to NaN for points without any neighbors within the cutoff
        weights[~has_neighbors_within_cutoff] = float('nan')
    else:
        # Inverse of the distances raised to the given power
        inv_distances = 1.0 / nearest_distances ** power

        # Normalize weights
        weights = inv_distances / torch.sum(inv_distances, axis=1, keepdim=True)

    return weights


def inverse_distance_weighting_with_weights_SC(template_values, nearest_indices, weights):
    """
    Performs inverse distance weighting interpolation using precomputed weights.

    Args:
        template_values (torch.Tensor): A tensor of values corresponding to each point in the template.
        nearest_indices (torch.Tensor): Indices of the nearest neighbors for each point in the target.
        weights (torch.Tensor): Precomputed weights for each of the nearest neighbors.

    Returns:
        torch.Tensor: A tensor representing interpolated values at the target coordinates.
    """
    # Use the precomputed weights to interpolate the values
    interpolated_values = torch.sum(template_values[nearest_indices] * weights, axis=1)

    return interpolated_values

def inverse_distance_weighting_with_weights_MC(template_values, nearest_indices, weights):
    """
    Adjusted to support input shapes and return a (C, K) tensor.

    Args:
        template_values (torch.Tensor): Shape (M, C), values for each template point.
        nearest_indices (torch.Tensor): Shape (K, nearest), indices of nearest points for each target.
        weights (torch.Tensor): Shape (K, nearest), weights for each of the nearest neighbors.

    Returns:
        torch.Tensor: Shape (C, K), interpolated values for each feature across all targets.
    """
    # Ensure weights sum to 1 across the nearest dimension for proper interpolation
    normalized_weights = weights / weights.sum(dim=1, keepdim=True)

    # Select the nearest template values for each target point
    # template_values_selected shape will be (K, nearest, C)
    template_values_selected = template_values[nearest_indices]

    # Perform weighted sum across the nearest neighbors, resulting in shape (K, C)
    # We use einsum for better control over the multiplication and summation axes
    interpolated_values = torch.einsum('knc,kn->kc', template_values_selected, normalized_weights)

    # Transpose the result to get shape (C, K)
    # interpolated_values = interpolated_values.transpose(0, 1)

    return interpolated_values

def inverse_distance_weighting_with_weights(template_values, nearest_indices, weights):
    if len(template_values.shape) == 1:
        return inverse_distance_weighting_with_weights_SC(template_values, nearest_indices, weights)
    else:
        return inverse_distance_weighting_with_weights_MC(template_values, nearest_indices, weights)

