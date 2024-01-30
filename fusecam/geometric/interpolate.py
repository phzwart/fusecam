import torch


def find_nearest(template_tensor, target_tensor, N):
    """
    Finds the N nearest neighbors for each point in the target tensor from the template tensor.

    Args:
        template_tensor (torch.Tensor): A tensor of shape (M, 3) representing M points in 3D space.
        target_tensor (torch.Tensor): A tensor of shape (N, 3) representing N points in 3D space.
        N (int): The number of nearest neighbors to find.

    Returns:
        torch.Tensor, torch.Tensor: Two tensors representing the indices and distances of the N nearest neighbors.
    """
    # Calculate distances
    distances = torch.cdist(target_tensor, template_tensor, p=2)  # p=2 for Euclidean distance
    # Find the indices of the N nearest points
    indices = torch.topk(distances, N, largest=False, sorted=True).indices
    nearest_distances = torch.topk(distances, N, largest=False, sorted=True).values

    return indices, nearest_distances


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


def inverse_distance_weighting_with_weights(template_values, nearest_indices, weights):
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
