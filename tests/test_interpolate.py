import torch
from fusecam.geometric.interpolate import find_nearest
from fusecam.geometric.interpolate import inverse_distance_weighting_with_weights
from fusecam.geometric.interpolate import compute_weights
import pytest
import numpy as np
import einops




def create_3D_grid():
    x = y = z = torch.linspace(-1, 1, 32)
    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
    grid = torch.stack((xx.flatten(), yy.flatten(), zz.flatten()), dim=1)
    values = 1.0 + grid[:, 0]**2 + grid[:, 1]**2 + grid[:, 2]**2
    return grid, values

def create_target_coords():
    x = y = torch.linspace(-1, 1, 100)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    zz = 0.0*torch.ones_like(xx) / np.sqrt(3)
    target_coords = torch.stack((xx.flatten(), yy.flatten(), zz.flatten()), dim=1)
    return target_coords

def test_find_nearest():
    grid, _ = create_3D_grid()
    target_coords = create_target_coords()
    N = 5
    indices, distances = find_nearest(grid, target_coords, N)
    assert indices.shape == (10000, N)
    assert distances.shape == (10000, N)
    assert torch.all(distances >= 0)

def test_compute_weights():
    grid, values = create_3D_grid()
    target_coords = create_target_coords()
    _, distances = find_nearest(grid, target_coords,  5)
    weights = compute_weights(distances, cutoff=0.5)
    assert weights.shape == distances.shape
    assert torch.all(torch.isnan(weights) | (weights >= 0))  # Check for non-negative weights or NaN

def test_inverse_distance_weighting_with_weights():
    grid, values = create_3D_grid()
    target_coords = create_target_coords()
    indices, distances = find_nearest(grid, target_coords, 16)
    weights = compute_weights(distances, power=3.0, cutoff=0.15)
    interpolated_values = inverse_distance_weighting_with_weights(values, indices, weights)
    assert interpolated_values.shape == (target_coords.shape[0],)
    import matplotlib.pyplot as plt
    interpolated_values = einops.rearrange(interpolated_values, "(X Y)-> X Y", Y=100, X=100 ).numpy()

    funct = 1.0 + target_coords[:,0]**2 + target_coords[:,1]**2 + target_coords[:,2]**2
    funct = einops.rearrange(funct, "(X Y)-> X Y", Y=100, X=100 ).numpy()
    assert np.max( (np.abs(interpolated_values - funct)) / funct )  < 5e-2
    # Additional checks can be added for expected behavior (e.g., NaN handling)

if __name__ == "__main__":
    test_find_nearest()
    test_compute_weights()
    test_inverse_distance_weighting_with_weights()
