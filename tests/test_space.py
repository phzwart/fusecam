import numpy as np
import torch
import einops

from fusecam.geometric import space

import pytest


def random_rotation_matrix_2d():
    """
    Generate a random 2D rotation matrix.

    Returns:
    - numpy.ndarray
        A 2x2 numpy array representing the rotation matrix.
    """
    angle = np.random.uniform(0, 2 * np.pi)
    return np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])


def random_rotation_matrix():
    """
    Generate a random 3D rotation matrix using quaternions.

    Returns:
    - numpy.ndarray
        A 3x3 numpy array representing the rotation matrix.

    This function creates a rotation matrix representing a uniformly random
    rotation in 3D space, using quaternions to ensure uniform sampling from SO(3).
    """
    # Generate random quaternion components from a normal distribution
    quaternion = np.random.normal(size=4)

    # Normalize the quaternion to unit length
    quaternion /= np.linalg.norm(quaternion)

    # Quaternion components
    q0, q1, q2, q3 = quaternion

    # Construct the corresponding rotation matrix
    rot_matrix = np.array([
        [1 - 2 * q2 * q2 - 2 * q3 * q3, 2 * q1 * q2 - 2 * q3 * q0, 2 * q1 * q3 + 2 * q2 * q0],
        [2 * q1 * q2 + 2 * q3 * q0, 1 - 2 * q1 * q1 - 2 * q3 * q3, 2 * q2 * q3 - 2 * q1 * q0],
        [2 * q1 * q3 - 2 * q2 * q0, 2 * q2 * q3 + 2 * q1 * q0, 1 - 2 * q1 * q1 - 2 * q2 * q2]
    ])

    return rot_matrix



def test_object_3d():

    spatial_metric = space.SpatialVolumeMetric(
        origin=[10, 10, 10],
        step_size=[1.0, 0.5, 0.5],  # Different step sizes for Z, Y, X
        orientation=torch.eye(3)  # Identity matrix for simplicity
    )
    lab_coords = spatial_metric.to_lab_coordinates([5, 10, 10])  # Example indices
    ref_coords = torch.tensor([15., 15., 15.])
    delta = torch.linalg.norm(ref_coords - lab_coords)
    assert(delta.item() < 1e-6)

    spatial_metric = space.SpatialVolumeMetric(
        origin=[10, 10, 10],
        step_size=[1.0, 0.5, 0.5],  # Different step sizes for Z, Y, X
        orientation=-torch.eye(3)  # Identity matrix for simplicity
    )

    lab_coords = spatial_metric.to_lab_coordinates([5, 10, 10])  # Example indices
    ref_coords = torch.tensor([5., 5., 5.])
    delta = torch.linalg.norm(ref_coords - lab_coords)
    assert(delta.item() < 1e-6)

    for _ in range(10):
        matrix = random_rotation_matrix()
        spatial_metric = space.SpatialVolumeMetric(
            origin=[0, 0, 0],
            step_size=[1.0, 1.0, 1.0],
            orientation=matrix)
        lab_coords = spatial_metric.to_lab_coordinates([1, 1, 1])
        lab_coordinates = np.dot(matrix, np.array([1,1,1]))
        assert np.linalg.norm(lab_coordinates - lab_coords.numpy() ) < 1e-5

    for _ in range(10):
        matrix = random_rotation_matrix()
        step_size = np.random.uniform(0.25,2.0, 3)
        spatial_metric = space.SpatialVolumeMetric(
                origin=[0, 0, 0],
                step_size=[1,1,1] , #step_size,
                orientation=matrix)
        tmp_in = np.random.uniform(3,5,(10,3))
        lab_coords = spatial_metric.to_lab_coordinates(tmp_in)
        tmp = spatial_metric.to_tensor_indices(lab_coords).detach()
        assert torch.linalg.norm(tmp - tmp_in).item() < 1e-5

def test_object_2d():
    # Simple transformation test
    spatial_metric = space.SpatialPlaneMetric(
        origin=[5, 5],
        step_size=[1.0, 1.0],  # Equal step sizes for X and Y
        orientation=torch.eye(2)  # Identity matrix
    )
    plane_coords = spatial_metric.to_plane_coordinates([3, 4])
    ref_coords = torch.tensor([8., 9.])
    delta = torch.linalg.norm(ref_coords - plane_coords)
    assert(delta.item() < 1e-6)

    # Inverse orientation test
    spatial_metric = space.SpatialPlaneMetric(
        origin=[5, 5],
        step_size=[1.0, 1.0],
        orientation=-torch.eye(2)
    )
    plane_coords = spatial_metric.to_plane_coordinates([3, 4])
    ref_coords = torch.tensor([2., 1.])
    delta = torch.linalg.norm(ref_coords - plane_coords)
    assert(delta.item() < 1e-6)

    # Random rotation test
    for _ in range(10):
        matrix = random_rotation_matrix_2d()
        spatial_metric = space.SpatialPlaneMetric(
            origin=[0, 0],
            step_size=[1.0, 1.0],
            orientation=matrix)
        plane_coords = spatial_metric.to_plane_coordinates([1, 1])
        plane_coordinates = np.dot(matrix, np.array([1, 1]))
        assert np.linalg.norm(plane_coordinates - plane_coords.numpy()) < 1e-5

    # Round trip test
    for _ in range(10):
        matrix = random_rotation_matrix_2d()
        step_size = np.random.uniform(0.25, 2.0, 2)
        spatial_metric = space.SpatialPlaneMetric(
            origin=[0, 0],
            step_size=[1, 1],  # step_size,
            orientation=matrix)
        tmp_in = np.random.uniform(3, 5, (10, 2))
        plane_coords = spatial_metric.to_plane_coordinates(tmp_in)
        tmp = spatial_metric.to_tensor_indices(plane_coords).detach()
        assert torch.linalg.norm(tmp - tmp_in).item() < 1e-5


