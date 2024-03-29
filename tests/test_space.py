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


def random_rotation_matrix_3d():
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
    # Simple transformation test with translation
    spatial_metric = space.SpatialVolumeMetric(
        origin=[10, 10, 10],
        step_size=[1.0, 0.5, 0.5],
        orientation=torch.eye(3),
        translation=[2.0, 2.0, 2.0]  # Adding translation
    )
    lab_coords = spatial_metric.to_lab_coordinates([5, 10, 10])
    ref_coords = torch.tensor([7., 6., 6.])  # Adjusted for translation
    delta = torch.linalg.norm(ref_coords - lab_coords)
    assert(delta.item() < 1e-6)

    # Inverse orientation test with translation
    spatial_metric = space.SpatialVolumeMetric(
        origin=[0, 0, 0],
        step_size=[1.0, 0.5, 0.5],
        orientation=-torch.eye(3),
        translation=[-2.0, -2.0, -2.0]  # Adding translation
    )
    lab_coords = spatial_metric.to_lab_coordinates([5, 10, 10])
    ref_coords = torch.tensor([-7, -6, -6])  # Adjusted for translation
    delta = torch.linalg.norm(ref_coords - lab_coords)
    assert(delta.item() < 1e-6)

    # Random rotation test with translation
    for _ in range(10):
        matrix = random_rotation_matrix_3d()
        translation = np.random.uniform(-1, 1, 3)  # Random translation
        spatial_metric = space.SpatialVolumeMetric(
            origin=[0, 0, 0],
            step_size=[1.0, 1.0, 1.0],
            orientation=matrix,
            translation=translation)
        tensor_indices = [1, 1, 1]
        lab_coords = spatial_metric.to_lab_coordinates(tensor_indices)
        lab_coordinates = np.dot(matrix, np.array(tensor_indices)) + translation
        assert np.linalg.norm(lab_coordinates - lab_coords.numpy()) < 1e-5

    # Round trip test with translation
    for _ in range(10):
        matrix = random_rotation_matrix_3d()
        translation = np.random.uniform(-1, 1, 3)  # Random translation
        step_size = np.random.uniform(0.25, 2.0, 3)
        spatial_metric = space.SpatialVolumeMetric(
            origin=[0, 0, 0],
            step_size=step_size,
            orientation=matrix,
            translation=translation)
        tmp_in = np.random.uniform(3, 5, (10, 3))
        lab_coords = spatial_metric.to_lab_coordinates(tmp_in)
        tmp = spatial_metric.to_tensor_indices(lab_coords).detach()
        assert torch.linalg.norm(tmp - tmp_in).item() < 1e-5


def test_object_2d():
    # Simple transformation test with translation
    spatial_metric = space.SpatialPlaneMetric(
        origin=[5, 5],
        step_size=[1.0, 1.0],
        orientation=torch.eye(2),
        translation=[1.0, 1.0]  # Adding translation
    )
    plane_coords = spatial_metric.to_plane_coordinates([3, 4])
    ref_coords = torch.tensor([4., 5.])  # Adjusted for translation
    delta = torch.linalg.norm(ref_coords - plane_coords)
    assert(delta.item() < 1e-6)

    # Inverse orientation test with translation
    spatial_metric = space.SpatialPlaneMetric(
        origin=[2, 2],
        step_size=[1.0, 1.0],
        orientation=-torch.eye(2),
        translation=[1.0, -1.0]  # Adding translation
    )
    plane_coords = spatial_metric.to_plane_coordinates([1, 1])
    ref_coords = torch.tensor([4., 2.])  # Adjusted for translation
    delta = torch.linalg.norm(ref_coords - plane_coords)
    assert(delta.item() < 1e-6)

    # Random rotation test with translation
    for _ in range(10):
        matrix = random_rotation_matrix_2d()
        translation = np.random.uniform(-1, 1, 2)  # Random translation
        spatial_metric = space.SpatialPlaneMetric(
            origin=[0, 0],
            step_size=[1.0, 1.0],
            orientation=matrix,
            translation=translation)
        plane_coords = spatial_metric.to_plane_coordinates([1, 1])
        plane_coordinates = np.dot(matrix, np.array([1, 1])) + translation
        assert np.linalg.norm(plane_coordinates - plane_coords.numpy()) < 1e-5

    # Round trip test with translation
    for _ in range(10):
        matrix = random_rotation_matrix_2d()
        translation = np.random.uniform(-1, 1, 2)  # Random translation
        step_size = np.random.uniform(0.25, 2.0, 2)
        spatial_metric = space.SpatialPlaneMetric(
            origin=[0, 0],
            step_size=[1, 1],  # step_size,
            orientation=matrix,
            translation=translation)
        tmp_in = np.random.uniform(3, 5, (10, 2))
        plane_coords = spatial_metric.to_plane_coordinates(tmp_in)
        tmp = spatial_metric.to_tensor_indices(plane_coords).detach()
        assert torch.linalg.norm(tmp - tmp_in).item() < 1e-5


if __name__ == "__main__":
    test_object_2d()
    test_object_3d()
