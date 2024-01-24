import torch
import numpy as np
from fusecam.geometric.space import SpatialVolumeMetric
from fusecam.geometric.space import SpatialPlaneMetric

class Plane3DAligner:
    def __init__(self, normal, point_on_plane):
        """
        Initializes the Plane3DAligner object.

        Parameters:
            normal (array-like): The normal vector [A, B, C] of the plane in 3D space.
            point_on_plane (array-like): A point [x0, y0, z0] on the plane in 3D coordinates.
        """
        self.normal = torch.Tensor(normal)
        self.normal /= torch.norm(self.normal)  # Ensure it's a unit vector
        self.point_on_plane = torch.Tensor(point_on_plane)
        self.rotation_matrix = None

    def _compute_rotation_matrix(self, uvw_basis):
        # Extract the w vector from the basis
        w_vector = uvw_basis[:, 2]

        # Calculate the rotation axis (cross product of w_vector and normal)
        rotation_axis = torch.cross(w_vector, self.normal)

        # Check if w_vector and self.normal are nearly parallel
        if torch.isclose(rotation_axis.norm(), torch.tensor(0.0), atol=1e-6):
            # If nearly parallel, check if they are pointing in the same direction
            if torch.isclose(torch.dot(w_vector, self.normal), torch.tensor(1.0), atol=1e-6):
                # If they are in the same direction, return identity matrix (no rotation needed)
                return torch.eye(3)
            else:
                # If they are in opposite directions, rotate 180 degrees around any perpendicular axis
                # For simplicity, choose a perpendicular axis in the plane of one of the other basis vectors
                perpendicular_axis = torch.cross(w_vector, uvw_basis[:, 0])  # Using u_vector
                perpendicular_axis /= perpendicular_axis.norm()  # Normalize the axis
                theta = torch.tensor(np.pi)  # 180 degrees in radians
                # Rodrigues' rotation formula for 180 degree rotation
                K = torch.tensor([
                    [0, -perpendicular_axis[2], perpendicular_axis[1]],
                    [perpendicular_axis[2], 0, -perpendicular_axis[0]],
                    [-perpendicular_axis[1], perpendicular_axis[0], 0]
                ])
                return 2 * torch.matmul(K, K) - torch.eye(3)
        else:
            rotation_axis /= rotation_axis.norm()  # Normalize the axis
            # Calculate the rotation angle (arccosine of the dot product of normalized vectors)
            cos_theta = torch.dot(w_vector, self.normal)
            theta = torch.acos(cos_theta)
            # Rodrigues' rotation formula components
            K = torch.tensor([
                [0, -rotation_axis[2], rotation_axis[1]],
                [rotation_axis[2], 0, -rotation_axis[0]],
                [-rotation_axis[1], rotation_axis[0], 0]
            ])
            I = torch.eye(3)
            K_square = torch.matmul(K, K)
            # Rotation matrix
            rotation_matrix = I + torch.sin(theta) * K + (1 - torch.cos(theta)) * K_square
            return rotation_matrix


    def align_points_to_3d(self, plane_points, point_on_plane_2D, rotation_angle):
        """
        Aligns points from the 2D plane to the 3D plane using the specified rotation.

        Parameters:
            plane_points (array-like): Points on the 2D plane.
            point_on_plane_2D (array-like): A specific point in the 2D plane's local coordinates.
            rotation_angle (float): The rotation angle to apply to the plane around its origin.

        Returns:
            torch.Tensor: Points aligned in 3D coordinates.
        """
        # Convert points and point_on_plane_2D to tensors
        plane_points = torch.Tensor(plane_points)
        point_on_plane_2D = torch.Tensor(point_on_plane_2D)

        # Shift plane points by point_on_plane_2D
        shifted_plane_points = plane_points - point_on_plane_2D

        # Apply rotation to shifted points
        angle_rad = np.radians(rotation_angle)
        plane_rotation = torch.Tensor([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad),  np.cos(angle_rad)]
        ])

        rotated_points = torch.matmul(plane_rotation, shifted_plane_points.T).T
        zeros = torch.zeros(rotated_points.shape[0], 1)
        rotated_points = torch.cat((rotated_points, zeros), dim=1)

        # Define and align the u'v'w' basis vectors
        uvw_basis = torch.eye(3)
        uvw_basis[:2, :2] = plane_rotation
        self.rotation_matrix = self._compute_rotation_matrix(uvw_basis)

        aligned_points = torch.matmul(self.rotation_matrix, rotated_points.T).T

        # Translate the aligned points by the 3D point on the plane
        aligned_points += self.point_on_plane

        return aligned_points
    
    def transform_3d_to_plane(self, points_3d):
        """
        Transforms 3D points back into the 2D plane coordinates.

        Parameters:
            points_3d (array-like): Points in 3D coordinates.

        Returns:
            torch.Tensor: Points in 2D plane coordinates.
        """
        # Convert points_3d to tensor if it's not already
        points_3d = torch.Tensor(points_3d)

        # Translate the points by the negative of self.point_on_plane
        translated_points = points_3d - self.point_on_plane

        # Apply the inverse rotation (transpose of the rotation matrix)
        # since rotation_matrix is orthogonal
        rotated_points = torch.matmul(self.rotation_matrix.t(), translated_points.T).T

        # Project the points onto the 2D plane (ignore z component)
        plane_points = rotated_points[:, :2]

        return plane_points


if __name__ == "__main__":
    # Example usage
    aligner = Plane3DAligner(normal=[1., 0., 0.], point_on_plane=[0., 0., 0.])
    plane_points = [[1., 0.], [0., 1.]]  # Points on the 2D plane
    point_on_plane_2D = [0.0, 0.0]   # Specific point in 2D plane's local coordinates
    aligned_points = aligner.align_points_to_3d(plane_points, point_on_plane_2D, rotation_angle=0)
    print(aligned_points)
