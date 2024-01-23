import torch
from fusecame.geometric.space import SpatialVolumeMetric
from fusecame.geometric.space import SpatialPlaneMetric

class Plane3DAligner:
    def __init__(self, normal, point_on_plane):
        # ... (existing initialization code)

    def _compute_rotation_matrix(self, uvw_basis):
        # ... (existing rotation matrix computation code)

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
        plane_points = torch.tensor(plane_points, dtype=torch.float32)
        point_on_plane_2D = torch.tensor(point_on_plane_2D, dtype=torch.float32)

        # Shift plane points by point_on_plane_2D
        shifted_plane_points = plane_points - point_on_plane_2D

        # Apply rotation to shifted points
        angle_rad = np.radians(rotation_angle)
        plane_rotation = torch.tensor([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad),  np.cos(angle_rad)]
        ])
        rotated_points = torch.matmul(plane_rotation, shifted_plane_points.T).T

        # Define and align the u'v'w' basis vectors
        uvw_basis = torch.eye(3)
        uvw_basis[:2, :2] = plane_rotation
        self.rotation_matrix = self._compute_rotation_matrix(uvw_basis)
        aligned_points = torch.matmul(self.rotation_matrix, rotated_points.T).T

        # Translate the aligned points by the 3D point on the plane
        aligned_points += self.point_on_plane

        return aligned_points

# Example usage
aligner = Plane3DAligner(normal=[0, 0, 1], point_on_plane=[0, 0, 0])
plane_points = [[1, 0], [0, 1]]  # Points on the 2D plane
point_on_plane_2D = [0.5, 0.5]   # Specific point in 2D plane's local coordinates
aligned_points = aligner.align_points_to_3d(plane_points, point_on_plane_2D, rotation_angle=45)
print(aligned_points)
