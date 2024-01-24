import torch
import numpy as np
from fusecam.geometric.embedplane import Plane3DAligner

def test_round_trip():

    normal = np.random.uniform(-1,1, (3))
    point_on_plane = np.random.uniform(-1,1, (3))
    rotation_angle = np.random.uniform(-1,1)*180.0
    point_on_plane_2D = [0,0] #np.random.uniform(-1,1, (2))
    # Initialize Plane3DAligner
    aligner = Plane3DAligner(normal=normal, point_on_plane=point_on_plane)

    # Test data: points on a 2D plane
    test_points_2d = [[1., 2.], [3., 4.], [-1., -2.], [0., 0.]]

    # Convert 2D points to 3D
    points_3d = aligner.align_points_to_3d(test_points_2d, point_on_plane_2D, rotation_angle)

    # Convert back to 2D
    round_trip_points_2d = aligner.transform_3d_to_plane(points_3d)

    # Check if the points match (within a small tolerance)
    for original, round_trip in zip(test_points_2d, round_trip_points_2d):
        print(original, round_trip)
        assert torch.allclose(torch.tensor(original), round_trip, atol=1e-6), f"Round trip failed for point {original}"

if __name__ == "__main__":
    test_round_trip()
    print("All round trip tests passed.")
