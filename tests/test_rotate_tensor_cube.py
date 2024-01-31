import torch
from einops import rearrange
from fusecam.manipimg.rotate_tensor_cube import rotate_90

def test_rotate_90():
    # Define a 3x3x3 tensor with known values
    tensor = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                           [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                           [[19, 20, 21], [22, 23, 24], [25, 26, 27]]])

    # Expected tensors after rotation
    expected_x = torch.tensor([[[7, 4, 1], [8, 5, 2], [9, 6, 3]],
                               [[16, 13, 10], [17, 14, 11], [18, 15, 12]],
                               [[25, 22, 19], [26, 23, 20], [27, 24, 21]]])


    expected_y = torch.tensor([[[ 3, 12, 21],
                                [ 6, 15, 24],
                                [ 9, 18, 27]],
                               [[ 2, 11, 20],
                                [ 5, 14, 23],
                                [ 8, 17, 26]],
                               [[ 1, 10, 19],
                                [ 4, 13, 22],
                                [ 7, 16, 25]]])

    expected_z = torch.tensor([[[19, 20, 21],
                                [10, 11, 12],
                                [ 1,  2,  3]],
                               [[22, 23, 24],
                                [13, 14, 15],
                                [ 4,  5,  6]],
                               [[25, 26, 27],
                                [16, 17, 18],
                                [ 7,  8,  9]]])

    # Perform rotations
    rotated_x = rotate_90(tensor, 'x')
    rotated_y = rotate_90(tensor, 'y')
    rotated_z = rotate_90(tensor, 'z')
    # Assertions
    assert torch.equal(rotated_x, expected_x), "Rotation around X-axis failed"
    assert torch.equal(rotated_y, expected_y), "Rotation around Y-axis failed"
    assert torch.equal(rotated_z, expected_z), "Rotation around Z-axis failed"

