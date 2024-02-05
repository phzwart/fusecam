import pytest
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim

import mm3dtestdata as mm3d
import fusecam
from fusecam.geometric import space, embedplane, interpolate
from fusecam.manipimg import rotate_tensor_cube
from fusecam.aiutil import train_scripts, ensembling
import einops
from dlsia.core.networks import sms3d
from dlsia.core import helpers


def test_3d_mapping_mc():
    scale = 32
    sigma_low = 3.0
    sigma_high = .5

    # Generate class maps
    class_map_0 = torch.zeros((scale, scale, scale)).type(torch.LongTensor).numpy()
    class_map_0[10:-10, 10:-10, 10:-10] = 1
    class_map_0[13:-13, 13:-13, 13:-13] = 2

    class_map_low = mm3d.blur_it(class_map_0, sigma=sigma_low)
    class_map_high = mm3d.blur_it(class_map_0, sigma=sigma_high)




    tomo_class_0 = np.array([0])
    tomo_class_1 = np.array([2.00])
    tomo_class_2 = np.array([4.00])

    class_actions_tomo = np.column_stack([tomo_class_0, tomo_class_1, tomo_class_2]).T



    spectral_class_0 = np.array([1.0,0.0,0.0])
    spectral_class_1 = np.array([0.0,1.0,0.0])
    spectral_class_2 = np.array([0.0,0.0,1.0])

    class_actions_spec = np.column_stack([spectral_class_0,
                                          spectral_class_1,
                                          spectral_class_2,]).T



    low_map = mm3d.compute_weighted_map(class_map_low, class_actions_tomo)
    high_map = mm3d.compute_weighted_map(class_map_high, class_actions_spec)
    low_map = low_map + mm3d.noise(low_map, 0.01, 0.0)
    high_map = high_map + mm3d.noise(high_map, 0.01, 0.0)

    # Define spatial and plane metrics (omitted for brevity)
    space_object = space.SpatialVolumeMetric(origin=(0, 0, 0),
                                             step_size=(1, 1, 1),
                                             orientation=torch.eye(3),
                                             translation=(0, 0, 0),
                                             )
    plane_object = space.SpatialPlaneMetric(origin=(0, 0),
                                            step_size=(1, 1),
                                            orientation=torch.eye(2),
                                            translation=(0, 0))



    # Define plane and align points (omitted for brevity)
    u = torch.linspace(0, scale - 1, scale)
    U, V = torch.meshgrid(u, u, indexing='ij')
    UV = torch.concat([U.flatten().reshape(1, -1), V.flatten().reshape(1, -1)]).T
    UV = plane_object.to_plane_coordinates(UV)


    x = torch.linspace(0, scale - 1, scale)
    X, Y, Z = torch.meshgrid(x, x, x, indexing="ij")
    XYZ = torch.concat([X.flatten().reshape(1, -1), Y.flatten().reshape(1, -1), Z.flatten().reshape(1, -1), ]).T
    XYZ = space_object.to_lab_coordinates(XYZ)

    aligner_1 = embedplane.Plane3DAligner(
        normal=[0.0, 0.0, 1.0],
        point_on_plane=[scale // 2, scale // 2, scale // 2]
    )
    point_on_plane_2D_1 = (scale // 2, scale // 2)
    aligned_points_1 = aligner_1.align_points_to_3d(UV, point_on_plane_2D_1, rotation_angle=0)

    aligner_2 = embedplane.Plane3DAligner(
        normal=[0.0, 1.0, 0.0],
        point_on_plane=[scale // 2, scale // 2, scale // 2]
    )
    point_on_plane_2D_2 = (scale // 2, scale // 2)
    aligned_points_2 = aligner_2.align_points_to_3d(UV, point_on_plane_2D_2, rotation_angle=0)

    aligner_3 = embedplane.Plane3DAligner(
        normal=[1.0, 0.0, 0.0],
        point_on_plane=[scale // 2, scale // 2, scale // 2]
    )
    point_on_plane_2D_3 = (scale // 2, scale // 2)
    aligned_points_3 = aligner_3.align_points_to_3d(UV, point_on_plane_2D_3, rotation_angle=0)

    indices_1, near_dist_1 = interpolate.find_nearest(XYZ, aligned_points_1, 5)
    weights_1 = interpolate.compute_weights(near_dist_1, power=3.0, cutoff=2.0)

    indices_2, near_dist_2 = interpolate.find_nearest(XYZ, aligned_points_2, 5)
    weights_2 = interpolate.compute_weights(near_dist_2, power=3.0, cutoff=2.0)

    indices_3, near_dist_3 = interpolate.find_nearest(XYZ, aligned_points_3, 5)
    weights_3 = interpolate.compute_weights(near_dist_3, power=3.0, cutoff=2.0)

    # Interpolate and create dataset (omitted for brevity)
    flat_high_map = einops.rearrange( torch.Tensor(high_map), "C X Y Z -> (X Y Z) C ")
    gt_1 = interpolate.inverse_distance_weighting_with_weights(flat_high_map,
                                                               indices_1,
                                                               weights_1)
    gt_2 = interpolate.inverse_distance_weighting_with_weights(flat_high_map,
                                                               indices_2,
                                                               weights_2)
    gt_3 = interpolate.inverse_distance_weighting_with_weights(flat_high_map,
                                                               indices_3,
                                                               weights_3)

    my_3d_maps = torch.concat([torch.Tensor(low_map).unsqueeze(0)])

    my_2d_maps = torch.concat([gt_1,
                               gt_2,
                               gt_3
                              ]).unsqueeze(0)

    my_weights = torch.concat([weights_1, weights_2, weights_3]).unsqueeze(0)

    my_indices = torch.concat([indices_1, indices_2, indices_3]).unsqueeze(0)

    my_data = TensorDataset(my_3d_maps, my_2d_maps, my_weights, my_indices)
    data_loader = DataLoader(my_data, batch_size=1)

    # Define and train networks (omitted for brevity)
    n_networks = 3
    networks = ensembling.construct_3dsms_ensembler(n_networks=n_networks,
                                                    in_channels=1,
                                                    out_channels=3,
                                                    layers=10,
                                                    alpha=0.00,
                                                    gamma=0.00,
                                                    hidden_channels=[2],
                                                    parameter_bounds=[6000, 9000],
                                                    network_type="Regression"
                                                    )
    for net in networks:
        loss_function = nn.L1Loss()
        optimizer = optim.Adam(net.parameters(), lr=0.01)

        device = 'cpu'
        if torch.cuda.is_available():
            device='cuda'

        train_scripts.train_volume_on_slice(net,
                                            loss_function,
                                            optimizer,
                                            data_loader,
                                            250,
                                            interpolate.inverse_distance_weighting_with_weights, device=device)

        loss_function = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        train_scripts.train_volume_on_slice(net,
                                            loss_function,
                                            optimizer,
                                            data_loader,
                                            250,
                                            interpolate.inverse_distance_weighting_with_weights, device=device)

    m = 0.0
    s = 0.0
    with torch.no_grad():
        for net in networks:
            tmp3 = net.cpu()(torch.Tensor(low_map).unsqueeze(0))
            m += tmp3
            s += tmp3 ** 2.0

    m = m / n_networks
    s = torch.sqrt(s / n_networks - m * m)

    a = m.numpy()[0,0].flatten()
    b = high_map[0].flatten()
    ab0 = np.corrcoef(a, b)[0, 1]

    a = m.numpy()[0,1].flatten()
    b = high_map[1].flatten()
    ab1 = np.corrcoef(a, b)[0, 1]


    a = m.numpy()[0,2].flatten()
    b = high_map[2].flatten()
    ab2 = np.corrcoef(a, b)[0, 1]

    print(f"Correlation (Predicted-High) for channel 0: {ab0}")
    print(f"Correlation (Predicted-High) for channel 1: {ab1}")
    print(f"Correlation (Predicted-High) for channel 2: {ab2}")

    assert ab0 > 0.95
    assert ab1 > 0.90
    assert ab2 > 0.90



if __name__ == "__main__":
    test_3d_mapping_mc()

