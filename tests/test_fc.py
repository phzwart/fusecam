import pytest
from fusecam.models.fcnetwork import FCNetworkND
import torch
import einops

@pytest.mark.parametrize("input_shape,Cmiddle,Cout", [
    ((1, 10, 24, 24, 24), [100, 50], 10),  # 3D input
    ((1, 10, 24, 24), [100, 50], 10),     # 2D input
])
def test_fc_network_nd_forward_shape(input_shape, Cmiddle, Cout):
    Cin = input_shape[1]  # Input channels based on the test case
    network = FCNetworkND(Cin, Cmiddle, Cout)
    input_tensor = torch.randn(input_shape)
    output = network(input_tensor)

    assert output.shape[0] == input_shape[0], "Batch size should remain unchanged."
    assert output.shape[1] == Cout, "Output channel size should match Cout."
    assert output.shape[2:] == input_shape[2:], "Spatial dimensions should remain unchanged."

def test_topology_dict():
    Cin, Cmiddle, Cout, dropout_rate = 10, [100, 50], 10, 0.05
    network = FCNetworkND(Cin, Cmiddle, Cout, dropout_rate)
    topo_dict = network.topology_dict()

    assert topo_dict["Cin"] == Cin, "Cin should match the initialized value."
    assert topo_dict["Cmiddle"] == Cmiddle, "Cmiddle should match the initialized value."
    assert topo_dict["Cout"] == Cout, "Cout should match the initialized value."
    assert topo_dict["dropout_rate"] == dropout_rate, "dropout_rate should match the initialized value."

# Optional: You can add more tests here to check other functionalities like save_network_parameters.
