import torch
import torch.nn as nn
from collections import OrderedDict
import einops

class FCNetworkND(nn.Module):
    """
    A fully connected neural network that is dimension-agnostic, designed to process N-dimensional data.

    This network adapts to the dimensionality of the input data, reshaping it for processing through
    fully connected layers, and includes batch normalization, ReLU activations, and Dropout in each
    hidden layer.

    Parameters:
    - Cin (int): Number of input channels.
    - Cmiddle (list of int): Number of channels in the middle layers.
    - Cout (int): Number of output channels.
    - dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.05.

    Methods:
    - forward(x): Performs the forward pass with N-dimensional input.
    - topology_dict(): Returns a dictionary of the network topology.
    - save_network_parameters(name=None): Saves or returns the network parameters.
    """
    def __init__(self, Cin, Cmiddle, Cout, dropout_rate=0.05):
        """
        Initializes the network with the specified architecture.
        """
        super(FCNetworkND, self).__init__()
        self.Cin = Cin
        self.Cmiddle = Cmiddle
        self.Cout = Cout
        self.dropout_rate = dropout_rate

        # Initialize the layers of the network
        layers = [nn.Linear(self.Cin, self.Cmiddle[0]),
                  nn.BatchNorm1d(self.Cmiddle[0]),
                  nn.ReLU(),
                  nn.Dropout(self.dropout_rate)]

        for i in range(len(self.Cmiddle)-1):
            layers += [nn.Linear(self.Cmiddle[i], self.Cmiddle[i+1]),
                       nn.BatchNorm1d(self.Cmiddle[i+1]),
                       nn.ReLU(),
                       nn.Dropout(self.dropout_rate)]

        layers.append(nn.Linear(self.Cmiddle[-1], self.Cout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        original_shape = x.shape
        if len(original_shape) == 4: #2D case
            N,C,X,Y = original_shape
            x = einops.rearrange(x, "N C X Y -> (N Y X) C")
            x = self.network(x)
            x = einops.rearrange(x, "(N Y X) C -> N C X Y", N=N,X=X,Y=Y)
            return x

        elif len(original_shape) == 5: #3D case
            N, C, X, Y, Z = original_shape
            x = einops.rearrange(x, "N C X Y Z -> (N Y X Z ) C")
            x = self.network(x)
            x = einops.rearrange(x, "(N Y X Z) C -> N C X Y Z", N=N, X=X, Y=Y)
            return x
        else:
            raise ValueError(f"Unsupported input shape: {original_shape}. FCNetworkND expects 4D (N, C, H, W) or 5D (N, C, D, H, W) tensors.")

    def topology_dict(self):
        """
        Returns a dictionary representing the network's topology.

        Returns:
        - OrderedDict: A dictionary containing the network's architecture details.
        """
        topo = OrderedDict([
            ("Cin", self.Cin),
            ("Cmiddle", self.Cmiddle),
            ("Cout", self.Cout),
            ("dropout_rate", self.dropout_rate)
        ])
        return topo

    def save_network_parameters(self, name=None):
        """
        Saves the network parameters to a file or returns them as a dictionary.

        Parameters:
        - name (str, optional): The filename to save the parameters. If None, returns the dict.

        Returns:
        - OrderedDict: The network's parameters if `name` is None. Otherwise, the parameters are saved to the specified file.
        """
        network_dict = OrderedDict([
            ("type", "FCNetworkND"),
            ("topo_dict", self.topology_dict()),
            ("state_dict", self.state_dict())
        ])
        if name is None:
            return network_dict
        else:
            torch.save(network_dict, name)
