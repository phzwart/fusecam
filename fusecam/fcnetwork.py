import torch
import torch.nn as nn
import einops


class FCNetwork3D(nn.Module):
    """
    A fully connected neural network for 3D data.

    This network is designed to process 3D data by reshaping it and passing it through multiple
    fully connected layers. Batch normalization, ReLU activations, and Dropout are used in each
    hidden layer.

    Parameters:
    - Cin (int): Number of input channels.
    - Cmiddle (list of int): Number of channels in the middle layers.
    - Cout (int): Number of output channels.
    - dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.5.

    Methods:
    - forward: Defines the forward pass of the network.
    - topology_dict: Returns a dictionary of the network topology.
    - save_network_parameters: Saves the network parameters to a file or returns them as a dict.
    """
    def __init__(self, Cin, Cmiddle, Cout, dropout_rate=0.05):
        """
        Initializes the network with the specified architecture.
        """
        super(FCNetwork3D, self).__init__()

        self.Cin = Cin
        self.Cmiddle = Cmiddle
        self.Cout = Cout
        self.dropout_rate = dropout_rate

        layers = []
        layers.append(nn.Linear(self.Cin, self.Cmiddle[0]))
        layers.append(nn.BatchNorm1d(self.Cmiddle[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout_rate))

        for i in range(len(self.Cmiddle)-1):
            layers.append(nn.Linear(self.Cmiddle[i], self.Cmiddle[i+1]))
            layers.append(nn.BatchNorm1d(self.Cmiddle[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))

        layers.append(nn.Linear(self.Cmiddle[-1], self.Cout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the network. Reshapes the input and passes it through the layers.

        Parameters:
        - x (Tensor): The input tensor with shape (N, C, Z, Y, X).

        Returns:
        - Tensor: The output tensor after processing by the network.
        """
        N,C,Z,Y,X = x.shape
        x = einops.rearrange(x, "N C Z Y X -> (N Z Y X) C")
        x = self.network(x)
        x = einops.rearrange(x, "(N Z Y X) C -> N C Z Y X", N=N, Z=Z, Y=Y, X=X)
        return x

    def topology_dict(self):
        """
        Returns a dictionary representing the network's topology.

        Returns:
        - OrderedDict: A dictionary containing the network's architecture details.
        """
        topo = OrderedDict()
        topo["Cin"] = self.Cin
        topo["Cmiddle"] = self.Cmiddle
        topo["Cout"] = self.Cout
        topo["dropout_rate"] = self.dropout_rate
        return topo

    def save_network_parameters(self, name=None):
        """
        Saves the network parameters to a file or returns them as a dictionary.

        Parameters:
        - name (str, optional): The filename to save the parameters. If None, returns the dict.

        Returns:
        - OrderedDict: The network's parameters if `name` is None.
        """
        network_dict = OrderedDict()
        network_dict["type"] = "FCNetwork3D"
        network_dict["topo_dict"] = self.topology_dict()
        network_dict["state_dict"] = self.state_dict()
        if name is None:
            return network_dict
        torch.save(network_dict, name)
