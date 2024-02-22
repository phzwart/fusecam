import torch
import torch.nn as nn
from collections import OrderedDict
import einops
from fusecam.models.fcnetwork import FCNetworkND
from fusecam.manipimg.blur import GaussianBlur2D



class SemXctTransformer(nn.Module):
    def __init__(self, Cin, Cmiddle,sigma, dropout_rate=0.001, max_size=21):
        super(SemXctTransformer, self).__init__()
        self.Cin = Cin
        self.Cmiddle = Cmiddle
        self.Cout = 1
        self.dropout_rate = dropout_rate
        self.sigma = sigma
        self.max_size = max_size

        self.blur_object = GaussianBlur2D(sigma, self.max_size)
        self.sem_tomo_projector = FCNetworkND(Cin, Cmiddle, self.Cout, dropout_rate=dropout_rate)
        #self.prelu = nn.PReLU(num_parameters=1)


    def forward(self, x):
        # first FC
        x = self.sem_tomo_projector(x)
        y = self.blur_object(x)
        return x, y





