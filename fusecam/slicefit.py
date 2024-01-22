import torch
import torch.nn as nn
import einops


class SliceLoss(nn.Module):
    def __init__(self,
                 base_loss,
                 cube_shape,
                 cube_metric,
                 slice_shape,
                 slice_metric):
        super(SliceLoss).__init__()
        self.base_loss = base_loss
        self.cube_shape = cube_shape
        self.slice_shape = slice_shape
        self.cube_metric = cube_metric
        self.slice_metric = slice_metric
        # precompute grids here



    def forward(self, cube_A, slice_B):
        # do stuff


