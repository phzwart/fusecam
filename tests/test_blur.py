import numpy as np
import torch
import einops

import matplotlib.pyplot as plt
from fusecam import blur

from scipy.ndimage import gaussian_filter

import pytest

def test_all():
    N = 64
    img = torch.zeros((1,N,N,N))
    img[:, 20:-20,20:-20,20:-20] = 1.0
    mask = img[0].numpy()
    windows = np.arange(1,15)*2+1
    sigmas = np.linspace(0.25, 5.25, 10)
    eps = 1e-12
    threshold = 6

    for sigma in sigmas:
        for window in windows:
            tmp = blur.GaussianBlur3D(sigma,window)(img)
            tmp2 = gaussian_filter(img[0].numpy(), sigma)
            d = np.abs( tmp.detach().numpy() - tmp2 ) / (tmp2+eps)
            d = d*mask
            if window / sigma >  threshold:
                assert np.max(d) < 1.0e-2

