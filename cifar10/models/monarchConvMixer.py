# https://github.com/lucidrains/mlp-mixer-pytorch/blob/main/mlp_mixer_pytorch/mlp_mixer_pytorch.py
from torch import nn
from functools import partial
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import init

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce

pair = lambda x: x if isinstance(x, tuple) else (x, x)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x):
        return self.fn(x) + x

def MonarchConvMixer(dim, depth, kernel_size=9, patch_size=7, n_classes=1000):
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        
        *[nn.Sequential(ConvMonarchMixerLayer(dim, dim, kernel_size))
        for i in range(depth)],

        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes))

def blockdiag_matmul(x, w):
    return torch.einsum("bnm,...bm->...bn", w, x.view(*x.shape[:-1], w.shape[0], w.shape[-1]) ).reshape(*x.shape)

class MonarchMatrix(nn.Module):
    def __init__(self, sqrt_n: int):
        super().__init__()
        self.sqrt_n = sqrt_n
        self.L = nn.Parameter(torch.randn((sqrt_n, sqrt_n, sqrt_n)))
        self.R = nn.Parameter(torch.randn((sqrt_n, sqrt_n, sqrt_n)))
        
    def forward(self, x):
        x = rearrange(x, "... (m n) -> ... (n m)", n=self.sqrt_n)
        x = blockdiag_matmul(x, self.L)
        x = rearrange(x, "... (m n) -> ... (n m)", n=self.sqrt_n)
        x = blockdiag_matmul(x, self.R)

class MonarchMixerLayer(nn.Module):
    def __init__(self, sqrt_n: int, sqrt_d: int):
        super().__init__()
        self.m1 = MonarchMatrix(sqrt_n) 
        self.m2 = MonarchMatrix(sqrt_n) 
        self.m3 = MonarchMatrix(sqrt_d) 
        self.m4 = MonarchMatrix(sqrt_d)

        self.n_kernel = nn.Parameter(torch.randn(sqrt_d ** 2, sqrt_n ** 2))
        self.d_kernel = nn.Parameter(torch.randn(1, sqrt_d ** 2))
        self.layer_norm = nn.LayerNorm(sqrt_d ** 2)

    def forward(self, x: torch.Tensor): # x.shape = (b, n, d)
        x_tilde = self.m2(torch.relu(self.n_kernel * self.m1(x.transpose(-1, -2)))).transpose(-1, -2) # mix sequence
        y = self.m4(torch.relu(self.d_kernel * self.m3(x_tilde))) # mix features
        return self.layer_norm(y + x_tilde) # skip connection


class ConvMonarchMixerLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding='same'):
        super().__init__()
        
        assert padding=='same' #Only implement same padding for now
        assert isinstance(kernel_size, int) #Only support square kernel
        assert input_channels == out_channels

        # Normal Conv2D weights (out channels, in_channels/groups, kernel_size[0], kernel_size[1])
        # Monarch Conv2D weights (out channels, in_channels/groups, nblocks, blocks_size[0], blocks_size[1])
        
        # self.m1 = MonarchMatrix(sqrt_n) 
        # self.m2 = MonarchMatrix(sqrt_n)

        # self.m3 = MonarchMatrix(sqrt_d) 
        # self.m4 = MonarchMatrix(sqrt_d)

        # self.n_kernel = nn.Parameter(torch.randn(sqrt_d ** 2, sqrt_n ** 2))
        # self.d_kernel = nn.Parameter(torch.randn(1, sqrt_d ** 2))
        # self.layer_norm = nn.BatchNorm2d(in_channels)

    def forward(self, x: torch.Tensor): # x.shape = (b, n, d)

        x0 = x
        x = self.depth_wise(x)
        x = nn.GELU(x)
        x = self.layer_norm(x)

        x = x0 + x

        x = self.point_wise(x)
        x = nn.GELU(x)
        x = self.layer_norm(x)

        return x

        x_tilde = self.m2(nn.GELU(self.n_kernel * self.m1(x.transpose(-1, -2)))).transpose(-1, -2) # Depthwise conv

        y = self.m4(torch.relu(self.d_kernel * self.m3(x_tilde))) # Pointwise Conv

        return self.layer_norm(y + x_tilde) # skip connection