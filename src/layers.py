import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, init
from torch import Tensor
from torch.nn.parameter import Parameter
import math

def generate_mask(base, block_size=None):
    if block_size is not None:
        num_r, num_c = base.shape
        b_r, b_c = block_size
        mask = torch.zeros(base.shape)
        for i in range(0, num_r, b_r):
            for j in range(0, num_c, b_c):
                lighten = torch.sum(base[i:(i+b_r), j:(j+b_c)])
                if lighten > 0.0:
                    mask[i:(i+b_r), j:(j+b_c)] = 1.
        return mask
    else:
        return base

class ButterflyLinear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, window_size: int = 6, stripes: int = 3, step = 1,
                 block_size=None, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        pseudo_mask_size = (min(in_features, out_features), max(in_features, out_features))
        tmp_mask = torch.zeros(pseudo_mask_size)
        stride = int(math.sqrt(pseudo_mask_size[0]))
        d = math.ceil(pseudo_mask_size[1] / pseudo_mask_size[0])

        for k in range(stripes):
            patch_start = stride * k
            for i in range(0, pseudo_mask_size[0], window_size):
                tmp_mask[patch_start + i:patch_start + i + window_size, i * d: i * d + step * d * window_size] = 1.

        for k in range(stripes):
            patch_start = stride * k
            for i in range(0, pseudo_mask_size[0], window_size):
                tmp_mask[i: i + window_size, (i + patch_start) * d: (patch_start + i) * d + step * d * window_size] = 1.
        tmp_mask = generate_mask(tmp_mask, block_size)

        if in_features <= out_features:
            self.register_buffer('sparse_mask', tmp_mask.t())
        else:
            self.register_buffer('sparse_mask', tmp_mask)
            
        self.saving = torch.sum(generate_mask(tmp_mask))/(self.in_features*self.out_features)

        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.data = self.weight.data/math.sqrt(torch.sum(self.sparse_mask)/(self.in_features*self.out_features))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return input @ ((self.sparse_mask*self.weight).t()) + self.bias

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )