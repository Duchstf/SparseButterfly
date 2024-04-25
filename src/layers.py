import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, init
from torch import Tensor
from torch.nn.parameter import Parameter

import numpy as np
import math

from  ops import StructuredLinear, blockdiag_butterfly_multiply
#test
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

class ButterflyGlobalLinear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, window_size: int = 6,
                 stripes: int = 3, step=1, gtoken=1, block_size=None, device=None, dtype=None) -> None:
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
        tmp_mask[:, :gtoken] = 1.
        tmp_mask[:gtoken, :] = 1.

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
              
class MonarchLinear(StructuredLinear):

    def __init__(self, *args, nblocks=4, **kwargs):
        super().__init__(*args, **kwargs)
        in_blksz = int(math.ceil(self.in_features / nblocks))
        out_blksz = int(math.ceil(self.out_features / nblocks))
        self.in_features_extended = in_blksz * nblocks
        self.out_features_extended = out_blksz * nblocks

        if self.in_features_extended < self.out_features_extended:
            self.blkdiag1 = nn.Parameter(torch.empty(nblocks, in_blksz, in_blksz))
            self.blkdiag2 = nn.Parameter(torch.empty(nblocks, out_blksz, in_blksz))
        else:
            self.blkdiag1 = nn.Parameter(torch.empty(nblocks, out_blksz, in_blksz))
            self.blkdiag2 = nn.Parameter(torch.empty(nblocks, out_blksz, out_blksz))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Mimic init.kaiming_uniform: https://github.com/pytorch/pytorch/blob/24087d07ca7ffa244575d259711dd7c99245a67a/torch/nn/init.py#L360
        for blkdiag in [self.blkdiag1, self.blkdiag2]:
            fan_in = blkdiag.shape[-1]
            gain = init.calculate_gain(nonlinearity='leaky_relu', param=math.sqrt(5))
            std = gain / math.sqrt(fan_in)
            bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            with torch.no_grad():
                blkdiag.uniform_(-bound, bound)
        self.reset_parameters_bias()

    @property
    def saving(self):
        return ((self.blkdiag1.numel() + self.blkdiag2.numel())
                / (self.in_features * self.out_features))

    def forward_matmul(self, x):
        output = blockdiag_butterfly_multiply(self.preprocess(x), self.blkdiag1, self.blkdiag2)
        return self.postprocess(output)
    
#Implementation from this paper: https://arxiv.org/pdf/2310.12109.pdf
from einops import rearrange
from torch import nn

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
        return rearrange(x, "... (m n) -> ... (n m)", n=self.sqrt_n)
    
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
        
def blockdiag_multiply_reference(x, weight):
    """
    This implementation is slow but more likely to be correct.
    Arguments:
        x: (..., n)
        weight: (nblocks, q, n / nblocks)
    Outputs:
        out: (..., nblocks * q)
    """
    n = x.shape[-1]
    nblocks, q, p = weight.shape
    assert nblocks * p == n

    x_reshaped = rearrange(x, '... (nblocks p) -> ... nblocks p', nblocks=nblocks)
    return rearrange(torch.einsum('...kp, kqp -> ...kq', x_reshaped, weight),
                     '... nblocks q -> ... (nblocks q)')
    
class BlockdiagMultiply(torch.autograd.Function):

    """This is a faster implementation, with careful memory copies for the fastest
    bmm performance.
    The backward pass is also written manually with careful memory copies.
    Arguments:
        x: (..., n)
        weight: (nblocks, q, n / nblocks)
    Outputs:
        out: (..., nblocks * q)
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.bfloat16)
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        nblocks, q, p = weight.shape
        assert nblocks * p == n
        x_reshaped = x.reshape(batch_dim, nblocks, p).transpose(0, 1)
        out = torch.empty(batch_dim, nblocks, q, device=x.device, dtype=x.dtype).transpose(0, 1)
        out = torch.bmm(x_reshaped, weight.transpose(-1, -2), out=out).transpose(0, 1)
        return out.reshape(*batch_shape, nblocks * q)

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout):
        x, weight = ctx.saved_tensors
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        nblocks, q, p = weight.shape
        assert nblocks * p == n
        dx, dweight = None, None
        dout_reshaped = dout.reshape(batch_dim, nblocks, q).transpose(0, 1)
        if ctx.needs_input_grad[0]:
            dx = torch.empty(batch_dim, nblocks, p, device=x.device, dtype=x.dtype)
            dx = torch.bmm(dout_reshaped, weight.conj(),
                           out=dx.transpose(0, 1)).transpose(0, 1).reshape(*batch_shape, n)
        if ctx.needs_input_grad[1]:
            x_reshaped = x.reshape(batch_dim, nblocks, p).transpose(0, 1)
            dweight = torch.bmm(dout_reshaped.transpose(-1, -2), x_reshaped.conj())
        return dx, dweight
    
blockdiag_multiply = BlockdiagMultiply.apply
        
class BlockdiagLinear(StructuredLinear):

    def __init__(self, *args, nblocks=10, shuffle=False, **kwargs):
        """shuffle: apply channel_shuffle operation before the matmul as in ShuffleNet
        """
        super().__init__(*args, **kwargs)
        nblocks = 10
        in_blksz = int(math.ceil(self.in_features / nblocks))
        out_blksz = int(math.ceil(self.out_features / nblocks))
        self.in_features_extended = in_blksz * nblocks
        self.out_features_extended = out_blksz * nblocks
        self.shuffle = shuffle
        self.weight = nn.Parameter(torch.empty(nblocks, out_blksz, in_blksz))
        self.reset_parameters()

    def set_weights_from_dense_init(self, dense_init_fn_):
        dense_weight = torch.empty(self.out_features_extended, self.in_features_extended,
                                   device=self.weight.device, dtype=self.weight.dtype)
        dense_init_fn_(dense_weight)
        # Scale by sqrt because the weight is sparse
        scaling = math.sqrt(dense_weight.numel() / self.weight.numel())
        dense_weight *= scaling
        with torch.no_grad():
            nblocks = self.weight.shape[0]
            self.weight.copy_(rearrange(dense_weight, '(b o) (b1 i) -> b b1 o i',
                                        b=nblocks, b1=nblocks)[0])

    @property
    def saving(self):
        return self.weight.numel() / (self.in_features * self.out_features)

    def forward_matmul(self, x):
        x = self.preprocess(x)
        if self.shuffle:
            x = rearrange(x, '... (group c_per_group) -> ... (c_per_group group)',
                          group=self.weight.shape[0])  # group=nblocks
        output = blockdiag_multiply(x, self.weight)
        return self.postprocess(output)
