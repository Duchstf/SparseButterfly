import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, init
from torch import Tensor
from torch.nn.parameter import Parameter

import math
import numpy as np
from einops import rearrange

class MonarchMlp(nn.Module):
    """
    Mlp with butterfly matrices:
    
    https://arxiv.org/pdf/2112.00029.pdf
    """
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = MonarchLinear(in_features, 3000, bias=True)
        self.fc2 = MonarchLinear(3000, 1000, bias=True)
        self.fc3 = MonarchLinear(1000, 1000, bias=True)
        self.fc4 = MonarchLinear(1000, 100, bias=True)
        self.fc5 = nn.Linear(100, out_features, bias=True)
        
    def forward(self, x):
        print(x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        return x

class BlockdiagButterflyMultiply(torch.autograd.Function):

    """This is a faster implementation, with careful memory copies for the fastest
    bmm performance.
    The backward pass is also written manually with careful memory copies.
    Arguments:
        x: (batch, n)
        w1_bfly: (k, q, p), where k = n / p
        w2_bfly: (l, s, r), where l = k * q / r = n * q / (p * r)
    Outputs:
        out: (batch, m), where m = l * s = n * s * q / (p * r)
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, x, w1_bfly, w2_bfly):
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        k, q, p = w1_bfly.shape
        l, s, r = w2_bfly.shape
        assert k * p == n
        assert l * r == k * q
        x_reshaped = x.reshape(batch_dim, k, p).transpose(0, 1)
        out1 = torch.empty(batch_dim, k, q, device=x.device, dtype=x.dtype).transpose(0, 1)
        out1 = torch.bmm(x_reshaped, w1_bfly.transpose(-1, -2), out=out1)
        out1 = out1.transpose(0, 1).reshape(batch_dim, r, l).transpose(-1, -2).contiguous().transpose(0, 1)
        out2 = torch.empty(batch_dim, l, s, device=x.device, dtype=x.dtype).transpose(0, 1)
        out2 = torch.bmm(out1, w2_bfly.transpose(-1, -2), out=out2)
        out2 = out2.permute(1, 2, 0).reshape(*batch_shape, s * l)
        ctx.save_for_backward(x, w1_bfly, w2_bfly, out1)
        return out2

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout):
        x, w1_bfly, w2_bfly, out1 = ctx.saved_tensors
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        k, q, p = w1_bfly.shape
        l, s, r = w2_bfly.shape
        # assert k * p == n
        # assert l * r == k * q
        dx, dw1_bfly, dw2_bfly = None, None, None
        # dout_reshaped = dout.reshape(batch_dim, sqrtn, sqrtn).permute(2, 1, 0).contiguous()
        dout_reshaped = dout.reshape(batch_dim, s, l).transpose(-1, -2).contiguous()
        dout_reshaped = dout_reshaped.transpose(0, 1)
        if ctx.needs_input_grad[2]:
            # dw2_bfly = torch.empty(l, s, r, device=w2_bfly.device, dtype=w2_bfly.dtype)
            # dw2_bfly = torch.bmm(dout_reshaped.transpose(-1, -2), out1, out=dw2_bfly)
            dw2_bfly = torch.bmm(dout_reshaped.transpose(-1, -2), out1.conj())
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[0]:
            dout1 = torch.empty(batch_dim, l, r, device=x.device, dtype=x.dtype).transpose(0, 1)
            dout1 = torch.bmm(dout_reshaped, w2_bfly.conj(), out=dout1)
            dout1 = dout1.transpose(0, 1).transpose(-1, -2).contiguous().reshape(batch_dim, k, q).transpose(0, 1)
            # dout1 = dout1.permute(1, 2, 0).contiguous().transpose(0, 1)
            if ctx.needs_input_grad[0]:
                dx = torch.empty(batch_dim, k, p, device=x.device, dtype=x.dtype)
                dx = torch.bmm(dout1, w1_bfly.conj(), out=dx.transpose(0, 1)).transpose(0, 1).reshape(*batch_shape, n)
            if ctx.needs_input_grad[1]:
                x_reshaped = x.reshape(batch_dim, k, p).transpose(0, 1)
                dw1_bfly = torch.bmm(dout1.transpose(-1, -2), x_reshaped.conj())
        return dx, dw1_bfly, dw2_bfly
blockdiag_butterfly_multiply = BlockdiagButterflyMultiply.apply

class StructuredLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        """Subclasses should call reset_parameters
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Subclasses may override {in,out}_features_extended
        if not hasattr(self, 'in_features_extended'):
            self.in_features_extended = in_features
        if not hasattr(self, 'out_features_extended'):
            self.out_features_extended = out_features
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def reset_parameters(self) -> None:
        self.set_weights_from_dense_init(dense_init_fn_=partial(init.kaiming_uniform_, a=math.sqrt(5)))
        self.reset_parameters_bias()

    def set_weights_from_dense_init(self, dense_init_fn_):
        raise NotImplementedError

    def reset_parameters_bias(self):
        if self.bias is not None:
            fan_in = self.bias.shape[-1]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    @property
    def saving(self):
        raise NotImplementedError

    def convert_to_dense_weight(self):
        factory_kwargs = {'device': self.weight.device, 'dtype': self.weight.dtype}
        dense_weight = self.forward_matmul(torch.eye(self.in_features, **factory_kwargs)).T
        return dense_weight

    def preprocess(self, x):
        in_features = x.shape[-1]
        if in_features < self.in_features_extended:
            x = F.pad(x, (0, self.in_features_extended - in_features))
        return x

    def postprocess(self, output):
        out_features_extended = output.shape[-1]
        if out_features_extended > self.out_features:
            output = output[..., :self.out_features]
        return output

    def forward_matmul(self, x):
        raise NotImplementedError

    def forward(self, x):
        output = self.forward_matmul(x)
        # Convert bias to output.dtype in case of AMP, otherwise bias and activation will be in FP32
        return (output + self.bias.to(dtype=output.dtype)) if self.bias is not None else output

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