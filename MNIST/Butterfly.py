import math
import numpy as np
import torchvision.datasets as datasets

import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torch
from torch.nn import functional as F
import torch.nn as nn
from einops import rearrange


mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
print(mnist_trainset)

def blockdiag_butterfly_multiply_reference(x, w1_bfly, w2_bfly, version=2):
    """
    This implementation is slow but more likely to be correct.
    There are 3 implementations, which should all yield the same answer
    Arguments:
        x: (batch, n)
        w1_bfly: (k, q, p), where k = n / p
        w2_bfly: (l, s, r), where l = k * q / r = n * q / (p * r)
    Outputs:
        out: (batch, m), where m = l * s = n * s * q / (p * r)
    """
    if version not in [1, 2, 3]:
        raise NotImplementedError('version must be either 1, 2, or 3')
    batch, n = x.shape
    k, q, p = w1_bfly.shape
    l, s, r = w2_bfly.shape
    assert k * p == n
    assert l * r == k * q

    x_reshaped = rearrange(x, 'b (k p) -> b k p', k=k)
    if version == 1:  # Implementation 1 (only works for when k = q = p = l = s = r = sqrt(n))
        assert k == q == p == l == s == r == int(math.sqrt(n))
        return torch.einsum('bkp,kqp,qlk->blq', x_reshaped, w1_bfly, w2_bfly).reshape(batch, n)
    elif version == 2:  # Implementation 2
        out1 = torch.einsum('kqp,bkp->bkq', w1_bfly, x_reshaped)
        out1 = rearrange(rearrange(out1, 'b k q -> b (k q)'), 'b (r l) -> b l r', l=l)
        return torch.einsum('lsr,blr->bsl', w2_bfly, out1).reshape(batch, s * l)
    # Implementation 3: most likely to be correct, but it's the slowest
    elif version == 3:
        w1_dense = torch.block_diag(*torch.unbind(w1_bfly, dim=0))
        out1 = F.linear(x, w1_dense)
        out1 = rearrange(out1, 'b (r l) -> b (l r)', l=l)
        w2_dense = torch.block_diag(*torch.unbind(w2_bfly, dim=0))
        out2 = F.linear(out1, w2_dense)
        out2 = rearrange(out2, 'b (l s) -> b (s l)', l=l)
        return out2


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
        #print(f'x.shape:{x.shape}')
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        k, q, p = w1_bfly.shape
        # print(f'k, q, p: {k, q, p}')
        # print(f'k*p:{k * p}')
        # print(f'n:{n}')
        l, s, r = w2_bfly.shape
        # print(f'l, s, r: {l, s, r}')
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
print(blockdiag_butterfly_multiply)
class ButterflyBlockDiagLayer(nn.Module):
    def __init__(self, n, q, p, s, r):
        super(ButterflyBlockDiagLayer, self).__init__()
        k = n // p
        l = n * q // (p * r)

        # Initialize weights
        self.w1_bfly = nn.Parameter(torch.randn(k, q, p))
        self.w2_bfly = nn.Parameter(torch.randn(l, s, r))

    def forward(self, x):
        return blockdiag_butterfly_multiply(x, self.w1_bfly, self.w2_bfly)
         

class Butterfly_DNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Butterfly_DNN, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = ButterflyBlockDiagLayer(input_size, q=4, p=16, s=8, r=4)
        self.layer2 = ButterflyBlockDiagLayer(hidden_size, q=4, p=4, s=4, r=4)
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.layer3(x)
        return x


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

class ButterflyMlp(nn.Module):

    """ newMlp, similar to Mlp layers in MLP-Mixer but with extra gelu act and low-rank
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., window_size=3, step=1, stripes_1=3, stripes_2=1, block_size=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ButterflyLinear(in_features, hidden_features, bias=True, window_size=window_size, stripes=stripes_1, step=step, block_size=block_size)
        self.act = act_layer()
        self.fc2 = ButterflyLinear(hidden_features, out_features, bias=True, window_size=window_size, stripes=stripes_2, step=step, block_size=block_size)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

###### Training on MNIST:
train_set = datasets.MNIST('./data', train=True, download=True)
test_set = datasets.MNIST('./data', train=False, download=True)

x_train, y_train = train_set.data/255., train_set.targets
x_test, y_test = test_set.data/255., test_set.targets
print(x_train.shape)
print(x_test.shape)


### Training/testing loop for normal DNN:

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_base_model(LinearLayer):
    model = nn.Sequential(
        nn.Flatten(),
        LinearLayer(28*28, 128),
        nn.ReLU(),
        LinearLayer(128, 128),
        nn.ReLU(),
        LinearLayer(128, 10),
    )
    return model


def train(model, epochs=2, batch_size=32, lr=0.001):
    num_iters = x_train.shape[0] // batch_size
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for iter in range(num_iters):
            i_start = iter * batch_size
            x_batch = x_train[i_start : i_start+batch_size].to(device)
            y_batch = y_train[i_start : i_start+batch_size].to(device)
            #print(x_batch.shape)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        y_pred = model(x_test.to(device))
        print(y_pred.shape)
        cls_pred = y_pred.argmax(dim=1, keepdim=True).cpu().numpy()
        accuracy = np.mean(cls_pred[:,0] == y_test.numpy())
        print(f"epoch: {epoch}  ||  loss: {loss:.4f}  ||  acc: {100*accuracy:.2f}%")

    return model

# base_model = create_base_model(LinearLayer=nn.Linear)
# base_model.to(device)
# base_model = train(base_model)


# torch.save(base_model.state_dict(), 'model_weights.pth')
# base_model.load_state_dict(torch.load('model_weights.pth'))
# import matplotlib.pyplot as plt

# layer_weights = base_model[3].weight.data.numpy()  # Convert to numpy array
# print(layer_weights)

# plt.imshow(layer_weights, cmap='gray')  # 'viridis' is a colormap, you can choose others like 'gray'
# plt.colorbar()
# plt.title("Visualization of Layer Weights")
# plt.show()
# plt.savefig("bob.png")

## Training/testing loop for butterfly matrices DNN:

butterfly_dnn = Butterfly_DNN(28*28, 392, 10)
butterfly_dnn.to(device)
butterfly_dnn = train(butterfly_dnn)
torch.save(butterfly_dnn.state_dict(), 'model_weights.pth')
butterfly_dnn.load_state_dict(torch.load('model_weights.pth'))
import matplotlib.pyplot as plt

layer_weights = butterfly_dnn._modules['layer1']  
print(layer_weights)
print(type(layer_weights))

w1_bfly = butterfly_dnn._modules['layer1'].w1_bfly.detach().cpu().numpy()
w2_bfly = butterfly_dnn._modules['layer1'].w2_bfly.detach().cpu().numpy()
print(w1_bfly)
print(w2_bfly)
print(w1_bfly.shape)
print(w2_bfly.shape)
plt.imshow(w2_bfly[0, :, :], cmap='gray') 
plt.colorbar()
plt.title("Visualization of Layer Weights")
plt.show()
plt.savefig("bob_hi.png")

# def visualize_matrix(matrix, title="Matrix Visualization"):
#     plt.imshow(matrix, cmap='viridis')
#     plt.colorbar()
#     plt.title(title)
#     plt.show()


# for i, block in enumerate(w1_bfly):
#     visualize_matrix(block, title=f"W1 Block {i}")

# for i, block in enumerate(w2_bfly):
#     visualize_matrix(block, title=f"W2 Block {i}")




