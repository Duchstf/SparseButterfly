import torch.nn as nn
import torch.nn.functional as F
import torch
from layers import *
from functools import partial
#from mlp_mixer_pytorch import MLPMixer


class MNIST_MLP_Vanilla(nn.Module):
    """
    Mlp with butterfly matrices:
    
    https://arxiv.org/pdf/2112.00029.pdf
    """
    
    def __init__(self, in_features, out_features=10):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 392, bias=True)
        self.fc2 = nn.Linear(392, 128, bias=True)
        self.fc3 = nn.Linear(128, 64, bias=True)
        self.fc4 = nn.Linear(64, out_features, bias=True)
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        output = F.log_softmax(x, dim=1)
        return output
class MNIST_Monarch_MLP(nn.Module):
    """ 
    MLP with Monarch matrices: https://arxiv.org/pdf/2204.00595.pdf
    Monarch Mixer: https://arxiv.org/pdf/2310.12109.pdf
    """
    def __init__(self, in_features, out_features=10):
        super().__init__()
        self.fc1 = MonarchLinear(in_features, 392, nblocks=15)
        self.fc2 = MonarchLinear(392, 128,  nblocks=15, bias=True)
        self.fc3 = MonarchLinear(128, 64,  nblocks=15, bias=True)
        self.fc4 = MonarchLinear(64, out_features, nblocks=15, bias=True)
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = F.gelu(x)
        x = self.fc3(x)
        x = F.gelu(x)
        x = self.fc4(x)
        output = F.log_softmax(x, dim=1)
        return output
        return output
    
        return output  
    
class CIFAR10_MLP_Vanilla(nn.Module):
    """
    Mlp with butterfly matrices:
    
    https://arxiv.org/pdf/2112.00029.pdf
    """
    
    def __init__(self, in_features, out_features=10):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 3000, bias=True)
        self.fc2 = nn.Linear(3000, 2000, bias=True)
        self.fc3 = nn.Linear(2000, 1000, bias=True)
        self.fc4 = nn.Linear(1000, 500, bias=True)
        self.fc5 = nn.Linear(500, 200, bias=True)
        self.fc6 = nn.Linear(200, 100, bias=False)
        self.fc7 = nn.Linear(100, out_features, bias=True)
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        x = self.fc6(x)
        x = F.relu(x)
        x = self.fc7(x)
        output = F.log_softmax(x, dim=1)
        return output
    
class CIFAR10_Monarch_MLP(nn.Module):
    """ 
    MLP with Monarch matrices: https://arxiv.org/pdf/2204.00595.pdf
    Monarch Mixer: https://arxiv.org/pdf/2310.12109.pdf
    """
    def __init__(self, in_features, nblocks=4, out_features=10):
        super().__init__()
        self.fc1 = MonarchLinear(in_features, 3000, nblocks=nblocks, bias=False)
        print(self.fc1.saving)
        self.fc2 = MonarchLinear(3000, 2000,  nblocks=nblocks, bias=False)
        self.fc3 = MonarchLinear(2000, 1000,  nblocks=nblocks, bias=False)
        self.fc4 = MonarchLinear(1000, 500,  nblocks=nblocks, bias=False)
        self.fc5 = MonarchLinear(500, 200,  nblocks=nblocks, bias=False)
        self.fc6 = MonarchLinear(200, 100,  nblocks=nblocks, bias=False)
        self.fc7 = MonarchLinear(100, out_features, nblocks=nblocks, bias=False)
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        x = self.fc6(x)
        x = F.relu(x)
        x = self.fc7(x)
        output = F.log_softmax(x, dim=1)
        return output
class CIFAR10_Butterfly_MLP(nn.Module):
    """ 
    MLP with Monarch matrices: https://arxiv.org/pdf/2204.00595.pdf
    Monarch Mixer: https://arxiv.org/pdf/2310.12109.pdf
    """
    def __init__(self, in_features, stripes=5, window_size=6, block_size=(30,30), out_features=10):
        super().__init__()
        self.fc1 = ButterflyGlobalLinear(in_features, 392, stripes=5, window_size=6, block_size=(30,30), bias=True)
        self.fc2 = ButterflyGlobalLinear(392, 128, stripes=4, window_size=4, block_size=(15,15), bias=True)
        self.fc3 = ButterflyGlobalLinear(128, 64, stripes=4, window_size=4, block_size=(15,15), bias=True)
        self.fc4 = ButterflyGlobalLinear(64, out_features, stripes=2, window_size=3, block_size=None, bias=True)
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        output = F.log_softmax(x, dim=1)
        return output
      
class CIFAR10_Monarch_MLP2(nn.Module):
    """ 
    MLP with Monarch matrices: https://arxiv.org/pdf/2204.00595.pdf
    Monarch Mixer: https://arxiv.org/pdf/2310.12109.pdf
    """
    def __init__(self, in_features, nblocks=1, out_features=10):
        super().__init__()
        self.fc1 = MonarchLinear(in_features, 3000, nblocks=nblocks, bias=True)
        self.fc2 = MonarchLinear(3000, 1000,  nblocks=nblocks, bias=True)
        self.fc3 = MonarchLinear(1000, 100,  nblocks=nblocks, bias=True)
        self.fc4 = MonarchLinear(100, out_features, nblocks=nblocks, bias=True)
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        output = F.log_softmax(x, dim=1)
        return output


from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce

pair = lambda x: x if isinstance(x, tuple) else (x, x)

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = BlockdiagLinear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )


def MLPMixerbob(*, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_h // patch_size) * (image_w // patch_size)
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), BlockdiagLinear

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        nn.Linear((patch_size ** 2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes)
    )

class CIFAR10_MLP_Mixer(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.mixer = MLPMixerbob(
            image_size = (32, 32),
            channels = 3,
            patch_size = 8,
            dim = 512,
            depth = 12,
            num_classes = 10
        )

    def forward(self, x):
        output = F.log_softmax(self.mixer(x))
        return output



