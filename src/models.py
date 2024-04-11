import torch.nn as nn
import torch.nn.functional as F
import torch
from layers import *

class ButterflyMlp(nn.Module):
    """
    Mlp with butterfly matrices:
    
    https://arxiv.org/pdf/2112.00029.pdf
    """
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.,
                 window_size=10, step=3, stripes=5, block_size=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ButterflyLinear(784, 784, bias=True, window_size=window_size, stripes=stripes, step=step, block_size=block_size)
        self.fc2 = ButterflyLinear(784, 128, bias=True, window_size=window_size, stripes=stripes, step=step, block_size=block_size)
        self.fc3 = ButterflyLinear(128, 10, bias=True, window_size=window_size, stripes=stripes, step=step, block_size=block_size)
        self.act = act_layer()
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output
    
class MonarchMlp(nn.Module):
    """ 
    Mlp with Monarch matrices:
    
    https://arxiv.org/pdf/2204.00595.pdf
    """
    def __init__(self, in_features, out_features=None):
        super().__init__()
        self.fc1 = MonarchLinear(in_features, 784, bias=True)
        self.fc2 = MonarchLinear(784, 128, bias=True)
        self.fc3 = MonarchLinear(128, 10, bias=True)
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output
    
class MonarchMLP2(nn.Module):
    """ 
    MLP with Monarch matrices: https://arxiv.org/pdf/2204.00595.pdf
    Monarch Mixer: https://arxiv.org/pdf/2310.12109.pdf
    """
    def __init__(self, in_features, out_features=10):
        super().__init__()
        self.fc1 = BlockdiagLinear(in_features, 784, nblocks=20)
        self.fc2 = BlockdiagLinear(784, 128,  nblocks=20, bias=True)
        self.fc3 = BlockdiagLinear(128, out_features, nblocks=20, bias=True)
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = F.gelu(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output
    
