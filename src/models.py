import torch.nn as nn
import torch.nn.functional as F
import torch
from layers import *

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
        self.fc1 = MonarchLinear(in_features, 392, nblocks=10)
        self.fc2 = MonarchLinear(392, 128,  nblocks=10, bias=True)
        self.fc3 = MonarchLinear(128, 64,  nblocks=10, bias=True)
        self.fc4 = MonarchLinear(64, out_features, nblocks=10, bias=True)
        
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
    
class CIFAR10_MLP_Vanilla(nn.Module):
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
    
class CIFAR10_Monarch_MLP(nn.Module):
    """ 
    MLP with Monarch matrices: https://arxiv.org/pdf/2204.00595.pdf
    Monarch Mixer: https://arxiv.org/pdf/2310.12109.pdf
    """
    def __init__(self, in_features, out_features=10):
        super().__init__()
        self.fc1 = MonarchLinear(in_features, 392, nblocks=10)
        self.fc2 = MonarchLinear(392, 128,  nblocks=10, bias=True)
        self.fc3 = MonarchLinear(128, 64,  nblocks=10, bias=True)
        self.fc4 = MonarchLinear(64, out_features, nblocks=10, bias=True)
        
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