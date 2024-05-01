import torch.nn as nn
import torch.nn.functional as F
import torch
from functools import partial

class VanillaMlp(nn.Module):
    """
    Mlp with butterfly matrices:
    
    https://arxiv.org/pdf/2112.00029.pdf
    """
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 3000, bias=True)
        self.fc2 = nn.Linear(3000, 1000, bias=True)
        self.fc3 = nn.Linear(1000, 1000, bias=True)
        self.fc4 = nn.Linear(1000, 100, bias=True)
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