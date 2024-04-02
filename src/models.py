import torch.nn as nn
import torch.nn.functional as F
import torch
from layers import ButterflyLinear

class ButterflyMlp(nn.Module):
    """ newMlp, similar to Mlp layers in MLP-Mixer but with extra gelu act and low-rank
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., window_size=3, step=1, stripes_1=3, stripes_2=1, block_size=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ButterflyLinear(784, 784, bias=True, window_size=window_size, stripes=stripes_1, step=step, block_size=block_size)
        self.fc2 = ButterflyLinear(784, 128, bias=True, window_size=window_size, stripes=stripes_1, step=step, block_size=block_size)
        self.fc3 = ButterflyLinear(128, 10, bias=True, window_size=window_size, stripes=stripes_2, step=step, block_size=block_size)
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