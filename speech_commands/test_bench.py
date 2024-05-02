from models import *
import torchsummary
from einops.layers.torch import Rearrange, Reduce

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )

def Mixer(input_length, num_classes, patch_size=100, dim=128, depth=1, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.):

    num_patches = (input_length // patch_size)
    assert (input_length % patch_size) == 0, 'input length must be divisible by patch size'
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    return nn.Sequential(
        Rearrange('b c (l p)  -> b l (p c)', p = patch_size),
        nn.Linear(patch_size, dim),

        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
        ) for _ in range(depth)],

        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes),
        nn.Softmax(dim=1)
    )

device = 'cuda' if torch.cuda.is_available() else 'cpu'
inputs = torch.rand(256,1,8000)
print(inputs.shape)
#Test the NN
net = Mixer(8000, 35)
net=net.to(device)

x = net(inputs)
print(x.shape)
print(torchsummary.summary(net, (1,8000)))


#
# inputs = 