from models import *
import torchsummary
from einops.layers.torch import Rearrange, Reduce

device = 'cpu'
inputs = torch.rand(256,1,8000).to(device)
print(inputs.shape)

#Test the NN
net = Mixer(8000, 35)
net=net.to(device)

x = net(inputs)
print(x.shape)
# print(torchsummary.summary(net, (1,8000.to(device))))


#
# inputs = 