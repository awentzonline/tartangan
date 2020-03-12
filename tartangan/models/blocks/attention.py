import torch
from torch import nn
import torch.nn.functional as F


class SelfAttention2d(nn.Module):
  """
  Adapted from https://github.com/ajbrock/BigGAN-PyTorch/blob/master/layers.py
  """
  def __init__(self, in_dims, attention_dims=None):
    super().__init__()
    # Channel multiplier
    self.in_dims = in_dims
    self.theta = nn.Conv2d(in_dims, in_dims // 8, 1, bias=False)
    self.phi = nn.Conv2d(in_dims, in_dims // 8, 1, bias=False)
    self.g = nn.Conv2d(in_dims, in_dims // 2, 1, bias=False)
    self.o = nn.Conv2d(in_dims // 2, in_dims, 1, bias=False)
    # Learnable gain parameter
    self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)

  def forward(self, x, y=None):
    batch, channels, height, width = x.shape
    # Apply convs
    theta = self.theta(x)
    phi = F.max_pool2d(self.phi(x), [2, 2])
    g = F.max_pool2d(self.g(x), [2, 2])
    # Perform reshapes
    theta = theta.view(-1, self.in_dims // 8, height * width)
    phi = phi.view(-1, self.in_dims // 8, height * width // 4)
    g = g.view(-1, self.in_dims // 2, height * width // 4)
    # Matmul and softmax to get attention maps
    beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
    # Attention map times g path
    o = self.o(torch.bmm(g, beta.transpose(1,2)).view(-1, self.in_dims // 2, height, width))
    return self.gamma * o + x
