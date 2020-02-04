import torch
from torch import nn
import torch.nn.functional as F


class Interpolate(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return F.interpolate(x, *self.args, **self.kwargs)


class PixelNorm(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x / torch.sqrt((x ** 2).mean(1, keepdim=True) + self.eps)
