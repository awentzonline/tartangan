import functools

import torch
from torch import nn


class PPNInput(nn.Module):
    def __init__(self, latent_dims, out_dims, position_dims=2,
                 activation_factory=functools.partial(nn.LeakyReLU, 0.2)):
        super().__init__()
        self.xform = nn.Sequential(
            nn.Linear(latent_dims + position_dims, out_dims),
            activation_factory()
        )

    def forward(self, z, position):
        x = torch.cat([z, position], dim=-1)
        return self.xform(x)


class TransformBlock(nn.Module):
    def __init__(self, in_dims, out_dims,
                 activation_factory=functools.partial(nn.LeakyReLU, 0.2)):
        super().__init__()
        self.xform = nn.Sequential(
            nn.Linear(in_dims, out_dims),
            activation_factory()
        )

    def forward(self, x):
        return self.xform(x)
