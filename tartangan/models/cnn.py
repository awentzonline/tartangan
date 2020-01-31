import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .blocks import (
    ResidualDiscriminatorBlock, ResidualGeneratorBlock, TiledZGeneratorInput
)


class GeneratorCNN(nn.Module):
    def __init__(
        self, latent_dims, img_size, base_dims=16, depth=3,
        input_factory=TiledZGeneratorInput,
        block_factory=ResidualGeneratorBlock,
    ):
        super().__init__()
        self.img_size = img_size
        self.img_channels = 3
        last_dims = base_dims
        layers = []
        base_size = img_size
        for i in range(depth):
            dims = 2 ** (i + 1) * base_dims
            base_size //= 2
            layers.append(
                block_factory(dims, last_dims, upsample=True)
            )
            last_dims = dims
        layers = list(reversed(layers))
        layers += [
            nn.Conv2d(
                base_dims, self.img_channels, 1, padding=0
            ),
            nn.Tanh(), #nn.Sigmoid(),
        ]
        self.generator = nn.Sequential(*layers)
        self.input_block = input_factory(latent_dims, size=base_size)

    def forward(self, z):
        base_img = self.input_block(z)
        img = self.generator(base_img)
        return img


class DiscriminatorCNN(nn.Module):
    def __init__(
        self, img_size, base_dims=32, depth=3,
        block_factory=ResidualDiscriminatorBlock
    ):
        super().__init__()
        layers = []
        layer_dims = 2 ** np.arange(depth) * base_dims
        layer_dims = (3,) + tuple(layer_dims)
        for in_dims, out_dims in zip(layer_dims[:-1], layer_dims[1:]):
            layers += [
                block_factory(in_dims, out_dims)
            ]
        layers += [
            nn.Conv2d(out_dims, 1, 1),
            nn.Sigmoid(),  # nn.Tanh()  #
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, img):
        feats = self.conv(img)
        return F.avg_pool2d(feats, feats.size()[2:]).view(-1, 1)
