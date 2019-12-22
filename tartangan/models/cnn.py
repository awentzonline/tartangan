import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class GeneratorCNN(nn.Module):
    def __init__(self, latent_dims, img_size, base_dims=16, depth=3):
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
                GeneratorBlock(dims, last_dims)
            )
            last_dims = dims
        layers = list(reversed(layers))
        layers += [
            nn.Conv2d(
                base_dims, self.img_channels, 1, padding=0
            ),
            nn.Sigmoid(),
        ]
        self.generator = nn.Sequential(*layers)

        # generate a base image from the latent code
        self.base_dims = 2 ** depth * base_dims
        self.base_size = base_size
        self.base_img_dims = self.base_size ** 2 * self.base_dims
        self.base_image = nn.Sequential(
            nn.Linear(latent_dims, self.base_img_dims, bias=False),
            nn.ReLU(),
        )

    def forward(self, z):
        base_img = self.base_image(z)
        base_img = base_img.view(-1, self.base_dims, self.base_size, self.base_size)
        img = self.generator(base_img)
        return img


class GeneratorBlock(nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_dims, out_dims, 3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(out_dims),
        )

    def forward(self, x):
        return self.conv(x)


class ResidualAttentionModule(nn.Module):
    def __init__(self, in_dims):
        super().__init__()
        self.x_form = nn.Sequential(
            nn.Linear(in_dims, in_dims)
        )
        self.attention = nn.Sequential(
            nn.Linear(in_dims, in_dims),
            nn.Sigmoid(),
        )

    def forward(self, x):
        att = self.attention(x)
        att_x = att * x
        return x + self.x_form(att_x)


class DiscriminatorCNN(nn.Module):
    def __init__(self, img_size, base_dims=32, depth=3):
        super().__init__()
        layers = []
        layer_dims = 2 ** np.arange(depth) * base_dims
        layer_dims = (3,) + tuple(layer_dims)
        for in_dims, out_dims in zip(layer_dims[:-1], layer_dims[1:]):
            layers += [
                nn.Conv2d(in_dims, out_dims, 3, 2),
                nn.BatchNorm2d(out_dims),
                nn.ReLU(),
            ]
        layers += [
            nn.Conv2d(out_dims, 1, 1),
            nn.Sigmoid(),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, img):
        feats = self.conv(img)
        return F.avg_pool2d(feats, feats.size()[2:]).view(-1, 1)
