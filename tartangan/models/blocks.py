import torch
from torch import nn
import torch.nn.functional as F


class GeneratorBlock(nn.Module):
    def __init__(self, in_dims, out_dims, upsample=True):
        super().__init__()
        layers = [
            nn.utils.spectral_norm(
                nn.Conv2d(in_dims, out_dims, 3, padding=1, bias=False)),
            nn.BatchNorm2d(out_dims),
            nn.ReLU(True),
            nn.utils.spectral_norm(
                nn.Conv2d(out_dims, out_dims, 3, padding=1, bias=False)),
            nn.BatchNorm2d(out_dims),
            nn.ReLU(True),
        ]
        if upsample:
            layers.insert(0, Interpolate(scale_factor=2, mode='bilinear', align_corners=True))
        self.convs = nn.Sequential(*layers)
        map(nn.init.orthogonal_, self.parameters())

    def forward(self, x):
        return self.convs(x)


class GeneratorBlock(nn.Module):
    def __init__(self, in_dims, out_dims, upsample=True):
        super().__init__()
        layers = [
            nn.utils.spectral_norm(
                nn.Conv2d(in_dims, out_dims, 3, padding=1, bias=False)),
            nn.BatchNorm2d(out_dims),
            nn.ReLU(True),
            nn.utils.spectral_norm(
                nn.Conv2d(out_dims, out_dims, 3, padding=1, bias=False)),
            nn.BatchNorm2d(out_dims),
            nn.ReLU(True),
        ]
        if upsample:
            layers.insert(0, Interpolate(scale_factor=2, mode='bilinear', align_corners=True))
        self.convs = nn.Sequential(*layers)
        map(nn.init.orthogonal_, self.parameters())

    def forward(self, x):
        return self.convs(x)


class ResidualGeneratorBlock(nn.Module):
    def __init__(self, in_dims, out_dims, upsample=True):
        super().__init__()
        layers = [
            nn.utils.spectral_norm(
                nn.Conv2d(in_dims, out_dims, 3, padding=1, bias=False)),
            nn.BatchNorm2d(out_dims),
            nn.ReLU(True),
            nn.utils.spectral_norm(
                nn.Conv2d(out_dims, out_dims, 3, padding=1, bias=False)),
            nn.BatchNorm2d(out_dims),
            nn.ReLU(True),
        ]
        self.upsample = upsample
        if upsample:
            layers.insert(0, Interpolate(scale_factor=2, mode='bilinear', align_corners=True))
        self.project_input = None
        if in_dims != out_dims:
            self.project_input = nn.Sequential(
                nn.utils.spectral_norm(nn.Linear(in_dims, out_dims)),
            )
        self.convs = nn.Sequential(*layers)
        map(nn.init.orthogonal_, self.parameters())

    def forward(self, x):
        h = self.convs(x)
        if self.project_input:
            x = self.project_input(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2)
        return x + h


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_dims, out_dims, first_conv=None):
        super().__init__()
        layers = [
            nn.utils.spectral_norm(
                nn.Conv2d(in_dims, out_dims, 3, padding=1, bias=False)),
            nn.BatchNorm2d(out_dims),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(
                nn.Conv2d(out_dims, out_dims, 3, padding=1, bias=False)),
            nn.BatchNorm2d(out_dims),
            nn.LeakyReLU(0.2, inplace=True),
            Interpolate(scale_factor=0.5, mode='bilinear'),
        ]
        if first_conv:
            layers = [
                nn.utils.spectral_norm(
                    nn.Conv2d(first_conv, in_dims, 1, bias=False)),
                nn.ReLU()] + layers
        self.convs = nn.Sequential(*layers)
        map(nn.init.orthogonal_, self.parameters())

    def forward(self, x):
        return self.convs(x)


class ResidualDiscriminatorBlock(nn.Module):
    def __init__(self, in_dims, out_dims, first_conv=None):
        super().__init__()
        layers = [
            nn.utils.spectral_norm(
                nn.Conv2d(in_dims, out_dims, 3, padding=1, bias=False)),
            nn.BatchNorm2d(out_dims),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(
                nn.Conv2d(out_dims, out_dims, 3, padding=1, bias=False)),
            nn.BatchNorm2d(out_dims),
            nn.LeakyReLU(0.2, inplace=True),
            Interpolate(scale_factor=0.5, mode='bilinear'),
        ]
        if first_conv:
            layers = [
                nn.utils.spectral_norm(
                    nn.Conv2d(first_conv, in_dims, 1, bias=False)),
                nn.ReLU()] + layers
        self.convs = nn.Sequential(*layers)
        map(nn.init.orthogonal_, self.parameters())

    def forward(self, x):
        return self.convs(x)


class Interpolate(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return F.interpolate(x, *self.args, **self.kwargs)
