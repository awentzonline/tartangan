import torch
from torch import nn
import torch.nn.functional as F


class SharedConvBlock(nn.Module):
    """Performs a single convolution in a preactivation setting."""

    def __init__(self, shared_filters, in_dims, out_dims,
                 apply_norm=True, bias=True,
                 norm_factory=nn.BatchNorm2d, activation_factory=nn.LeakyReLU):
        super().__init__()

        self.norm_and_activate = nn.Sequential(
            norm_factory(in_dims),
            activation_factory()
        )
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_dims, requires_grad=True)
            )
        else:
            self.bias = None

        self.shared_filters = shared_filters
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.apply_norm = apply_norm

    def forward(self, x):
        if self.apply_norm:
            x = self.norm_and_activate(x)
        x = F.conv2d(
            x, narrow_filters(self.shared_filters, self.in_dims, self.out_dims),
            bias=self.bias, padding=1
        )
        return x


class SharedResidualGeneratorBlock(nn.Module):
    def __init__(self, shared_filters, in_dims, out_dims,
                 apply_norm=True, bias=True,
                 norm_factory=nn.BatchNorm2d, activation_factory=nn.LeakyReLU):
        super().__init__()
        self.blocks = nn.Sequential(
            SharedConvBlock(
                shared_filters, in_dims, out_dims, apply_norm=apply_norm, bias=bias,
                norm_factory=norm_factory, activation_factory=activation_factory
            ),
            SharedConvBlock(
                shared_filters, out_dims, out_dims, apply_norm=True, bias=bias,
                norm_factory=norm_factory, activation_factory=activation_factory
            ),
        )
        self.shared_filters = shared_filters
        self.in_dims = in_dims
        self.out_dims = out_dims

    def forward(self, x):
        x = F.interpolate(
            x, scale_factor=2, mode='bilinear',
            align_corners=True
        )
        h = self.blocks(x)
        if self.in_dims != self.out_dims:
            x = self.project_input(x)
        return x + h

    def project_input(self, x):
        x = F.conv2d(
            x, narrow_filters(self.shared_filters, self.in_dims, self.out_dims),
            padding=1
        )
        return x


class SharedResidualDiscriminatorBlock(nn.Module):
    def __init__(self, shared_filters, in_dims, out_dims,
                 apply_norm=True, bias=True,
                 norm_factory=nn.BatchNorm2d, activation_factory=nn.LeakyReLU):
        super().__init__()
        self.blocks = nn.Sequential(
            SharedConvBlock(
                shared_filters, in_dims, out_dims, apply_norm=apply_norm, bias=bias,
                norm_factory=norm_factory, activation_factory=activation_factory
            ),
            SharedConvBlock(
                shared_filters, out_dims, out_dims, apply_norm=True, bias=bias,
                norm_factory=norm_factory, activation_factory=activation_factory
            ),
        )
        self.shared_filters = shared_filters
        self.in_dims = in_dims
        self.out_dims = out_dims

    def forward(self, x):
        h = self.blocks(x)
        h = F.interpolate(
            h, scale_factor=0.5, mode='bilinear',
            align_corners=True
        )
        # resize and project input
        x = F.interpolate(
            x, scale_factor=0.5, mode='bilinear',
            align_corners=True
        )
        if self.in_dims != self.out_dims:
            x = self.project_input(x)
        return x + h

    def project_input(self, x):
        x = F.conv2d(
            x, narrow_filters(self.shared_filters, self.in_dims, self.out_dims),
            padding=1
        )
        return x


def narrow_filters(filters, in_dims, out_dims):
    """Extract the first NxM filters."""
    filters = filters.narrow(0, 0, out_dims).narrow(1, 0, in_dims)
    return filters
