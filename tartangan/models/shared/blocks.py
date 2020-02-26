import torch
from torch import nn
import torch.nn.functional as F


class SharedConvBlock(nn.Module):
    def __init__(self, shared_filters, in_dims, out_dims,
                 pre_interpolate=None, post_interpolate=None,
                 apply_norm=True, bias=True,
                 norm_factory=nn.BatchNorm2d, activation_factory=nn.LeakyReLU):
        super().__init__()

        self.norm_and_activate = [
            norm_factory(in_dims),
            activation_factory()
        ]
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_dims, requires_grad=True)
            )
        else:
            self.bias = None

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.apply_norm = apply_norm
        self.pre_interpolate = pre_interpolate
        self.post_interpolate = post_interpolate
        self.project_input = in_dims != out_dims

    def forward(self, x):
        if self.pre_interpolate is not None:
            x = F.interpolate(
                x, scale_factor=self.pre_interpolate, mode='bilinear',
                align_corners=True
            )

        if self.apply_norm:
            x = self.norm_and_activate(x)

        x = F.conv2d(
            x, self.shared_filters[:self.out_dims, :self.in_dims],
            bias=self.bias, padding=1
        )

        if self.post_interpolate is not None:
            x = F.interpolate(
                x, scale_factor=self.post_interpolate, mode='bilinear',
                align_corners=True
            )
        if self.project_input:
            x = F.conv2d(
                x, self.shared_filters[:self.out_dims, :self.in_dims],
                padding=1
            )
        return x


class SharedResidualGeneratorBlock(nn.Module):
    def __init__(self, shared_filters, in_dims, out_dims,
                 apply_norm=True, bias=True,
                 norm_factory=nn.BatchNorm2d, activation_factory=nn.LeakyReLU):
        super().__init__()
        self.blocks = nn.Sequential([
            SharedConvBlock(
                shared_filters, in_dims, out_dims,
                pre_interpolate=2.,
                norm_factory=norm_factory, activation_factory=activation_factory
            ),
            SharedConvBlock(
                shared_filters, out_dims, out_dims,
                norm_factory=norm_factory, activation_factory=activation_factory
            ),
        ])

    def forward(self, x):
        h = self.blocks(x)
        x = F.interpolate(
            x, scale_factor=2., mode='bilinear',
            align_corners=True
        )
        return x + h
