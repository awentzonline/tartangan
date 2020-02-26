"""
Share parameters between blocks.
"""
from collections import deque, namedtuple
from functools import reduce
from itertools import chain

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from ..blocks import (
    GeneratorInputMLP,
    IQNDiscriminatorOutput,
    SelfAttention2d, TiledZGeneratorInput
)
from .blocks import SharedConvBlock


class SharedModel(nn.Module):
    default_input = GeneratorInputMLP
    default_block = SharedConvBlock
    default_output = GeneratorOutput

    def __init__(
        self, config,
        input_factory=None,
        block_factory=None,
        output_factory=None
    ):
        super().__init__()
        self.config = config
        self.input_factory = input_factory or self.default_input
        self.block_factory = block_factory or self.default_block
        self.output_factory = output_factory or self.default_output
        num_filters = max(config.blocks)
        filter_size = config.filter_size
        self.shared_filters = nn.Parameter(
            torch.randn(
                num_filters, num_filters, filter_size, filter_size,
                requires_grad=True
            ) * 0.1
        )
        self.build()

    def forward(self, x):
        return reduce(lambda f, b: b(f), self.blocks, x)

    @property
    def max_size(self):
        num_scale_blocks = len(self.config.blocks)
        max_scale = 2 ** num_scale_blocks
        return max_scale * self.config.base_size


class SharedGenerator(SharedModel):
    """
    Generator using shared blocks
    """
    default_input = GeneratorInputMLP
    default_block = SharedConvBlock
    default_output = GeneratorOutput

    def build(self):
        blocks = [
            self.input_factory(self.config.latent_dims, self.config.base_size)
        ]
        in_dims = self.config.latent_dims
        apply_norm = False
        for block_i, out_dims in enumerate(self.config.blocks):
            scale_blocks = [
                self.block_factory(
                    self.shared_filters, in_dims, out_dims, apply_norm=apply_norm,
                    pre_interpolate=2.
                )
            ]
            apply_norm = True
            scale_blocks.append(
                self.block_factory(
                    self.shared_filters, out_dims, out_dims, apply_norm=apply_norm
                )
            )

            if self.config.attention and block_i in self.config.attention:
                scale_blocks.append(SelfAttention2d(out_dims))

            blocks += scale_blocks
            in_dims = out_dims
        blocks.append(
            self.output_factory(out_dims, self.config.data_dims)
        )
        self.blocks = nn.Sequential(*blocks)


class SharedDiscriminator(Discriminator):
    default_input = DiscriminatorInput
    default_block = SharedConvBlock
    default_output = DiscriminatorOutput

    def build(self):
        first_block_input_dims = next(reversed(self.config.blocks))
        blocks = [
            self.input_factory(self.config.data_dims, first_block_input_dims),
        ]
        in_dims = first_block_input_dims
        first_block = True
        for block_i, out_dims in reversed(list(enumerate(self.config.blocks))):
            block = self.block_factory(in_dims, out_dims, first_block=first_block)
            blocks.append(block)
            if self.config.attention and block_i in self.config.attention:
                blocks.append(SelfAttention2d(out_dims))
            in_dims = out_dims
            first_block = False
        blocks.append(
            self.output_factory(out_dims, 1)
        )
        self.blocks = nn.Sequential(*blocks)

    def build(self):
        blocks = [
            self.input_factory(self.config.latent_dims, self.config.base_size)
        ]
        in_dims = self.config.latent_dims
        apply_norm = False
        for block_i, out_dims in enumerate(self.config.blocks):
            scale_blocks = [
                self.block_factory(
                    self.shared_filters, in_dims, out_dims, apply_norm=apply_norm,
                    pre_interpolate=2.
                )
            ]
            apply_norm = True
            scale_blocks.append(
                self.block_factory(
                    self.shared_filters, out_dims, out_dims, apply_norm=apply_norm
                )
            )

            if self.config.attention and block_i in self.config.attention:
                scale_blocks.append(SelfAttention2d(out_dims))

            blocks += scale_blocks
            in_dims = out_dims
        blocks.append(
            self.output_factory(out_dims, self.config.data_dims)
        )
        self.blocks = nn.Sequential(*blocks)
