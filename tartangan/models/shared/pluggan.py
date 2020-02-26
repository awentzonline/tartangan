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
    GeneratorInputMLP, DiscriminatorInput, DiscriminatorOutput, GeneratorOutput,
    IQNDiscriminatorOutput, SelfAttention2d, TiledZGeneratorInput
)
from .blocks import (
    SharedConvBlock, SharedResidualDiscriminatorBlock,
    SharedResidualGeneratorBlock
)

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
        max_in_filters = max([config.latent_dims] + config.blocks)
        max_out_filters = max(config.blocks)
        filter_size = 3
        self.shared_filters = nn.Parameter(
            torch.randn(
                max_out_filters, max_in_filters, filter_size, filter_size,
                requires_grad=True
            ) * 0.1
        )
        self.build()

    def forward(self, x):
        return self.blocks(x)

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
    default_block = SharedResidualGeneratorBlock
    default_output = GeneratorOutput

    def build(self):
        blocks = [
            self.input_factory(self.config.latent_dims, self.config.base_size)
        ]
        in_dims = self.config.latent_dims
        apply_norm = False
        for block_i, out_dims in enumerate(self.config.blocks):
            scale_blocks = []

            if self.config.attention and block_i in self.config.attention:
                scale_blocks.append(SelfAttention2d(in_dims))

            scale_blocks.append(
                self.block_factory(
                    self.shared_filters, in_dims, out_dims, apply_norm=apply_norm
                )
            )
            apply_norm = True

            blocks += scale_blocks
            in_dims = out_dims
        blocks.append(
            self.output_factory(out_dims, self.config.data_dims)
        )
        self.blocks = nn.Sequential(*blocks)


class SharedDiscriminator(SharedModel):
    default_input = DiscriminatorInput
    default_block = SharedResidualGeneratorBlock
    default_output = DiscriminatorOutput

    def build(self):
        first_block_input_dims = next(reversed(self.config.blocks))
        blocks = [
            self.input_factory(self.config.data_dims, first_block_input_dims)
        ]
        in_dims = first_block_input_dims
        apply_norm = False
        for block_i, out_dims in reversed(list(enumerate(self.config.blocks))):
            scale_blocks = []
            scale_blocks.append(
                self.block_factory(
                    self.shared_filters, in_dims, out_dims, apply_norm=apply_norm,
                )
            )
            apply_norm = True

            if self.config.attention and block_i in self.config.attention:
                scale_blocks.append(SelfAttention2d(out_dims))

            blocks += scale_blocks
            in_dims = out_dims
        blocks.append(
            self.output_factory(out_dims, 1)
        )
        self.blocks = nn.Sequential(*blocks)


class SharedIQNDiscriminator(SharedDiscriminator):
    default_output = IQNDiscriminatorOutput

    def build(self):
        first_block_input_dims = next(reversed(self.config.blocks))
        blocks = [
            self.input_factory(self.config.data_dims, first_block_input_dims)
        ]
        in_dims = first_block_input_dims
        apply_norm = False
        for block_i, out_dims in reversed(list(enumerate(self.config.blocks))):
            scale_blocks = []
            scale_blocks.append(
                self.block_factory(
                    self.shared_filters, in_dims, out_dims, apply_norm=apply_norm,
                )
            )
            apply_norm = True

            if self.config.attention and block_i in self.config.attention:
                scale_blocks.append(SelfAttention2d(out_dims))

            blocks += scale_blocks
            in_dims = out_dims
        self.blocks = nn.Sequential(*blocks)
        self.to_output = self.output_factory(out_dims, 1)

    def forward(self, x, targets=None):
        y = self.blocks(x)
        out = self.to_output(y, targets=targets)
        return out
