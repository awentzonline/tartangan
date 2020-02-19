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
from .blocks import (
    DiscriminatorBlock, DiscriminatorOutput, DiscriminatorInput,
    GeneratorBlock, GeneratorInputMLP, GeneratorOutput,
    IQNDiscriminatorOutput,
    ResidualDiscriminatorBlock, ResidualGeneratorBlock,
    SelfAttention2d, TiledZGeneratorInput
)
from .pluggan import BlockModel, Discriminator, Generator


class SharedGenerator(Generator):
    """
    Generator using shared blocks
    """
    default_input = GeneratorInputMLP
    default_block = ResidualGeneratorBlock
    default_output = GeneratorOutput

    def build(self):
        self.input_block = self.input_factory(self.config.latent_dims, self.config.base_size)
        self.input_conv = self.block_factory(self.config.latent_dims, self.config.block_dims, first_block=True)
        self.shared_conv = self.block_factory(self.config.block_dims, self.config.block_dims)
        self.output_block = self.output_factory(self.config.block_dims, self.config.data_dims)
        self.attention_block = SelfAttention2d(self.config.block_dims)

    def forward(self, x):
        x = self.input_block(x)
        x = self.input_conv(x)
        depth = self.config.num_scales - 1
        for i in range(depth):
            x = self.shared_conv(x)
        return self.output_block(x)

    @property
    def max_size(self):
        num_scale_blocks = self.config.num_scales
        max_scale = 2 ** num_scale_blocks
        return max_scale * self.config.base_size


class SharedIQNDiscriminator(Discriminator):
    default_output = IQNDiscriminatorOutput

    def build(self):
        self.input_block = self.input_factory(self.config.data_dims, self.config.block_dims)
        self.shared_conv = self.block_factory(self.config.block_dims, self.config.block_dims)
        self.output_block = self.output_factory(self.config.block_dims, 1)
        self.attention_block = SelfAttention2d(self.config.block_dims)

    def forward(self, x, targets=None):
        x = self.input_block(x)
        depth = self.config.num_scales
        for i in range(depth):
            x = self.shared_conv(x)
        return self.output_block(x, targets=targets)


class SharedGANConfig(
    namedtuple(
        'SharedGANConfig',
        'base_size, latent_dims, data_dims, num_scales, block_dims'
    )
):
    def scale_model(self, scale):
        scaled = list(map(lambda x: int(x * scale), self.blocks))
        kwargs = self._asdict()
        kwargs['blocks'] = scaled
        return self.__class__(**kwargs)


SHARED_GAN_CONFIGS = {
    '16': SharedGANConfig(
        base_size=4,
        data_dims=3,
        latent_dims=100,
        num_scales=2,
        block_dims=64,
    ),
    # '32': GANConfig(
    #     base_size=4,
    #     data_dims=3,
    #     latent_dims=128,
    #     attention=(1,),
    #     num_blocks_per_scale=1,
    #     blocks=(
    #         128,  # 8,
    #         64,  # 16,
    #         32,  # 32,
    #     )
    # ),
    # '64': GANConfig(
    #     base_size=4,
    #     data_dims=3,
    #     latent_dims=128,
    #     attention=(2,),
    #     num_blocks_per_scale=1,
    #     blocks=(
    #         128,  # 8,
    #         128,  # 16,
    #         64,  # 32,
    #         32,  # 64,
    #     )
    # ),
    '128': SharedGANConfig(
        base_size=4,
        data_dims=3,
        latent_dims=256,
        num_scales=5,
        block_dims=128,
    ),
    # '256': GANConfig(
    #     base_size=4,
    #     data_dims=3,
    #     latent_dims=256,
    #     attention=(3,),
    #     num_blocks_per_scale=1,
    #     blocks=(
    #         256,  # 8,
    #         256,  # 16,
    #         128,  # 32,
    #         64,  # 64,
    #         32,  # 128,
    #         16,  # 256
    #     )
    # ),
    # '512': GANConfig(
    #     base_size=4,
    #     data_dims=3,
    #     latent_dims=512,
    #     attention=(3,),
    #     num_blocks_per_scale=1,
    #     blocks=(
    #         256,  # 8
    #         256,  # 16
    #         256,  # 32
    #         128,  # 64
    #         64,  # 128
    #         32,  # 256
    #         16   # 512
    #     )
    # ),
    # '512thin': GANConfig(
    #     base_size=4,
    #     data_dims=3,
    #     latent_dims=256,
    #     attention=(3,),
    #     num_blocks_per_scale=1,
    #     blocks=(
    #         128,  # 8,
    #         128,  # 16,
    #         128,  # 32,
    #         64,  # 64,
    #         32,  # 128,
    #         16,  # 256
    #         8   # 512
    #     )
    # ),
    # # Test the effects of shortcut projection to other n-filters
    # '512thin-test': GANConfig(
    #     base_size=4,
    #     data_dims=3,
    #     latent_dims=128,
    #     attention=(3,),
    #     num_blocks_per_scale=1,
    #     blocks=(
    #         128,  # 8,
    #         120,  # 16,
    #         100,  # 32,
    #         64,  # 64,
    #         32,  # 128,
    #         16,  # 256
    #         8   # 512
    #     )
    # ),
    # '1024': GANConfig(
    #     base_size=4,
    #     data_dims=3,
    #     latent_dims=512,
    #     attention=(3,),
    #     num_blocks_per_scale=1,
    #     blocks=(
    #         512,  # 8,
    #         512,  # 16,
    #         512,  # 32,
    #         256,  # 64,
    #         128,  # 128,
    #         64,  # 256
    #         32,  # 512
    #         16,  # 1024
    #     )
    # ),
    # '1024thin': GANConfig(
    #     base_size=4,
    #     data_dims=3,
    #     latent_dims=256,
    #     attention=(3,),
    #     num_blocks_per_scale=1,
    #     blocks=(
    #         256,  # 8,
    #         256,  # 16,
    #         256,  # 32,
    #         128,  # 64,
    #         64,  # 128,
    #         32,  # 256
    #         16,  # 512
    #         8,  # 1024
    #     )
    # ),
    # 'test128': GANConfig(
    #     base_size=4,
    #     data_dims=3,
    #     latent_dims=64,
    #     attention=(3,),
    #     num_blocks_per_scale=1,
    #     blocks=(
    #         64,  # 8,
    #         32,  # 16,
    #         16,  # 32,
    #         8,  # 64,
    #         4,  # 128,
    #     )
    # ),
    # 'test256': GANConfig(
    #     base_size=4,
    #     data_dims=3,
    #     latent_dims=256,
    #     attention=(3,),
    #     num_blocks_per_scale=1,
    #     blocks=(
    #         200,  # 8,
    #         180,  # 16,
    #         128,  # 32,
    #         64,  # 64,
    #         32,  # 128
    #         16   # 256
    #     )
    # ),
}
