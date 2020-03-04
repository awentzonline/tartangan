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
    ResidualDiscriminatorBlock, ResidualGeneratorBlock,
    SelfAttention2d, TiledZGeneratorInput
)


class GANConfig(
    namedtuple(
        'GANConfig',
        'base_size, latent_dims, data_dims, blocks, num_blocks_per_scale, attention'
    )
):
    def scale_model(self, scale):
        scaled = list(map(lambda x: int(x * scale), self.blocks))
        kwargs = self._asdict()
        kwargs['blocks'] = scaled
        return self.__class__(**kwargs)


class BlockModel(nn.Module):
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
        self.build()

    def build(self):
        raise NotImplementedError

    def forward(self, x):
        return reduce(lambda f, b: b(f), self.blocks, x)

    @property
    def max_size(self):
        num_scale_blocks = len(self.config.blocks)
        max_scale = 2 ** num_scale_blocks
        return max_scale * self.config.base_size


class Generator(BlockModel):
    default_input = TiledZGeneratorInput
    default_block = GeneratorBlock#ResidualGeneratorBlock
    default_output = GeneratorOutput

    def build(self):
        blocks = [
            self.input_factory(self.config.latent_dims, self.config.base_size)
        ]
        in_dims = self.config.latent_dims
        num_blocks_per_scale = self.config.num_blocks_per_scale
        first_block = True
        for block_i, out_dims in enumerate(self.config.blocks):
            scale_blocks = [self.block_factory(in_dims, out_dims, first_block=first_block)]
            first_block = False
            for i in range(num_blocks_per_scale - 1):
                scale_blocks.append(self.block_factory(out_dims, out_dims, upsample=False))
            if self.config.attention and block_i in self.config.attention:
                scale_blocks.append(SelfAttention2d(out_dims))
            blocks += scale_blocks
            in_dims = out_dims
        blocks.append(
            self.output_factory(out_dims, self.config.data_dims)
        )
        self.blocks = nn.Sequential(*blocks)
        #self.blocks = nn.ModuleList(blocks)


class Discriminator(BlockModel):
    default_input = DiscriminatorInput
    default_block = DiscriminatorBlock#ResidualDiscriminatorBlock
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
        #self.blocks = nn.ModuleList(blocks)


class IQNDiscriminator(Discriminator):
    default_output = DiscriminatorOutput

    def build(self):
        blocks = []
        in_dims = self.config.data_dims
        for block_i, out_dims in reversed(list(enumerate(self.config.blocks))):
            block = self.block_factory(in_dims, out_dims)
            blocks.append(block)
            if self.config.attention and block_i in self.config.attention:
                blocks.append(SelfAttention2d(out_dims))
            in_dims = out_dims
        self.to_output = self.output_factory(out_dims, 1)
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x, targets=None):
        y = self.blocks(x)
        out = self.to_output(y, targets=targets)
        return out


GAN_CONFIGS = {
    '16': GANConfig(
        base_size=4,
        data_dims=3,
        latent_dims=100,
        attention=(1,),
        num_blocks_per_scale=1,
        blocks=(
            64,  # 8,
            32,  # 16,
        )
    ),
    '32': GANConfig(
        base_size=4,
        data_dims=3,
        latent_dims=128,
        attention=(1,),
        num_blocks_per_scale=1,
        blocks=(
            128,  # 8,
            64,  # 16,
            32,  # 32,
        )
    ),
    '64': GANConfig(
        base_size=4,
        data_dims=3,
        latent_dims=128,
        attention=(1,),
        num_blocks_per_scale=1,
        blocks=(
            128,  # 8,
            128,  # 16,
            64,  # 32,
            32,  # 64,
        )
    ),
    '128': GANConfig(
        base_size=4,
        data_dims=3,
        latent_dims=256,
        attention=(2,),
        num_blocks_per_scale=1,
        blocks=(
            128,  # 8,
            128,  # 16,
            64,  # 32,
            32,  # 64,
            16,  # 128,
        )
    ),
    '128big': GANConfig(
        base_size=4,
        data_dims=3,
        #latent_dims=256,
        latent_dims=256,
        attention=(2,),
        num_blocks_per_scale=1,
        blocks=(
            1024,  # 8,
            1024,  # 16,
            512,  # 32,
            256,  # 64,
            128,  # 128,
        )
    ),
    '256': GANConfig(
        base_size=4,
        data_dims=3,
        latent_dims=256,
        attention=(3,),
        num_blocks_per_scale=1,
        blocks=(
            256,  # 8,
            256,  # 16,
            128,  # 32,
            64,  # 64,
            32,  # 128,
            16,  # 256
        )
    ),
    '256big': GANConfig(
        base_size=4,
        data_dims=3,
        latent_dims=256,
        attention=(3,),
        num_blocks_per_scale=1,
        blocks=(
            1024,  # 8,
            1024,  # 16,
            512,  # 32,
            256,  # 64,
            128,  # 128,
            64,  # 256
        )
    ),
    '512': GANConfig(
        base_size=4,
        data_dims=3,
        latent_dims=512,
        attention=(3,),
        num_blocks_per_scale=1,
        blocks=(
            256,  # 8
            256,  # 16
            256,  # 32
            128,  # 64
            64,  # 128
            32,  # 256
            16   # 512
        )
    ),
    '512thin': GANConfig(
        base_size=4,
        data_dims=3,
        latent_dims=256,
        attention=(3,),
        num_blocks_per_scale=1,
        blocks=(
            128,  # 8,
            128,  # 16,
            128,  # 32,
            64,  # 64,
            32,  # 128,
            16,  # 256
            8   # 512
        )
    ),
    # Test the effects of shortcut projection to other n-filters
    '512thin-test': GANConfig(
        base_size=4,
        data_dims=3,
        latent_dims=128,
        attention=(3,),
        num_blocks_per_scale=1,
        blocks=(
            128,  # 8,
            120,  # 16,
            100,  # 32,
            64,  # 64,
            32,  # 128,
            16,  # 256
            8   # 512
        )
    ),
    '1024': GANConfig(
        base_size=4,
        data_dims=3,
        latent_dims=512,
        attention=(3,),
        num_blocks_per_scale=1,
        blocks=(
            512,  # 8,
            512,  # 16,
            512,  # 32,
            256,  # 64,
            128,  # 128,
            64,  # 256
            32,  # 512
            16,  # 1024
        )
    ),
    '1024thin': GANConfig(
        base_size=4,
        data_dims=3,
        latent_dims=256,
        attention=(3,),
        num_blocks_per_scale=1,
        blocks=(
            256,  # 8,
            256,  # 16,
            256,  # 32,
            128,  # 64,
            64,  # 128,
            32,  # 256
            16,  # 512
            8,  # 1024
        )
    ),
    'test128': GANConfig(
        base_size=4,
        data_dims=3,
        latent_dims=64,
        attention=(3,),
        num_blocks_per_scale=1,
        blocks=(
            64,  # 8,
            32,  # 16,
            16,  # 32,
            8,  # 64,
            4,  # 128,
        )
    ),
    'test256': GANConfig(
        base_size=4,
        data_dims=3,
        latent_dims=256,
        attention=(3,),
        num_blocks_per_scale=1,
        blocks=(
            200,  # 8,
            180,  # 16,
            128,  # 32,
            64,  # 64,
            32,  # 128
            16   # 256
        )
    ),
}
