from collections import namedtuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class PPANConfig(
    namedtuple(
        'PPANConfig',
        'latent_dims, data_dims, blocks'
    )
):
    def scale_model(self, scale):
        scaled = list(map(lambda x: int(x * scale), self.blocks))
        kwargs = self._asdict()
        kwargs['blocks'] = scaled
        return self.__class__(**kwargs)


class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.build()

    def build(self):
        raise NotImplementedError


class Generator(BaseModel):

    def build(self):
        in_dims = self.config.blocks[0]
        blocks = []
        for block_i, out_dims in enumerate(self.config.blocks):

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


PPAN_CONFIGS = {
    '16': PPANConfig(
        data_dims=3,
        latent_dims=100,
        blocks=(
            64,  # 8,
            32,  # 16,
        )
    ),
    '32': PPANConfig(
        data_dims=3,
        latent_dims=128,
        blocks=(
            128,  # 8,
            64,  # 16,
            32,  # 32,
        )
    ),
    '64': PPANConfig(
        data_dims=3,
        latent_dims=128,
        blocks=(
            128,  # 8,
            128,  # 16,
            64,  # 32,
            32,  # 64,
        )
    ),
    '128': PPANConfig(
        data_dims=3,
        latent_dims=256,
        blocks=(
            128,  # 8,
            128,  # 16,
            64,  # 32,
            32,  # 64,
            16,  # 128,
        )
    ),
}
