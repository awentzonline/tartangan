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
    TiledZGeneratorInput, WeightedComponents
)

ProgressiveConfig = namedtuple('ProgressiveConfig', 'latent_dims blocks')


class ProgressiveGenerator(nn.Module):
    def __init__(
        self, config, output_channels=3, optimizer_factory=torch.optim.Adam,
        base_size=4, device='cpu', block_class=ResidualGeneratorBlock,
        input_class=WeightedComponents, output_class=GeneratorOutput
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.base_size = base_size
        self.output_channels = output_channels
        self.input_block = input_class(config.latent_dims, size=base_size)
        self.blocks = nn.ModuleList()
        self.block_class = block_class
        self.output_class = output_class
        self.optimizer_factory = optimizer_factory
        self.output_optimizers = deque([], 2)
        self.optimizers = [
        #    self.optimizer_factory(self.input_block.parameters())
        ]  # per-block
        map(nn.init.orthogonal_, self.input_block.parameters())
        self.to_output = None
        self.prev_to_output = None
        self.top_index = 0
        self.add_block()

    def add_block(self):
        if self.top_index >= len(self.config.blocks) - 1:
            print('At highest block. Not adding.')
            return False
        in_dims, out_dims = self.config.blocks[self.top_index:self.top_index + 2]
        first_block = self.top_index == 0
        new_block = self.block_class(in_dims, out_dims, upsample=not first_block).to(self.device)
        self.top_index += 1
        self.blocks.append(new_block)
        # add an output mapping
        self.prev_to_output = self.to_output
        self.to_output = self.output_class(
            out_dims, self.output_channels
        ).to(self.device)
        # add an optimizer
        self.output_optimizers.append(
            self.optimizer_factory(self.to_output.parameters())
        )
        self.optimizers.append(
            self.optimizer_factory(new_block.parameters())
        )
        return True

    def forward(self, z, blend=0):
        base_img = self.input_block(z)
        base_img = base_img.view(
            -1, self.config.latent_dims, self.base_size, self.base_size)
        if blend > 0:
            # get features up to the penultimate layer, and then the final one
            # upsample the penultimate features and blend them with the
            # new, last layer
            ms_head = self.blocks[:-1]
            m_tail = self.blocks[-1]
            # get the first N-1 output
            feats_head = reduce(lambda feats, b: b(feats), ms_head, base_img)
            feats_tail = m_tail(feats_head)
            head_out = self.prev_to_output(feats_head)
            head_out = F.interpolate(
                head_out, scale_factor=2, mode='bilinear', align_corners=True)
            tail_out = self.to_output(feats_tail)
            out = blend * head_out + (1 - blend) * tail_out
        else:
            feats = reduce(lambda feats, b: b(feats), self.blocks, base_img)
            out = self.to_output(feats)
        return out

    def zero_grad(self):
        for o in chain(self.optimizers, self.output_optimizers):
            o.zero_grad()

    def step_optimizers(self):
        for o in chain(self.optimizers, self.output_optimizers):
            o.step()

    @property
    def current_size(self):
        return int(self.base_size * 2 ** (len(self.blocks) - 1))

    def to(self, device, *args, **kwargs):
        self.device = device
        return super().to(device, *args, **kwargs)


# Discriminators

class ProgressiveDiscriminator(nn.Module):
    def __init__(
        self, config, input_channels=3, output_channels=1,
        optimizer_factory=torch.optim.Adam, device='cpu',
        block_class=ResidualDiscriminatorBlock, output_class=DiscriminatorOutput,
        input_class=DiscriminatorInput, norm_factory=nn.BatchNorm2d
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.blocks = nn.ModuleList()
        self.block_class = block_class
        self.input_class = input_class
        self.output_class = output_class
        self.optimizer_factory = optimizer_factory
        self.norm_factory = norm_factory
        self.from_input = None#GeneratorOutput(base_dims, output_channels)
        self.prev_from_input = None
        self.top_index = 0
        self.to_output = self.output_class(config.latent_dims, output_channels)
        self.input_optimizers = deque([], 2)
        self.optimizers = [
            self.optimizer_factory(self.to_output.parameters())
        ]  # per-block...maybe this is nuts?
        self.add_block()

    def add_block(self):
        if self.top_index >= len(self.config.blocks) - 1:
            print('At highest block. Not adding.')
            return False
        d_config_blocks = self.config.blocks#list(reversed(self.config.blocks))
        out_dims, in_dims = d_config_blocks[self.top_index:self.top_index + 2]
        first_conv = self.input_channels if self.top_index == 0 else 0
        self.top_index += 1
        new_block = self.block_class(in_dims, out_dims, first_conv).to(self.device)
        self.blocks.insert(0, new_block)
        # add an output mapping
        self.prev_from_input = self.from_input
        self.from_input = self.input_class(
            self.input_channels, in_dims
        ).to(self.device)
        self.input_optimizers.append(
            self.optimizer_factory(self.from_input.parameters())
        )
        self.optimizers.append(
            self.optimizer_factory(new_block.parameters())
        )
        return True

    def forward(self, img, blend=0):
        """
        blend == 0 means all tip o the network
        blend == 1 means all penultimate layer
        """
        if blend > 0 and len(self.blocks) > 1:
            # get features from first layer
            feats_new = self.from_input(img)
            newest_block = self.blocks[0]
            feats_new = newest_block(feats_new)
            # scale down the input and get features from the last input mapper
            img = F.interpolate(img, scale_factor=0.5, mode='bilinear', align_corners=True)
            feats_last = self.prev_from_input(img)
            blended_feats = blend * feats_last + (1. - blend) * feats_new
            # apply remaining transformations to the blended features
            remaining_blocks = self.blocks[1:]
            feats = reduce(lambda f, b: b(f), remaining_blocks, blended_feats)
        else:
            feats = self.from_input(img)
            feats = reduce(lambda f, b: b(f), self.blocks, feats)
        out = self.to_output(feats)
        return out

    def zero_grad(self):
        for o in chain(self.optimizers, self.input_optimizers):
            o.zero_grad()

    def step_optimizers(self):
        for o in chain(self.optimizers, self.input_optimizers):
            o.step()

    def to(self, device, *args, **kwargs):
        self.device = device
        return super().to(device, *args, **kwargs)


class ProgressiveIQNDiscriminator(ProgressiveDiscriminator):
    def forward(self, img, blend=0, targets=None):
        """
        blend == 0 means all tip o the network
        blend == 1 means all penultimate layer
        """
        if blend > 0 and len(self.blocks) > 1:
            # get features from first layer
            feats_new = self.from_input(img)
            newest_block = self.blocks[0]
            feats_new = newest_block(feats_new)
            # scale down the input and get features from the last input mapper
            img = F.interpolate(img, scale_factor=0.5, mode='bilinear', align_corners=True)
            feats_last = self.prev_from_input(img)
            # apply remaining transformations to the blended features
            blended_feats = blend * feats_last + (1. - blend) * feats_new
            remaining_blocks = self.blocks[1:]
            feats = reduce(lambda f, b: b(f), remaining_blocks, blended_feats)
        else:
            feats = self.from_input(img)
            feats = reduce(lambda f, b: b(f), self.blocks, feats)
        out = self.to_output(feats, targets=targets)
        return out


GAN_CONFIGS = {
    '64': ProgressiveConfig(
        latent_dims=128,
        blocks=(
            128,  # 4,
            128,  # 8,
            64,  # 16,
            32,  # 32,
            16,  # 64,
        )
    ),
    '128': ProgressiveConfig(
        latent_dims=256,
        blocks=(
            256,  # 4,
            256,  # 8,
            128,  # 16,
            64,  # 32,
            32,  # 64,
            16,  # 128
        )
    ),
    '256': ProgressiveConfig(
        latent_dims=256,
        blocks=(
            256,  # 4,
            256,  # 8,
            256,  # 16,
            128,  # 32,
            64,  # 64,
            32,  # 128
            16   # 256
        )
    ),
    '256thin': ProgressiveConfig(
        latent_dims=128,
        blocks=(
            128,  # 4,
            128,  # 8,
            128,  # 16,
            64,  # 32,
            32,  # 64,
            16,  # 128
            8   # 256
        )
    ),
    # Test the effects of shortcut projection to other n-filters
    '256thin-test': ProgressiveConfig(
        latent_dims=128,
        blocks=(
            128,  # 4,
            120,  # 8,
            100,  # 16,
            64,  # 32,
            32,  # 64,
            16,  # 128
            8   # 256
        )
    ),
    '512': ProgressiveConfig(
        latent_dims=512,
        blocks=(
            512,  # 4,
            512,  # 8,
            512,  # 16,
            256,  # 32,
            128,  # 64,
            64,  # 128
            32,  # 256
            16,  # 512
        )
    ),
    '512thin': ProgressiveConfig(
        latent_dims=256,
        blocks=(
            256,  # 4,
            256,  # 8,
            256,  # 16,
            128,  # 32,
            64,  # 64,
            32,  # 128
            16,  # 256
            8,  # 512
        )
    ),
    '1024': ProgressiveConfig(
        latent_dims=512,
        blocks=(
            512,  # 4,
            512,  # 8,
            512,  # 16,
            512,  # 32,
            256,  # 64,
            128,  # 128
            64,  # 256
            32,  # 512
            16,  # 1024
        )
    ),
    'test': ProgressiveConfig(
        latent_dims=64,
        blocks=(
            64,  # 4,
            32,  # 8,
            16,  # 16,
            8,  # 32,
            4,  # 64,
        )
    ),
    'test256': ProgressiveConfig(
        latent_dims=256,
        blocks=(
            256,  # 4,
            200,  # 8,
            180,  # 16,
            128,  # 32,
            64,  # 64,
            32,  # 128
            16   # 256
        )
    ),
}
