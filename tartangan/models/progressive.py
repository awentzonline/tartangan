from collections import namedtuple
from functools import reduce
from itertools import chain

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


ProgressiveConfig = namedtuple('ProgressiveConfig', 'latent_dims blocks')


class ProgressiveGenerator(nn.Module):
    def __init__(
        self, config, output_channels=3, optimizer_factory=torch.optim.Adam,
        base_size=4, device='cpu'
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.base_size = base_size
        self.output_channels = output_channels
        base_img_dims = self.base_size ** 2 * config.latent_dims
        self.base_img = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Linear(config.latent_dims, base_img_dims)),
            nn.ReLU(),
            nn.BatchNorm1d(base_img_dims),
        )
        map(nn.init.orthogonal_, self.base_img.parameters())
        self.blocks = nn.ModuleList()
        self.optimizer_factory = optimizer_factory
        self.optimizers = [
            self.optimizer_factory(self.base_img.parameters())
        ]  # per-block
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
        new_block = ProgressiveGeneratorBlock(in_dims, out_dims, upsample=not first_block).to(self.device)
        self.top_index += 1
        self.blocks.append(new_block)
        # add an output mapping
        self.prev_to_output = self.to_output
        self.to_output = ProgressiveGeneratorOutput(
            out_dims, self.output_channels
        ).to(self.device)
        # add an optimizer
        block_parameters = chain(
            new_block.parameters(), self.to_output.parameters()
        )
        self.optimizers.append(
            self.optimizer_factory(block_parameters)
        )
        return True

    def forward(self, z, blend=0):
        base_img = self.base_img(z)
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
            feats_head = F.interpolate(feats_head, scale_factor=2, mode='bilinear')
            head_out = self.prev_to_output(feats_head)
            tail_out = self.to_output(feats_tail)
            out = blend * head_out + (1 - blend) * tail_out
        else:
            feats = reduce(lambda feats, b: b(feats), self.blocks, base_img)
            out = self.to_output(feats)
        return out

    def zero_grad(self):
        for o in self.optimizers:
            o.zero_grad()

    def step_optimizers(self):
        for o in self.optimizers:
            o.step()

    @property
    def current_size(self):
        return int(self.base_size * 2 ** (len(self.blocks) - 1))

    def to(self, device, *args, **kwargs):
        self.device = device
        return super().to(device, *args, **kwargs)


class ProgressiveGeneratorBlock(nn.Module):
    def __init__(self, in_dims, out_dims, upsample=True):
        super().__init__()
        layers = [
            nn.utils.spectral_norm(
                nn.Conv2d(in_dims, out_dims, 3, padding=1, bias=False)),
            nn.ReLU(True),
            nn.BatchNorm2d(out_dims),
            nn.utils.spectral_norm(
                nn.Conv2d(out_dims, out_dims, 3, padding=1, bias=False)),
            nn.ReLU(True),
            nn.BatchNorm2d(out_dims),
        ]
        if upsample:
            layers.insert(0, Interpolate(scale_factor=2, mode='bilinear'))
        self.convs = nn.Sequential(*layers)
        map(nn.init.orthogonal_, self.parameters())

    def forward(self, x):
        return self.convs(x)


class ProgressiveGeneratorOutput(nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.convs = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(in_dims, out_dims, 1, padding=0, bias=False)),
            nn.Sigmoid()
        )
        map(nn.init.orthogonal_, self.parameters())

    def forward(self, x):
        return self.convs(x)

# Disciminators

class ProgressiveDiscriminator(nn.Module):
    def __init__(
        self, config, input_channels=3, output_channels=1,
        optimizer_factory=torch.optim.Adam, device='cpu'
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.blocks = nn.ModuleList()
        self.optimizer_factory = optimizer_factory
        self.optimizers = []  # per-block...maybe this is nuts?
        self.to_output = None#ProgressiveGeneratorOutput(base_dims, output_channels)
        self.prev_to_output = None
        self.top_index = 0
        self.add_block()

    def add_block(self):
        if self.top_index >= len(self.config.blocks) - 1:
            print('At highest block. Not adding.')
            return False
        d_config_blocks = list(reversed(self.config.blocks))
        in_dims, out_dims = d_config_blocks[self.top_index:self.top_index + 2]
        first_conv = self.input_channels if self.top_index == 0 else 0
        self.top_index += 1
        new_block = ProgressiveDiscriminatorBlock(in_dims, out_dims, first_conv).to(self.device)
        self.blocks.append(new_block)
        # add an output mapping
        self.prev_to_output = self.to_output
        self.to_output = ProgressiveDiscriminatorOutput(
            out_dims, self.output_channels
        ).to(self.device)
        block_parameters = chain(
            new_block.parameters(), self.to_output.parameters()
        )
        self.optimizers.append(
            self.optimizer_factory(block_parameters)
        )
        return True

    def forward(self, img, blend=0):
        if blend > 0:
            # get features up to the penultimate layer, and then the final one
            # upsample the penultimate features and blend them with the
            # new, last layer
            ms_head = self.blocks[:-1]
            m_tail = self.blocks[-1]
            # get the first N-1 output
            feats_head = reduce(lambda feats, b: b(feats), ms_head, img)
            feats_head = F.interpolate(feats_head, scale_factor=2, mode='bilinear')
            head_out = self.prev_to_output(feats_head)
            # then the new Nth output
            feats_tail = m_tail(feats_head)
            tail_out = self.to_output(feats_tail)
            out = blend * head_out + (1 - blend) * tail_out
        else:
            feats = reduce(lambda feats, b: b(feats), self.blocks, img)
            out = self.to_output(feats)
        return out

    def zero_grad(self):
        for o in self.optimizers:
            o.zero_grad()

    def step_optimizers(self):
        for o in self.optimizers:
            o.step()

    def to(self, device, *args, **kwargs):
        self.device = device
        return super().to(device, *args, **kwargs)


class ProgressiveDiscriminatorBlock(nn.Module):
    def __init__(self, in_dims, out_dims, first_conv=None):
        super().__init__()
        layers = [
            nn.utils.spectral_norm(
                nn.Conv2d(in_dims, out_dims, 3, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(out_dims),
            nn.utils.spectral_norm(
                nn.Conv2d(out_dims, out_dims, 3, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(out_dims),
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


class ProgressiveDiscriminatorOutput(nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.convs = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(in_dims, out_dims, 1, padding=0, bias=False)),
            nn.Sigmoid()
        )
        map(nn.init.orthogonal_, self.parameters())

    def forward(self, img):
        feats = self.convs(img)
        return F.avg_pool2d(feats, feats.size()[2:]).view(-1, 1)


class Interpolate(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return F.interpolate(x, *self.args, **self.kwargs)


GAN_CONFIGS = {
    64: ProgressiveConfig(
        latent_dims=128,
        blocks=(
            128,  # 4,
            128,  # 8,
            64,  # 16,
            32,  # 32,
            16,  # 64,
        )
    ),
    128: ProgressiveConfig(
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
    256: ProgressiveConfig(
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
    512: ProgressiveConfig(
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
    1024: ProgressiveConfig(
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
    )
}
