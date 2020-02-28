import argparse
import functools
import os

import numpy as np
import torch
from torch import nn
import torch.utils.data as data_utils
import torchvision
from torchvision import transforms
import tqdm

from tartangan.models.blocks import (
    GeneratorBlock, DiscriminatorBlock, IQNDiscriminatorOutput,
    ResidualDiscriminatorBlock, ResidualGeneratorBlock,
    GeneratorInputMLP, TiledZGeneratorInput,
    GeneratorOutput, DiscriminatorOutput,
)
from tartangan.models.layers import PixelNorm
from tartangan.models.losses import (
    discriminator_hinge_loss, generator_hinge_loss, gradient_penalty
)
from tartangan.models.pluggan import IQNDiscriminator, Generator, GAN_CONFIGS

from .trainer import Trainer


class IQNTrainer(Trainer):
    def build_models(self):
        self.gan_config = GAN_CONFIGS[self.args.config]
        self.gan_config = self.gan_config.scale_model(self.args.model_scale)
        norm_factory = {
            'id': nn.Identity,
            'bn': nn.BatchNorm2d,
        }[self.args.norm]
        g_input_factory = {
            'mlp': GeneratorInputMLP,
            'tiledz': TiledZGeneratorInput,
        }[self.args.g_base]
        g_block_factory = functools.partial(
            ResidualGeneratorBlock, norm_factory=norm_factory
        )
        d_block_factory = functools.partial(
            ResidualDiscriminatorBlock, norm_factory=norm_factory
        )
        g_output_factory = functools.partial(
            GeneratorOutput, norm_factory=norm_factory
        )
        d_output_factory = functools.partial(
            IQNDiscriminatorOutput, norm_factory=norm_factory
        )
        self.g = Generator(
            self.gan_config,
            input_factory=g_input_factory,
            block_factory=g_block_factory,
            output_factory=g_output_factory,
        ).to(self.device)
        self.target_g = Generator(
            self.gan_config,
            input_factory=g_input_factory,
            block_factory=g_block_factory,
            output_factory=g_output_factory,
        ).to(self.device)
        self.update_target_generator(1.)  # copy weights

        self.d = IQNDiscriminator(
            self.gan_config,
            block_factory=d_block_factory,
            output_factory=IQNDiscriminatorOutput,
        ).to(self.device)
        self.optimizer_g = torch.optim.Adam(self.g.parameters(), lr=self.args.lr_g, betas=(0., 0.999))
        self.optimizer_d = torch.optim.Adam(self.d.parameters(), lr=self.args.lr_d, betas=(0., 0.999))
        print(self.g)
        print(self.d)

    def train_batch(self, imgs):
        imgs = imgs.to(self.device)
        self.g.train()
        self.d.train()
        # train discriminator
        toggle_grad(self.g, False)
        toggle_grad(self.d, True)
        self.optimizer_d.zero_grad()
        batch_imgs, labels = self.make_adversarial_batch(imgs)
        real, fake = batch_imgs[:self.args.batch_size], batch_imgs[self.args.batch_size:]
        # torchvision.utils.save_image(real, 'batch_real.png', normalize=True, range=(-1, 1))
        # torchvision.utils.save_image(fake, 'batch_fake.png', normalize=True, range=(-1, 1))
        if self.args.grad_penalty:
            real.requires_grad_()
        p_labels_real, d_loss_real = self.d(real, targets=labels[:len(labels) // 2])
        p_labels_fake, d_loss_fake = self.d(fake.detach(), targets=labels[len(labels) // 2])
        d_loss = d_loss_real + d_loss_fake
        d_grad_penalty = 0.
        if self.args.grad_penalty:
            d_grad_penalty = self.args.grad_penalty * gradient_penalty(p_labels_real, real)
            d_loss += d_grad_penalty
        d_loss.backward()
        self.optimizer_d.step()

        # train generator
        toggle_grad(self.g, True)
        toggle_grad(self.d, False)
        self.optimizer_g.zero_grad()
        batch_imgs, labels = self.make_generator_batch(imgs)
        #torchvision.utils.save_image(batch_imgs, 'batch.png', normalize=True, range=(-1, 1))
        p_labels, g_loss = self.d(batch_imgs, targets=labels)
        g_loss.backward()
        self.optimizer_g.step()

        self.update_target_generator()

        return dict(
            g_loss=float(g_loss), d_loss=float(d_loss),
            gp=float(d_grad_penalty)
        )

    @torch.no_grad()
    def update_target_generator(self, lr=None):
        if lr is None:
            lr = self.args.lr_target_g
        for g_p, target_g_p in zip(self.g.parameters(), self.target_g.parameters()):
            target_g_p.add_(
                (g_p - target_g_p) * self.args.lr_target_g
            )


def toggle_grad(model, on_or_off):
  for param in model.parameters():
    param.requires_grad_(on_or_off)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('data_path')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--gen-freq', type=int, default=200)
    p.add_argument('--lr-g', type=float, default=1e-4)
    p.add_argument('--lr-d', type=float, default=4e-4)
    p.add_argument('--lr-target-g', type=float, default=1e-3)
    p.add_argument('--device', default='cpu')
    p.add_argument('--epochs', type=int, default=10000)
    p.add_argument('--sample-file', default='sample/tartangan')
    p.add_argument('--checkpoint-freq', type=int, default=100000)
    p.add_argument('--checkpoint', default='checkpoints/tartangan')
    p.add_argument('--dataset-cache', default='cache/{root}_{size}.pkl')
    p.add_argument('--grad-penalty', type=float, default=5.)
    p.add_argument('--config', default='64')
    p.add_argument('--model-scale', type=float, default=1.)
    p.add_argument('--cache-dataset', action='store_true')
    p.add_argument('--log-dir', default='./logs')
    p.add_argument('--tensorboard', action='store_true')
    p.add_argument('--g-base', default='mlp', help='mlp or tiledz')
    p.add_argument('--norm', default='bn', help='bn or id')
    args = p.parse_args()

    trainer = IQNTrainer(args)
    trainer.train()
