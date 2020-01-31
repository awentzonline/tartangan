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

from tartangan.image_dataset import JustImagesDataset
from tartangan.models.blocks import (
    ResidualDiscriminatorBlock, ResidualGeneratorBlock,
    GeneratorInputMLP, TiledZGeneratorInput
)
from tartangan.models.pluggan import Discriminator, Generator, GAN_CONFIGS
from tartangan.models.losses import (
    discriminator_hinge_loss, generator_hinge_loss, gradient_penalty
)

from .trainer import Trainer


class CNNTrainer(Trainer):
    def build_models(self):
        self.gan_config = GAN_CONFIGS[self.args.config]
        g_block_factory = functools.partial(
            ResidualGeneratorBlock, #norm_factory=nn.Identity#nn.BatchNorm2d
        )
        d_block_factory = functools.partial(
            ResidualDiscriminatorBlock, #norm_factory=nn.Identity#nn.BatchNorm2d
        )
        self.g = Generator(
            self.gan_config,
            input_factory=TiledZGeneratorInput,
            block_factory=g_block_factory
        ).to(self.device)
        self.target_g = Generator(
            self.gan_config,
            input_factory=TiledZGeneratorInput,
            block_factory=g_block_factory
        ).to(self.device)
        self.d = Discriminator(
            self.gan_config,
            block_factory=d_block_factory
        ).to(self.device)
        self.optimizer_g = torch.optim.Adam(self.g.parameters(), lr=self.args.lr_g)
        self.optimizer_d = torch.optim.Adam(self.d.parameters(), lr=self.args.lr_d)
        self.d_loss = discriminator_hinge_loss
        self.g_loss = generator_hinge_loss
        self.bce_loss = nn.BCELoss()
        print(self.g)
        print(self.d)

    def train_batch(self, imgs):
        imgs = (imgs * 2) - 1
        # train discriminator
        imgs = imgs.to(self.device)
        # train discriminator
        self.g.eval()
        self.d.train()
        self.optimizer_d.zero_grad()
        batch_imgs, labels = self.make_adversarial_batch(imgs)
        real, fake = batch_imgs[:self.args.batch_size], batch_imgs[self.args.batch_size:]
        p_labels_real = self.d(real)
        p_labels_fake = self.d(fake.detach())
        p_labels = torch.cat([p_labels_real, p_labels_fake], dim=0)
        loss_real, loss_fake = self.d_loss(labels, p_labels)
        d_loss = loss_real + loss_fake
        #d_loss = self.bce_loss(p_labels, labels)
        d_grad_penalty = 0#self.args.grad_penalty * gradient_penalty(p_labels, real)
        d_loss += d_grad_penalty
        d_loss.backward()
        self.optimizer_d.step()

        # train generator
        self.g.train()
        self.d.eval()
        self.optimizer_g.zero_grad()
        batch_imgs, labels = self.make_generator_batch(imgs)
        # torchvision.utils.save_image((imgs + 1) / 2, 'batch.png')
        p_labels = self.d(batch_imgs)
        g_loss = self.g_loss(p_labels)
        #g_loss = self.bce_loss(p_labels, labels)
        g_loss.backward()
        self.optimizer_g.step()

        self.update_target_generator()

        return dict(
            g_loss=float(g_loss), d_loss=float(d_loss),
            gp=float(d_grad_penalty)
        )

    @torch.no_grad()
    def update_target_generator(self):
        for g_p, target_g_p in zip(self.g.parameters(), self.target_g.parameters()):
            target_g_p.add_(
                (g_p - target_g_p) * self.args.lr_target_g
            )


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('data_path')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--gen-freq', type=int, default=200)
    p.add_argument('--lr-g', type=float, default=1e-3)
    p.add_argument('--lr-d', type=float, default=4e-3)
    p.add_argument('--lr-target-g', type=float, default=1e-2)
    p.add_argument('--device', default='cpu')
    p.add_argument('--epochs', type=int, default=10000)
    p.add_argument('--sample-file', default='sample/tartangan')
    p.add_argument('--checkpoint-freq', type=int, default=100000)
    p.add_argument('--checkpoint', default='checkpoint/tartangan')
    p.add_argument('--dataset-cache', default='cache/{root}_{size}.pkl')
    p.add_argument('--grad-penalty', type=float, default=5.)
    p.add_argument('--config', default='64')
    args = p.parse_args()

    trainer = CNNTrainer(args)
    trainer.train()
