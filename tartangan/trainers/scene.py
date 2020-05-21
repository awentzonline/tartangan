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
    SceneBlock, SceneInput, GeneratorOutput, SceneStructureBlock,
    ResidualGeneratorBlock,
    DiscriminatorBlock, ResidualDiscriminatorBlock, DiscriminatorOutput
)
from tartangan.models.pluggan import StructuredSceneGenerator, Discriminator, GAN_CONFIGS
from tartangan.models.losses import (
    discriminator_hinge_loss, generator_hinge_loss, gradient_penalty
)

from .trainer import Trainer
from .utils import set_device_from_args, toggle_grad


class SceneTrainer(Trainer):
    def build_models(self):
        self.gan_config = GAN_CONFIGS[self.args.config]
        self.gan_config = self.gan_config.scale_model(self.args.model_scale)
        g_norm_factory = {
            'id': nn.Identity,
            'bn': nn.BatchNorm2d,
        }[self.args.norm]
        d_norm_factory = g_norm_factory  # nn.Identity
        d_norm_factory = {
            'id': nn.Identity,
            'bn': nn.BatchNorm2d,
        }[self.args.norm]
        g_input_factory = functools.partial(
            SceneStructureBlock, scene_size=self.args.scene_size,
            patch_size=self.args.patch_size, num_patches=self.args.num_patches,
            refine_patches=self.args.refine_patches,
            patch_noise=self.args.patch_noise,
        )
        activation_factory = {
            'relu': functools.partial(nn.LeakyReLU, 0.2),
            'selu': nn.SELU,
            'elu': nn.ELU,
        }[self.args.activation]

        g_input_factory = functools.partial(
            g_input_factory, activation_factory=activation_factory
        )
        g_block_factory = functools.partial(
            ResidualGeneratorBlock, norm_factory=g_norm_factory,
            activation_factory=activation_factory
        )
        d_block_factory = functools.partial(
            ResidualDiscriminatorBlock, norm_factory=d_norm_factory,
            activation_factory=activation_factory
        )
        g_output_factory = functools.partial(
            GeneratorOutput, norm_factory=g_norm_factory,
            activation_factory=activation_factory
        )
        d_output_factory = functools.partial(
            DiscriminatorOutput, norm_factory=d_norm_factory,
            activation_factory=activation_factory
        )
        self.g = StructuredSceneGenerator(
            self.gan_config,
            input_factory=g_input_factory,
            block_factory=g_block_factory,
            output_factory=g_output_factory,
        ).to(self.device)
        self.target_g = StructuredSceneGenerator(
            self.gan_config,
            input_factory=g_input_factory,
            block_factory=g_block_factory,
            output_factory=g_output_factory,
        ).to(self.device)

        self.d = Discriminator(
            self.gan_config,
            block_factory=d_block_factory,
            output_factory=d_output_factory,
        ).to(self.device)
        self.optimizer_g = torch.optim.Adam(self.g.parameters(), lr=self.args.lr_g, betas=(0., 0.999))
        self.optimizer_d = torch.optim.Adam(self.d.parameters(), lr=self.args.lr_d, betas=(0., 0.999))
        self.d_loss = discriminator_hinge_loss
        self.g_loss = generator_hinge_loss
        self.bce_loss = nn.BCEWithLogitsLoss()
        # self.bce_loss = nn.BCELoss()
        if self.args.activation == 'selu':
            self.init_params_selu(self.g.parameters())
            self.init_params_selu(self.d.parameters())
        self.update_target_generator(1.)  # copy weights

    def init_params_selu(self, params):
        for p in params:
            d = p.data
            if len(d.shape) == 1:
                d.zero_()
                # d.normal_(std=1e-8)
            else:
                in_dims, _ = nn.init._calculate_fan_in_and_fan_out(d)
                d.normal_(std=np.sqrt(1. / in_dims))

    def train_batch(self, imgs):
        #print(imgs.min(), imgs.mean(), imgs.max())
        imgs = imgs.to(self.device)
        self.g.train()
        self.d.train()
        # train discriminator
        toggle_grad(self.g, False)
        toggle_grad(self.d, True)
        self.optimizer_d.zero_grad()
        batch_imgs, labels = self.make_adversarial_batch(imgs)
        real, fake = batch_imgs[:self.dist_batch_size], batch_imgs[self.dist_batch_size:]
        # torchvision.utils.save_image(real, 'batch_real.png', normalize=True, range=(-1, 1))
        # torchvision.utils.save_image(fake, 'batch_fake.png', normalize=True, range=(-1, 1))
        if self.args.grad_penalty:
            real.requires_grad_()
        p_labels_real = self.d(real)
        p_labels_fake = self.d(fake.detach())
        p_labels = torch.cat([p_labels_real, p_labels_fake], dim=0)
        # loss_real, loss_fake = self.d_loss(labels, p_labels)
        # d_loss = loss_real + loss_fake
        # d_loss = (
        #     self.bce_loss(p_labels_real, labels[:len(labels) // 2]) +
        #     self.bce_loss(p_labels_fake, labels[len(labels) // 2:])
        # )
        d_loss = self.bce_loss(p_labels, labels)
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
        p_labels = self.d(batch_imgs)
        #g_loss = self.g_loss(p_labels)
        g_loss = self.bce_loss(p_labels, labels)
        g_loss.backward()
        self.optimizer_g.step()

        self.update_target_generator()

        return dict(
            g_loss=float(g_loss), d_loss=float(d_loss),
            gp=float(d_grad_penalty)
        )
    #
    # def make_generator_batch(self, real_data, **g_kwargs):
    #     z, generated_data = self.sample_g(len(real_data), return_z_final=True, **g_kwargs)
    #     labels = torch.ones(len(generated_data), 1).to(self.device)
    #     return generated_data, labels, z
    #
    @torch.no_grad()
    def update_target_generator(self, lr=None):
        if lr is None:
            lr = self.args.lr_target_g
        for g_p, target_g_p in zip(self.g.parameters(), self.target_g.parameters()):
            target_g_p.add_(
                (g_p - target_g_p) * self.args.lr_target_g
            )

    @classmethod
    def add_args_to_parser(cls, p):
        super().add_args_to_parser(p)
        p.add_argument('--scene-size', type=int, default=16)
        p.add_argument('--patch-size', type=int, default=3)
        p.add_argument('--num-patches', type=int, default=20)
        p.add_argument('--refine-patches', action='store_true')
        p.add_argument('--patch-noise', action='store_true')

def main():
    trainer = SceneTrainer.create_from_cli()
    trainer.train()


if __name__ == '__main__':
    main()
