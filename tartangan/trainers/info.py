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
    GeneratorBlock, DiscriminatorBlock,
    LinearOutput, MultiModelDiscriminatorOutput,
    GaussianParametersOutput,
    ResidualDiscriminatorBlock, ResidualGeneratorBlock,
    GeneratorInputMLP, TiledZGeneratorInput,
    GeneratorOutput, DiscriminatorOutput,
)
from tartangan.models.layers import PixelNorm
from tartangan.models.losses import (
    discriminator_hinge_loss, generator_hinge_loss, gradient_penalty
)
from tartangan.models.pluggan import Discriminator, Generator, GAN_CONFIGS

from .components.info_image_sampler import InfoImageSamplerComponent
from .trainer import Trainer
from .utils import set_device_from_args, toggle_grad


class InfoTrainer(Trainer):
    def build_models(self):
        self.gan_config = GAN_CONFIGS[self.args.config]
        self.gan_config = self.gan_config.scale_model(self.args.model_scale)
        g_norm_factory = {
            'id': nn.Identity,
            'bn': nn.BatchNorm2d,
        }[self.args.norm]
        d_norm_factory = g_norm_factory  # nn.Identity
        g_input_factory = {
            'mlp': GeneratorInputMLP,
            'tiledz': TiledZGeneratorInput,
        }[self.args.g_base]
        activation_factory = {
            'relu': functools.partial(nn.LeakyReLU, 0.2),
            'selu': nn.SELU,
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
            MultiModelDiscriminatorOutput,
            output_model_factories=[
                functools.partial(LinearOutput, out_dims=1),
                functools.partial(
                    LinearOutput,
                    out_dims=self.args.info_cat_dims + self.args.info_cont_dims,
                )
            ],
            norm_factory=d_norm_factory,
            activation_factory=activation_factory
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

        self.d = Discriminator(
            self.gan_config,
            block_factory=d_block_factory,
            output_factory=d_output_factory,
        ).to(self.device)
        self.optimizer_g = torch.optim.Adam(self.g.parameters(), lr=self.args.lr_g, betas=(0., 0.999))
        self.optimizer_d = torch.optim.Adam(self.d.parameters(), lr=self.args.lr_d, betas=(0., 0.999))
        print(self.g)
        print(self.d)
        if self.args.activation == 'selu':
            self.init_params_selu(self.g.parameters())
            self.init_params_selu(self.d.parameters())
        self.update_target_generator(1.)  # copy weights
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

    @classmethod
    def get_component_classes(self, args):
        classes = super().get_component_classes(args)
        classes.append(InfoImageSamplerComponent)
        return classes

    def init_params_selu(self, params):
        for p in params:
            d = p.data
            if len(d.shape) == 1:
                d.zero_()
                #d.normal_(std=1e-8)
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
        batch_imgs, labels, z = self.make_adversarial_batch(imgs)
        real, fake = batch_imgs[:self.args.batch_size], batch_imgs[self.args.batch_size:]
        if self.args.grad_penalty:
            real.requires_grad_()
        p_labels_real, _ = self.d(real)
        p_labels_fake, p_codes = self.d(fake.detach())
        p_labels = torch.cat([p_labels_real, p_labels_fake], dim=0)
        d_loss = self.bce_loss(p_labels, labels)
        # infogan loss
        d_code_loss = 0
        if self.args.info_cat_dims:
            z_cat_code = self.z_categorical_code(z)
            p_z_cat_code = self.z_categorical_code(p_codes)
            d_cat_code_loss = self.bce_loss(p_z_cat_code, z_cat_code)
            d_code_loss += d_cat_code_loss
        if self.args.info_cont_dims:
            z_cont_code = self.z_continuous_code(z)
            p_z_cont_code = self.z_continuous_code(p_codes)
            d_cont_code_loss = self.mse_loss(p_z_cont_code, z_cont_code)
            d_code_loss += d_cont_code_loss
        d_loss += self.args.info_w * d_code_loss

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
        batch_imgs, labels, z = self.make_generator_batch(imgs)
        p_labels, p_codes = self.d(batch_imgs)
        g_loss = self.bce_loss(p_labels, labels)
        # infogan loss
        g_code_loss = 0.
        if self.args.info_cat_dims:
            z_cat_code = self.z_categorical_code(z)
            p_z_cat_code = self.z_categorical_code(p_codes)
            g_cat_code_loss = self.bce_loss(p_z_cat_code, z_cat_code)
            g_code_loss += g_cat_code_loss
        if self.args.info_cont_dims:
            z_cont_code = self.z_continuous_code(z)
            p_z_cont_code = self.z_continuous_code(p_codes)
            g_cont_code_loss = self.mse_loss(p_z_cont_code, z_cont_code)
            g_code_loss += g_cont_code_loss

        g_loss += self.args.info_w * g_code_loss

        g_loss.backward()
        self.optimizer_g.step()

        self.update_target_generator()

        return dict(
            g_loss=float(g_loss), g_code_loss=float(g_code_loss),
            d_loss=float(d_loss), d_code_loss=float(d_code_loss),
            gp=float(d_grad_penalty)
        )

    def log_likelihood_gaussian(self, x, mu, log_sigma, eps=1e-8):
        z = (x - mu) / (torch.exp(log_sigma) + eps)
        result = -0.5 * np.log(2. * np.pi) - log_sigma - 0.5 * z ** 2
        return result.sum(1)

    def z_categorical_code(self, z):
        return z[..., :self.args.info_cat_dims]

    def z_continuous_code(self, z):
        return z[..., self.args.info_cat_dims:self.args.info_cat_dims + self.args.info_cont_dims]

    def sample_z(self, n=None):
        if n is None:
            n = self.args.batch_size
        z = torch.randn(n, self.gan_config.latent_dims).to(self.device)
        # set up the categorical dimensions
        if self.args.info_cat_dims:
            z[..., :self.args.info_cat_dims] = 0.
            cats = np.random.randint(0, self.args.info_cat_dims, (n,))
            z[np.arange(n), ..., cats] = 1.
        return z

    def sample_g(self, n=None, target_g=False, **g_kwargs):
        z = self.sample_z(n)
        if target_g:
            imgs = self.target_g(z, **g_kwargs)
        else:
            imgs = self.g(z, **g_kwargs)
        return imgs, z

    def make_adversarial_batch(self, real_data, **g_kwargs):
        generated_data, z = self.sample_g(len(real_data), **g_kwargs)
        batch = torch.cat([real_data, generated_data], dim=0)
        labels = torch.zeros(len(batch), 1).to(self.device)
        labels[:len(labels) // 2] = 1  # first half is real
        return batch, labels, z

    def make_generator_batch(self, real_data, **g_kwargs):
        generated_data, z = self.sample_g(len(real_data), **g_kwargs)
        labels = torch.ones(len(generated_data), 1).to(self.device)
        return generated_data, labels, z

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
        p.add_argument('--info-cat-dims', type=int, default=10)
        p.add_argument('--info-cont-dims', type=int, default=5)
        p.add_argument('--info-w', type=float, default=1.)


def main():
    trainer = InfoTrainer.create_from_cli()
    trainer.train()


if __name__ == '__main__':
    main()
