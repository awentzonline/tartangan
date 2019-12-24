import argparse
import functools
import os

import torch
from torch import nn
import torch.utils.data as data_utils
import torchvision
from torchvision import transforms
import tqdm

from tartangan.image_dataset import JustImagesDataset
from tartangan.models.losses import discriminator_hinge_loss, generator_hinge_loss
from tartangan.models.progressive import (
    ProgressiveGenerator, ProgressiveDiscriminator, GAN_CONFIGS
)
from .trainer import Trainer


class ProgressiveTrainer(Trainer):
    def build_models(self):
        gan_config = GAN_CONFIGS[self.args.config]
        # generator
        g_optimizer_factory = functools.partial(
            torch.optim.Adam, lr=self.args.lr_g
        )
        self.g = ProgressiveGenerator(
            gan_config, optimizer_factory=g_optimizer_factory
        ).to(self.device)
        # discriminator
        d_optimizer_factory = functools.partial(
            torch.optim.Adam, lr=self.args.lr_d
        )
        self.d = ProgressiveDiscriminator(
            gan_config, optimizer_factory=d_optimizer_factory
        ).to(self.device)
        print(self.g)
        print(self.d)
        self.d_loss = discriminator_hinge_loss
        self.g_loss = generator_hinge_loss
        self.gan_config = gan_config
        # blending
        self.block_blend = 0
        self.blend_steps_remaining = self.args.blend_steps
        self.in_blend_phase = False

    def train(self):
        os.makedirs(os.path.dirname(self.args.sample_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.args.checkpoint), exist_ok=True)
        self.build_models()
        self.update_dataset()
        self.progress_samples = self.sample_z(32)
        steps = 0
        for epoch_i in range(self.args.epochs):
            loader_iter = tqdm.tqdm(self.train_loader)
            for batch_i, images in enumerate(loader_iter):
                metrics = self.train_batch(images)
                loader_iter.set_postfix(**metrics)
                steps += 1
                if steps % self.args.gen_freq == 0:
                    self.output_samples(f'{self.args.sample_file}_{steps}.png')
                if steps % self.args.checkpoint_freq == 0:
                    self.save_checkpoint(f'{self.args.checkpoint}_{steps}')
                if self.update_blend():
                    break

    def update_dataset(self):
        transform = transforms.Compose([
            transforms.Resize((self.g.current_size, self.g.current_size)),
            transforms.ToTensor()
        ])
        self.dataset = JustImagesDataset(self.args.data_path, transform=transform)
        self.train_loader = data_utils.DataLoader(
            self.dataset, batch_size=self.args.batch_size, shuffle=True
        )

    def train_batch(self, imgs):
        # train discriminator
        self.g.eval()
        self.d.train()
        imgs = imgs.to(self.device)
        # train discriminator
        for d_i in range(self.args.iters_d):
            self.d.zero_grad()
            batch_imgs, labels = self.make_adversarial_batch(imgs, blend=self.block_blend)
            p_labels = self.d(batch_imgs, blend=self.block_blend)
            p_real_labels = p_labels[:len(p_labels) // 2]
            p_fake_labels = p_labels[len(p_labels) // 2:]
            d_real_loss, d_fake_loss = self.d_loss(p_real_labels, p_fake_labels)
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            self.d.step_optimizers()
        # train generator
        self.g.zero_grad()
        self.g.train()
        self.d.eval()
        batch_imgs, labels = self.make_generator_batch(imgs, blend=self.block_blend)
        #torchvision.utils.save_image(imgs, 'batch.png')
        p_labels = self.d(batch_imgs, blend=self.block_blend)
        g_loss = self.g_loss(p_labels)
        g_loss.backward()
        # gs = [[p.grad.mean() for p in b.parameters()] for b in self.g.blocks]
        # print(gs)
        self.g.step_optimizers()

        return dict(
            g_loss=float(g_loss), d_loss=float(d_loss),
            blend=self.block_blend,
            real_loss=float(d_real_loss), fake_loss=float(d_fake_loss)
        )

    def output_samples(self, filename, n=None):
        with torch.no_grad():
            imgs = self.g(self.progress_samples, blend=self.block_blend)
            torchvision.utils.save_image(imgs, filename)
            if not hasattr(self, '_latent_grid_samples'):
                self._latent_grid_samples = self.sample_latent_grid(5, 5)
            grid_imgs = self.g(self._latent_grid_samples, blend=self.block_blend)
            torchvision.utils.save_image(
                grid_imgs, os.path.join(
                    os.path.dirname(filename), f'grid_{os.path.basename(filename)}'
                ),
                nrow=5
            )

    def sample_z(self, n=None):
        if n is None:
            n = self.args.batch_size
        return torch.randn(n, self.gan_config.latent_dims).to(self.device)

    def update_blend(self):
        self.blend_steps_remaining -= 1
        # update blend weight
        if self.in_blend_phase:
            self.block_blend = self.blend_steps_remaining / self.args.blend_steps
        else:
            self.block_blend = 0
        # flip states every `self.args.blend_steps`
        if self.blend_steps_remaining <= 0:
            self.in_blend_phase = not self.in_blend_phase
            self.blend_steps_remaining = self.args.blend_steps
            # potentially add a new block
            if self.in_blend_phase:
                print('Adding blocks')
                if not self.g.add_block():
                    return False
                self.d.add_block()
                print(self.g)
                print(self.d)
                self.update_dataset()
                return True
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('data_path')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--gen-freq', type=int, default=200)
    p.add_argument('--latent-dims', type=int, default=50)
    p.add_argument('--lr-g', type=float, default=5e-5)
    p.add_argument('--lr-d', type=float, default=2e-4)
    p.add_argument('--iters-d', type=int, default=2)
    p.add_argument('--device', default='cpu')
    p.add_argument('--epochs', type=int, default=10000)
    p.add_argument('--base-size', type=int, default=4)
    p.add_argument('--base-dims', type=int, default=32)
    p.add_argument('--sample-file', default='sample/tartangan')
    p.add_argument('--blend-steps', type=int, default=100)
    p.add_argument('--config', default='64')
    p.add_argument('--checkpoint-freq', type=int, default=100000)
    p.add_argument('--checkpoint', default='checkpoint/tartangan')
    args = p.parse_args()
    trainer = ProgressiveTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
