import argparse
import functools
import hashlib
import os

import numpy as np
from PIL import Image
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
from tartangan.models.losses import (
    discriminator_hinge_loss, generator_hinge_loss, gradient_penalty
)


class Trainer:
    def __init__(self, args):
        self.args = args

    def build_models(self):
        pass

    def train(self):
        os.makedirs(os.path.dirname(self.args.sample_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.args.checkpoint), exist_ok=True)
        self.build_models()
        self.progress_samples = self.sample_z(32)
        img_size = self.g.max_size
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.dataset = dataset = JustImagesDataset(
            self.args.data_path, transform=transform
        )
        if self.args.dataset_cache:
            self.dataset.load_cache(self.dataset_cache_path(img_size))
        train_loader = data_utils.DataLoader(
            dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=True
        )
        steps = 0
        for epoch_i in range(self.args.epochs):
            loader_iter = tqdm.tqdm(train_loader)
            for batch_i, images in enumerate(loader_iter):
                metrics = self.train_batch(images)
                loader_iter.set_postfix(**metrics)
                steps += 1
                if steps % self.args.gen_freq == 0:
                    self.output_samples(f'{self.args.sample_file}_{steps}.png')
                if steps % self.args.checkpoint_freq == 0:
                    self.save_checkpoint(f'{self.args.checkpoint}_{steps}')
            if epoch_i == 0:
                self.dataset.save_cache(self.dataset_cache_path(img_size))

    def dataset_cache_path(self, size):
        root_hash = hashlib.md5(self.dataset.root.encode('utf-8')).hexdigest()
        return self.args.dataset_cache.format(
            root=root_hash,
            size=size
        )

    def train_batch(self, imgs):
        # train discriminator
        self.g.eval()
        self.d.train()
        imgs = imgs.to(self.device)
        # train discriminator
        for d_i in range(self.args.iters_d):
            self.optimizer_d.zero_grad()
            batch_imgs, labels = self.make_adversarial_batch(imgs)
            p_labels = self.d(batch_imgs)
            d_loss = self.d_loss(p_labels, labels)
            d_loss.backward()
            self.optimizer_d.step()
        # train generator
        self.optimizer_g.zero_grad()
        self.g.train()
        self.d.eval()
        batch_imgs, labels = self.make_generator_batch(imgs)
        # torchvision.utils.save_image(batch_imgs, 'batch.png')
        p_labels = self.d(batch_imgs)
        g_loss = self.d_loss(p_labels, labels)
        g_loss.backward()
        self.optimizer_g.step()

        return dict(
            g_loss=float(g_loss), d_loss=float(d_loss)
        )

    def sample_z(self, n=None):
        if n is None:
            n = self.args.batch_size
        return torch.randn(n, self.gan_config.latent_dims).to(self.device)

    def sample_g(self, n=None, target_g=False, **g_kwargs):
        z = self.sample_z(n)
        if target_g:
            imgs = self.target_g(z, **g_kwargs)
        else:
            imgs = self.g(z, **g_kwargs)
        return imgs

    def make_adversarial_batch(self, real_data, **g_kwargs):
        generated_data = self.sample_g(len(real_data), **g_kwargs)
        batch = torch.cat([real_data, generated_data], dim=0)
        labels = torch.zeros(len(batch), 1).to(self.device)
        labels[:len(labels) // 2] = 1  # first half is real
        return batch, labels

    def make_generator_batch(self, real_data, **g_kwargs):
        generated_data = self.sample_g(len(real_data), **g_kwargs)
        labels = torch.ones(len(generated_data), 1).to(self.device)
        return generated_data, labels

    def output_samples(self, filename, n=None):
        with torch.no_grad():
            imgs = self.target_g(self.progress_samples)[:16]
            imgs_g = self.g(self.progress_samples)[:16]
            imgs = torch.cat([imgs, imgs_g], dim=0)
            torchvision.utils.save_image(imgs, filename, normalize=True, range=(-1, 1))
            if not hasattr(self, '_latent_grid_samples'):
                self._latent_grid_samples = self.sample_latent_grid(5, 5)
            grid_imgs = self.target_g(self._latent_grid_samples)
            # grid_imgs_g = self.g(self._latent_grid_samples)
            # grid_imgs = torch.cat([grid_imgs, grid_imgs_g], dim=0)
            torchvision.utils.save_image(
                grid_imgs, os.path.join(
                    os.path.dirname(filename), f'grid_{os.path.basename(filename)}'
                ), nrow=5,
                normalize=True, range=(-1, 1),
            )

    def sample_latent_grid(self, nrows, ncols):
        top_left, top_right, bottom_left, bottom_right = self.sample_z(4)
        left_drow = (bottom_left - top_left) / nrows
        right_drow = (bottom_right - top_right) / nrows
        rows = []
        weights = torch.linspace(0, 1, ncols)[None, ...].T.to(self.device)
        left, right = top_left, top_right
        for row_i in range(nrows):
            row = left + weights * (right - left)
            left += left_drow
            right += right_drow
            rows.append(row)
        grid = torch.cat(rows, dim=0)
        return grid

    def save_checkpoint(self, filename):
        g_filename = f'{filename}_g.pt'
        d_filename = f'{filename}_d.pt'
        torch.save(self.g, g_filename)
        torch.save(self.d, d_filename)

    def load_checkpoint(self, filename):
        g_filename = f'{filename}_g.pt'
        d_filename = f'{filename}_d.pt'
        self.g = torch.load(g_filename)
        self.d = torch.load(d_filename)

    @property
    def device(self):
        return self.args.device


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('data_path')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--gen-freq', type=int, default=200)
    p.add_argument('--img-size', type=int, default=128)
    p.add_argument('--latent-dims', type=int, default=128)
    p.add_argument('--lr-g', type=float, default=1e-4)
    p.add_argument('--lr-d', type=float, default=4e-4)
    p.add_argument('--device', default='cpu')
    p.add_argument('--epochs', type=int, default=10000)
    p.add_argument('--base-size', type=int, default=4)
    p.add_argument('--base-dims', type=int, default=16)
    p.add_argument('--sample-file', default='sample/tartangan')
    p.add_argument('--checkpoint-freq', type=int, default=100000)
    p.add_argument('--checkpoint', default='checkpoint/tartangan')
    p.add_argument('--dataset-cache', default='cache/{root}_{size}.pkl')
    p.add_argument('--grad-penalty', type=float, default=5.)
    args = p.parse_args()

    trainer = CNNTrainer(args)
    trainer.train()
