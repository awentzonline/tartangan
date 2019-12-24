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
from tartangan.models.cnn import GeneratorCNN, DiscriminatorCNN


class Trainer:
    def __init__(self, args):
        self.args = args

    def build_models(self):
        pass

    def train(self):
        os.makedirs(os.path.dirname(self.args.sample_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.args.checkpoint), exist_ok=True)
        self.progress_samples = self.sample_z(32)
        self.build_models()
        transform = transforms.Compose([
            transforms.Resize((self.args.img_size, self.args.img_size)),
            transforms.ToTensor()
        ])
        dataset = JustImagesDataset(self.args.data_path, transform=transform)
        train_loader = data_utils.DataLoader(
            dataset, batch_size=self.args.batch_size, shuffle=True
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
        #torchvision.utils.save_image(batch_imgs, 'batch.png')
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
        return torch.randn(n, self.args.latent_dims).to(self.device)

    def sample_g(self, n=None, **g_kwargs):
        z = self.sample_z(n)
        imgs = self.g(z, **g_kwargs)
        return imgs

    def make_adversarial_batch(self, real_data, **g_kwargs):
        generated_data = self.sample_g(len(real_data), **g_kwargs)
        batch = torch.cat([real_data, generated_data], dim=0)
        labels = torch.zeros(len(batch), 1).to(self.device)
        labels[:len(labels) // 2] = 1  # first half is real
        return batch, labels

    def make_generator_batch(self, real_data, **g_kwargs):
        generated_data = self.sample_g(len(real_data) * 2, **g_kwargs)
        labels = torch.ones(len(generated_data), 1).to(self.device)
        return generated_data, labels

    def output_samples(self, filename, n=None):
        with torch.no_grad():
            imgs = self.g(self.progress_samples)
            torchvision.utils.save_image(imgs, filename)
            if not hasattr(self, '_latent_grid_samples'):
                self._latent_grid_samples = self.sample_latent_grid(5, 5)
            grid_imgs = self.g(self._latent_grid_samples)
            torchvision.utils.save_image(
                grid_imgs, os.path.join(
                    os.path.dirname(filename), f'grid_{os.path.basename(filename)}'
                ),
                nrow=5
            )

    def sample_latent_grid(self, nrows, ncols):
        a0, a1, b0, b1 = self.sample_z(4)
        left, right = a0, a1
        left_dz, right_dz = b0 - left, b1 - right
        rows = []
        weights = torch.linspace(0, 1, ncols)[None, ...].T.to(self.device)
        for row_i in range(nrows):
            row = left + weights * (right - left)
            left += left_dz
            right += right_dz
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


class CNNTrainer(Trainer):
    def build_models(self):
        depth = int(np.log2(self.args.img_size) - np.log2(self.args.base_size))
        print(f'depth = {depth}')
        self.g = GeneratorCNN(
            self.args.latent_dims, self.args.img_size,
            depth=depth, base_dims=self.args.base_dims
        ).to(self.device)
        self.d = DiscriminatorCNN(
            self.args.img_size,
            depth=depth, base_dims=self.args.base_dims
        ).to(self.device)
        self.optimizer_g = torch.optim.Adam(self.g.parameters(), lr=self.args.lr_g)
        self.optimizer_d = torch.optim.Adam(self.d.parameters(), lr=self.args.lr_d)
        self.d_loss = nn.BCELoss()
        print(self.g)
        print(self.d)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('data_path')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--gen-freq', type=int, default=200)
    p.add_argument('--img-size', type=int, default=128)
    p.add_argument('--latent-dims', type=int, default=50)
    p.add_argument('--lr-g', type=float, default=0.003)
    p.add_argument('--lr-d', type=float, default=0.003)
    p.add_argument('--iters-d', type=int, default=2)
    p.add_argument('--device', default='cpu')
    p.add_argument('--epochs', type=int, default=10000)
    p.add_argument('--base-size', type=int, default=4)
    p.add_argument('--base-dims', type=int, default=32)
    p.add_argument('--sample-file', default='sample/tartangan')
    p.add_argument('--checkpoint-freq', type=int, default=100000)
    p.add_argument('--checkpoint', default='checkpoint/tartangan')
    args = p.parse_args()

    trainer = CNNTrainer(args)
    trainer.train()
