import argparse
import functools
import hashlib
import os

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.utils.data as data_utils
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print('Tensorboard not available.')
import torchvision
from torchvision import transforms
import tqdm

from tartangan.image_bytes_dataset import ImageBytesDataset
from tartangan.image_folder_dataset import ImageFolderDataset
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
        self.summary_writer = None
        if self.args.tensorboard:
            self.summary_writer = SummaryWriter()

    def build_models(self):
        pass

    def prepare_dataset(self):
        img_size = self.g.max_size
        if os.path.isdir(self.args.data_path):
            # use the old "JustImagesDataset"
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size), interpolation=Image.LANCZOS),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            dataset = ImageFolderDataset(
                self.args.data_path, transform=transform
            )
            if self.args.dataset_cache:
                dataset.load_cache(self.dataset_cache_path(img_size))
        else:
            # load up a tensor dataset
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            dataset = ImageBytesDataset.from_path(
                self.args.data_path, transform=transform
            )
        return dataset

    def train(self):
        os.makedirs(os.path.dirname(self.args.sample_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.args.checkpoint), exist_ok=True)
        self.build_models()
        self.progress_samples = self.sample_z(32)
        self.dataset = self.prepare_dataset()
        train_loader = data_utils.DataLoader(
            self.dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=True
        )
        steps = 0
        for epoch_i in range(self.args.epochs):
            loader_iter = tqdm.tqdm(train_loader)
            for batch_i, images in enumerate(loader_iter):
                metrics = self.train_batch(images)
                self.log_metrics(metrics, steps)
                round_metrics = {k: round(v, 4) for k, v in metrics.items()}
                loader_iter.set_postfix(**round_metrics)
                steps += 1
                if steps % self.args.gen_freq == 0:
                    self.output_samples(f'{self.args.sample_file}_{steps}.png')
                if steps % self.args.checkpoint_freq == 0:
                    self.save_checkpoint(f'{self.args.checkpoint}_{steps}')
            if epoch_i == 0 and self.args.cache_dataset:
                if hasattr(self.dataset, 'save_cache'):
                    self.dataset.save_cache(self.dataset_cache_path(img_size))

    def dataset_cache_path(self, size):
        root_hash = hashlib.md5(self.dataset.root.encode('utf-8')).hexdigest()
        return self.args.dataset_cache.format(
            root=root_hash,
            size=size
        )

    def log_metrics(self, metrics, i=None):
        if not self.summary_writer:
            return
        self.summary_writer.add_scalars('training', metrics, i)

    def train_batch(self, imgs):
        # train discriminator
        self.g.train()
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
        top_left, top_right, bottom_left, bottom_right = map(
            lambda x: x.cpu(), self.sample_z(4)
        )
        left_col = [
            slerp(x, top_left, bottom_left) for x in np.linspace(0, 1, nrows)
        ]
        right_col = [
            slerp(x, top_right, bottom_right) for x in np.linspace(0, 1, nrows)
        ]
        rows = []
        for left, right in zip(left_col, right_col):
            row = [
                slerp(x, left, right) for x in np.linspace(0, 1, ncols)
            ]
            rows.append(torch.from_numpy(np.vstack(row)))
        grid = torch.cat(rows, dim=0).to(self.device)
        return grid

    def save_checkpoint(self, filename):
        g_filename = f'{filename}_g.pt'
        g_target_filename = f'{filename}_g_target.pt'
        d_filename = f'{filename}_d.pt'
        torch.save(self.g, g_filename)
        torch.save(self.target_g, g_target_filename)
        torch.save(self.d, d_filename)

    def load_checkpoint(self, filename):
        g_filename = f'{filename}_g.pt'
        g_target_filename = f'{filename}_g_target.pt'
        d_filename = f'{filename}_d.pt'
        self.g = torch.load(g_filename)
        self.target_g = torch.load(g_target_filename)
        self.d = torch.load(d_filename)

    @property
    def device(self):
        return self.args.device


def slerp(val, low, high):
    """
    https://github.com/soumith/dcgan.torch/issues/14#issuecomment-200025792
    """
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high


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
    p.add_argument('--checkpoint', default='checkpoints/tartangan')
    p.add_argument('--dataset-cache', default='cache/{root}_{size}.pkl')
    p.add_argument('--grad-penalty', type=float, default=5.)
    p.add_argument('--model-scale', type=float, default=1.)
    p.add_argument('--cache-dataset', action='store_true')
    p.add_argument('--log-dir', default='runs')
    p.add_argument('--tensorboard', action='store_true')
    p.add_argument('--g-base', default='mlp', help='mlp or tiledz')
    p.add_argument('--norm', default='bn', help='bn or id')
    args = p.parse_args()

    trainer = CNNTrainer(args)
    trainer.train()
