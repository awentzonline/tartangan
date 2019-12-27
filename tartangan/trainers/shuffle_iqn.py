import argparse
import functools
import os

import numpy as np
import torch
from torch import nn
import torch.utils.data as data_utils
import torchvision
import tqdm

from tartangan.image_dataset import JustImagesDataset
from tartangan.models.blocks import (
    ResidualDiscriminatorBlock, ResidualGeneratorBlock, IQNDiscriminatorOutput
)
from tartangan.models.iqn import iqn_loss
from tartangan.models.losses import discriminator_hinge_loss, generator_hinge_loss
from tartangan.models.progressive import (
    ProgressiveGenerator, ProgressiveIQNDiscriminator, GAN_CONFIGS
)
from .iqn import ProgressiveIQNTrainer
from .trainer import Trainer


class ProgressiveIQNShuffledTrainer(ProgressiveIQNTrainer):
    def build_models(self):
        gan_config = GAN_CONFIGS[self.args.config]
        # generator
        g_optimizer_factory = functools.partial(
            torch.optim.Adam, lr=self.args.lr_g
        )
        g_block_factory = functools.partial(
            ResidualGeneratorBlock, norm_factory=nn.Identity#nn.BatchNorm2d
        )
        self.g = ProgressiveGenerator(
            gan_config, optimizer_factory=g_optimizer_factory, block_class=g_block_factory
        ).to(self.device)
        # discriminator
        d_optimizer_factory = functools.partial(
            torch.optim.Adam, lr=self.args.lr_d
        )
        d_block_factory = functools.partial(
            ResidualDiscriminatorBlock, norm_factory=nn.Identity#nn.BatchNorm2d
        )
        self.d = ProgressiveIQNDiscriminator(
            gan_config, optimizer_factory=d_optimizer_factory,
            output_class=IQNDiscriminatorOutput,
            block_class=d_block_factory,
            output_channels=2
        ).to(self.device)
        print(self.g)
        print(self.d)
        self.d_loss = iqn_loss
        self.g_loss = iqn_loss
        self.gan_config = gan_config
        # blending
        self.block_blend = 0
        self.blend_steps_remaining = self.args.blend_steps
        self.in_blend_phase = False

    def train_batch(self, imgs):
        # train discriminator
        self.g.eval()
        self.d.train()
        imgs = imgs.to(self.device)
        # train discriminator
        for d_i in range(self.args.iters_d):
            self.d.zero_grad()
            batch_imgs, labels = self.make_adversarial_batch(imgs, blend=self.block_blend)
            p_labels, d_loss = self.d(batch_imgs, targets=labels, blend=self.block_blend)
            d_loss.backward()
            self.d.step_optimizers()
        # train generator
        self.g.zero_grad()
        self.g.train()
        self.d.eval()
        batch_imgs, labels = self.make_generator_batch(imgs, blend=self.block_blend)
        #torchvision.utils.save_image(imgs, 'batch.png', range=(-1, 1), normalize=True)
        p_labels, g_loss = self.d(batch_imgs, targets=labels, blend=self.block_blend)
        g_loss.backward()
        # gs = [[p.grad.mean() for p in b.parameters()] for b in self.g.blocks]
        # print(gs)
        self.g.step_optimizers()

        return dict(
            g_loss=float(g_loss), d_loss=float(d_loss),
            blend=self.block_blend,
        #    real_loss=float(d_real_loss), fake_loss=float(d_fake_loss)
        )

    def shuffle_images(self, imgs, num_swaps=2, min_size=0.2, max_size=0.3):
        """Swap chunks of these imgs around."""
        num_imgs, c, h, w = imgs.shape
        for i in range(num_swaps):
            # choose size of swap windows
            for j in range(num_imgs):
                self.shuffle_img(
                    imgs[j], num_swaps=num_swaps, min_size=min_size,
                    max_size=max_size
                )
        return imgs

    def shuffle_img(self, img, num_swaps=2, min_size=0.2, max_size=0.3):
        c, h, w = img.shape
        ratio = np.random.uniform(min_size, max_size)
        width = np.clip(ratio * w, 1, None).astype(np.int64)
        height = np.clip(ratio * h, 1, None).astype(np.int64)
        x0, x1 = np.random.uniform(0, w - width, 2).astype(np.int64)
        y0, y1 = np.random.uniform(0, h - height, 2).astype(np.int64)
        a0 = img[:, y0:y0 + height, x0:x0 + width]
        a1 = img[:, y1:y1 + height, x0:x0 + width]
        img[:, y0:y0 + height, x0:x0 + width] = a1
        img[:, y1:y1 + height, x1:x1 + width] = a0

    def make_adversarial_batch(self, real_data, **g_kwargs):
        generated_data = self.sample_g(len(real_data), **g_kwargs)
        batch = torch.cat([real_data, generated_data], dim=0)
        labels = torch.zeros(len(batch), 2).to(self.device)
        labels[:len(labels) // 2, 0] = 1  # first half is real
        # shuffle
        p_shuffle_img = 0.1
        is_shuffled = torch.from_numpy(np.random.binomial(1, p_shuffle_img, len(batch)))
        for i, shuffle in enumerate(is_shuffled):
            if shuffle:
                self.shuffle_img(batch[i])
        labels[np.arange(len(labels)), 1] = is_shuffled.float().to(self.device)
        return batch, labels

    def make_generator_batch(self, real_data, **g_kwargs):
        generated_data = self.sample_g(len(real_data) * 2, **g_kwargs)
        labels = torch.ones(len(generated_data), 2).to(self.device)
        labels[:, 1] = 0
        return generated_data, labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument('data_path')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--gen-freq', type=int, default=200)
    p.add_argument('--latent-dims', type=int, default=50)
    p.add_argument('--lr-g', type=float, default=1e-3)
    p.add_argument('--lr-d', type=float, default=4e-3)
    p.add_argument('--iters-d', type=int, default=1)
    p.add_argument('--device', default='cpu')
    p.add_argument('--epochs', type=int, default=100000)
    p.add_argument('--base-size', type=int, default=4)
    p.add_argument('--base-dims', type=int, default=32)
    p.add_argument('--sample-file', default='sample/tartangan')
    p.add_argument('--blend-steps', type=int, default=100)
    p.add_argument('--config', default='64')
    p.add_argument('--checkpoint-freq', type=int, default=10000)
    p.add_argument('--checkpoint', default='checkpoint/tartangan')
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--dataset-cache', default='cache/cache_{size}.pkl')
    args = p.parse_args()

    trainer = ProgressiveIQNShuffledTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
