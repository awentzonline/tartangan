import argparse
import functools
import hashlib
import os

from PIL import Image
import torch
from torch import nn
import torch.utils.data as data_utils
import torchvision
from torchvision import transforms
import tqdm

from tartangan.image_dataset import JustImagesDataset
from tartangan.models.blocks import (
    ResidualDiscriminatorBlock, ResidualGeneratorBlock, IQNDiscriminatorOutput
)
from tartangan.models.iqn import iqn_loss
from tartangan.models.losses import discriminator_hinge_loss, generator_hinge_loss
from tartangan.models.pluggan import (
    Generator, IQNDiscriminator, GAN_CONFIGS
)
from .trainer import Trainer


class IQNTrainer(Trainer):
    def build_models(self):
        gan_config = GAN_CONFIGS[self.args.config]
        # generator
        g_optimizer_factory = functools.partial(
            torch.optim.Adam, lr=self.args.lr_g
        )
        g_block_factory = functools.partial(
            ResidualGeneratorBlock, #norm_factory=nn.Identity#nn.BatchNorm2d
        )
        self.g = Generator(
            gan_config,
            block_factory=g_block_factory
        ).to(self.device)
        # discriminator
        d_optimizer_factory = functools.partial(
            torch.optim.Adam, lr=self.args.lr_d
        )
        d_block_factory = functools.partial(
            ResidualDiscriminatorBlock, #norm_factory=nn.Identity#nn.BatchNorm2d
        )
        self.d = IQNDiscriminator(
            gan_config,
            output_factory=IQNDiscriminatorOutput
        ).to(self.device)
        print(self.g)
        print(self.d)
        self.g_optimizer = g_optimizer_factory(self.g.parameters())
        self.d_optimizer = d_optimizer_factory(self.d.parameters())
        self.d_loss = iqn_loss
        self.g_loss = iqn_loss
        self.gan_config = gan_config

    # def train(self):
    #     os.makedirs(os.path.dirname(self.args.sample_file), exist_ok=True)
    #     os.makedirs(os.path.dirname(self.args.checkpoint), exist_ok=True)
    #     self.build_models()
    #     self.update_dataset()
    #     self.progress_samples = self.sample_z(32)
    #     steps = 0
    #     for epoch_i in range(self.args.epochs):
    #         loader_iter = tqdm.tqdm(self.train_loader)
    #         for batch_i, images in enumerate(loader_iter):
    #             metrics = self.train_batch(images * 2 - 1)
    #             loader_iter.set_postfix(**metrics)
    #             steps += 1
    #             if steps % self.args.gen_freq == 0:
    #                 self.output_samples(f'{self.args.sample_file}_{steps}.png')
    #             if steps % self.args.checkpoint_freq == 0:
    #                 self.save_checkpoint(f'{self.args.checkpoint}_{steps}')

    def train_batch(self, imgs):
        # train discriminator
        self.g.eval()
        self.d.train()
        imgs = imgs.to(self.device)
        # train discriminator
        for d_i in range(self.args.iters_d):
            self.d.zero_grad()
            batch_imgs, labels = self.make_adversarial_batch(imgs)
            batch_size = len(imgs)
            # _, d_real_loss = self.d(
            #     batch_imgs[:batch_size], targets=labels[:batch_size], blend=self.block_blend
            # )
            # _, d_fake_loss = self.d(
            #     batch_imgs[batch_size:], targets=labels[batch_size:], blend=self.block_blend
            # )
            # d_loss = (d_fake_loss + d_real_loss)# / 2
            p_labels, d_loss = self.d(batch_imgs, targets=labels)
            d_loss.backward()
            self.d_optimizer.step()
        # train generator
        self.g.zero_grad()
        self.g.train()
        self.d.eval()
        batch_imgs, labels = self.make_generator_batch(imgs)
        #torchvision.utils.save_image(imgs, 'batch.png', range=(-1, 1), normalize=True)
        p_labels, g_loss = self.d(batch_imgs, targets=labels)
        g_loss.backward()
        # gs = [[p.grad.mean() for p in b.parameters()] for b in self.g.blocks]
        # print(gs)
        self.g_optimizer.step()

        return dict(
            g_loss=float(g_loss), d_loss=float(d_loss),
        #    real_loss=float(d_real_loss), fake_loss=float(d_fake_loss)
        )

    # def output_samples(self, filename, n=None):
    #     with torch.no_grad():
    #         imgs = self.g(self.progress_samples, blend=self.block_blend)
    #         torchvision.utils.save_image(imgs, filename, range=(-1, 1), normalize=True)
    #         if not hasattr(self, '_latent_grid_samples'):
    #             self._latent_grid_samples = self.sample_latent_grid(5, 5)
    #         grid_imgs = self.g(self._latent_grid_samples, blend=self.block_blend)
    #         torchvision.utils.save_image(
    #             grid_imgs, os.path.join(
    #                 os.path.dirname(filename), f'grid_{os.path.basename(filename)}'
    #             ),
    #             nrow=5, range=(-1, 1), normalize=True
    #         )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('data_path')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--gen-freq', type=int, default=200)
    p.add_argument('--lr-g', type=float, default=1e-3)
    p.add_argument('--lr-d', type=float, default=4e-3)
    p.add_argument('--iters-d', type=int, default=1)
    p.add_argument('--device', default='cpu')
    p.add_argument('--epochs', type=int, default=100000)
    p.add_argument('--sample-file', default='sample/tartangan')
    p.add_argument('--config', default='64')
    p.add_argument('--checkpoint-freq', type=int, default=10000)
    p.add_argument('--checkpoint', default='checkpoint/tartangan')
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--dataset-cache', default='cache/{root}_{size}.pkl')
    args = p.parse_args()

    trainer = IQNTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
