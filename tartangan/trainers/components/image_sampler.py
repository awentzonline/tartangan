import os

import smart_open
import torch
import torchvision

from tartangan.utils.slerp import slerp_grid
from .base import TrainerComponent


class ImageSamplerComponent(TrainerComponent):
    def on_train_begin(self, steps=0):
        os.makedirs(os.path.dirname(self.sample_root + '/'), exist_ok=True)
        self.progress_samples = self.trainer.sample_z(32)

    def on_train_end(self, steps):
        self.output_samples(f'{self.sample_root}/sample_{steps}.png')

    def on_batch_end(self, steps):
        if steps % self.trainer.args.gen_freq == 0:
            self.output_samples(f'{self.sample_root}/sample_{steps}.png')

    def output_samples(self, filename, n=None):
        with torch.no_grad():
            # Render some random samples
            imgs = self.trainer.target_g(self.progress_samples)[:16]
            imgs_g = self.trainer.g(self.progress_samples)[:16]
            imgs = torch.cat([imgs, imgs_g], dim=0)
            with smart_open.open(filename, 'wb') as output_file:
                torchvision.utils.save_image(
                    imgs, output_file, normalize=True, range=(-1, 1), format='png'
                )
            # Render a grid of interpolations
            if not hasattr(self, '_latent_grid_samples'):
                self._latent_grid_samples = self.sample_latent_grid(5, 5)
            grid_imgs = self.trainer.target_g(self._latent_grid_samples)
            grid_filename = os.path.join(
                os.path.dirname(filename), f'grid_{os.path.basename(filename)}'
            )
            with smart_open.open(grid_filename, 'wb') as output_file:
                torchvision.utils.save_image(
                    grid_imgs, output_file, nrow=5,
                    normalize=True, range=(-1, 1), format='png'
                )

    def sample_latent_grid(self, nrows, ncols):
        top_left, top_right, bottom_left, bottom_right = map(
            lambda x: x.cpu(), self.trainer.sample_z(4)
        )
        grid = slerp_grid(top_left, top_right, bottom_left, bottom_right, nrows, ncols)
        grid = grid.to(self.trainer.device)
        return grid

    @property
    def sample_root(self):
        return f'{self.trainer.output_root}/samples'
