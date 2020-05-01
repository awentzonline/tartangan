import os

import numpy as np
import smart_open
import torch
import torchvision

from tartangan.utils.fs import maybe_makedirs
from tartangan.utils.slerp import slerp_grid
from .image_sampler import ImageSamplerComponent


class InfoImageSamplerComponent(ImageSamplerComponent):
    def on_train_begin(self, steps, logs):
        super().on_train_begin(steps, logs)

        self.num_cont_dims = min(4, self.trainer.args.info_cont_dims)
        self.num_points_per_dim = 7
        base_z = self.trainer.sample_z(1)

        self.continuous_samples = (
            base_z[None, ...].repeat(self.num_points_per_dim, self.num_cont_dims + 1, 1)
        )
        # sweep from -1 to 1 for the continuous-valued infogan vars
        pts = torch.linspace(
            -2, 2, self.num_points_per_dim,
        )
        for i in range(self.num_cont_dims):
            self.continuous_samples[:, i, self.trainer.args.info_cat_dims + i] = pts
        # include row of a non-controlled dim
        self.continuous_samples[:, -1, -1] = pts

        if self.trainer.args.info_cat_dims:
            # render an image for each category
            num_cat_samples = 3  # including the base_z used with continuous vars
            base_z = torch.cat([base_z, self.trainer.sample_z(num_cat_samples - 1)], dim=0)
            self.categorical_samples = (
                base_z[:, None, ...].repeat(1, self.trainer.args.info_cat_dims, 1)
            )
            cat_sweep = torch.eye(self.trainer.args.info_cat_dims)
            self.categorical_samples[..., :self.trainer.args.info_cat_dims] = cat_sweep
        else:
            self.categorical_samples = None

    def output_samples(self, filename, n=None):
        with torch.no_grad():
            for name, samples in (
                ('cat', self.categorical_samples), ('cont', self.continuous_samples)
            ):
                if samples is None:
                    continue
                grid_imgs = self.trainer.target_g(samples)
                grid_filename = os.path.join(
                    os.path.dirname(filename), f'info_{name}_{os.path.basename(filename)}'
                )
                nrow = samples.shape[1]
                with smart_open.open(grid_filename, 'wb') as output_file:
                    torchvision.utils.save_image(
                        grid_imgs, output_file, nrow=nrow,
                        normalize=True, range=(-1, 1), format='png'
                    )
