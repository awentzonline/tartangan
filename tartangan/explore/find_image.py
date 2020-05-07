import os

import numpy as np
from PIL import Image
import smart_open
import torch
from torch import nn, optim
import torchvision.models as models
from torchvision import transforms
import tqdm

from tartangan.utils.fs import maybe_makedirs
from tartangan.trainers.utils import set_device_from_args
from .base import GOutputApp


class FindImage(GOutputApp):
    """Find a generator code for a given image."""
    app_name = "Find image"

    def run(self):
        set_device_from_args(self.args)
        self.load_generator()
        self.setup_feature_extractor()

        self.g = self.g.eval()
        self.g.requires_grad_(False)

        if os.path.dirname(self.args.output_prefix):
            maybe_makedirs(os.path.dirname(self.args.output_prefix))

        target_img = Image.open(self.args.target_image)
        target_img = self.transform_rgb_to_vgg(target_img)
        all_target_feats = []
        with torch.no_grad():
            for feature_extractor in self.feature_extractors:
                feats = feature_extractor(target_img[None, ...])
                all_target_feats.append(
                    feats.repeat(self.args.num_samples, 1, 1, 1)
                )
        # optimize z
        recon_loss = dict(
            mse=nn.MSELoss(), l1=nn.SmoothL1Loss()
        )[self.args.loss]
        z = self.sample_z(self.args.num_samples)
        z.requires_grad_(True)
        target_imgs = target_img.repeat(self.args.num_samples, 1, 1, 1)
        opt_class = {
            'adam': optim.Adam,
            'sgd': optim.SGD,
            'lbfgs': optim.LBFGS,
        }[self.args.optimizer]
        optimizer = opt_class([z], self.args.lr)
        tqdm_iter = tqdm.tqdm(range(self.args.max_steps))

        def train_step():
            optimizer.zero_grad()
            imgs = self.g(z)
            vgg_imgs = torch.stack(list(map(self.transform_tanh_to_vgg, imgs)))
            if self.args.vgg:
                loss = 0.
                for feature_extractor, target_feats in zip(
                        self.feature_extractors, all_target_feats
                ):
                    img_feats = feature_extractor(vgg_imgs)
                    loss = loss + recon_loss(img_feats, target_feats)
            else:
                loss = recon_loss(vgg_imgs, target_imgs)
            # L2 regularization of latent code
            loss += z.pow(2).mean() * self.args.l2
            loss.backward()
            self.save_image(imgs, f'{self.args.output_prefix}_{i}.png')
            return loss

        for i in tqdm_iter:
            # stocastic clipping https://openreview.net/pdf?id=HJC88BzFl
            with torch.no_grad():
                should_clip = (torch.gt(z, 3) + torch.lt(z, -3)).float()
                clip_noise = torch.randn_like(z)
                z -= z * should_clip  # zero out the clipped values
                z += clip_noise * should_clip

            loss = optimizer.step(train_step)
            z_min, z_mean, z_max = float(z.min()), float(z.mean()), float(z.max())
            tqdm_iter.set_postfix(loss=float(loss), z_min=z_min, z_mean=z_mean, z_max=z_max)

    def setup_feature_extractor(self):
        self.feature_extractor = models.vgg16(pretrained=True).to(self.args.device).eval()
        self.feature_extractor.requires_grad_(False)
        self.feature_extractors = []
        for vgg_layer in self.args.vgg_layers:
            feature_extractor = self.feature_extractor.features[:vgg_layer]
            self.feature_extractors.append(feature_extractor)
        img_size = self.g.max_size
        self.transform_rgb_to_vgg = transforms.Compose([
            transforms.Resize(img_size, interpolation=Image.LANCZOS),
            transforms.RandomCrop((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        def tanh_inv(img):
            return (img + 1.) / 2.

        self.transform_tanh_to_vgg = transforms.Compose([
            tanh_inv,
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @classmethod
    def add_args_to_parser(cls, p):
        super().add_args_to_parser(p)
        p.add_argument('target_image', help='Path to image to be found in G')
        p.add_argument('--max-steps', default=100000, type=int)
        p.add_argument('--num-samples', default=2, type=int)
        p.add_argument('--lr', default=0.5, type=float)
        p.add_argument('--vgg', action='store_true')
        p.add_argument('--vgg-layers', default=(9, 16, 23), type=int, nargs='+')
        p.add_argument('--optimizer', default='adam')
        p.add_argument('--l2', default=0., type=float)
        p.add_argument('--loss', default='mse')


if __name__ == '__main__':
    app = FindImage.create_from_cli()
    app.run()
