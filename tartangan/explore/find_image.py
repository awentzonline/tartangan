import os

import numpy as np
from PIL import Image
import smart_open
import torch
from torch import nn, optim
import torchvision
import torchvision.models as models
from torchvision import transforms
import tqdm

from tartangan.utils.fs import maybe_makedirs
from tartangan.utils.slerp import slerp
from tartangan.trainers.utils import set_device_from_args
from .base import GOutputApp


class FindImage(GOutputApp):
    """Find a generator code for a given image."""
    app_name = "Find image"

    def run(self):
        set_device_from_args(self.args)
        self.load_generator()
        self.g.requires_grad_(False)
        self.g.eval()
        self.setup_feature_extractor()

        if os.path.dirname(self.args.output_prefix):
            maybe_makedirs(os.path.dirname(self.args.output_prefix))

        target_img = Image.open(self.args.target_image)
        target_img = self.transform_rgb_to_vgg(target_img)
        all_target_feats = []
        with torch.no_grad():
            for feature_extractor in self.feature_extractors:
                all_target_feats.append(
                    feature_extractor(target_img[None, ...])
                )
        # optimize z
        mse_loss = nn.MSELoss()
        z = self.sample_z(self.args.num_samples)
        z.requires_grad_(True)
        opt_class = {
            'adam': optim.Adam,
            'sgd': optim.SGD,
        }[self.args.optimizer]
        optimizer = opt_class([z], self.args.lr)
        tqdm_iter = tqdm.tqdm(range(self.args.max_steps))
        self.g.train()
        self.g.requires_grad_(False)
        for i in tqdm_iter:
            optimizer.zero_grad()
            imgs = self.g(z)
            vgg_imgs = torch.stack(list(map(self.transform_tanh_to_vgg, imgs)))
            loss = 0.
            for feature_extractor, target_feats in zip(
                    self.feature_extractors, all_target_feats
            ):
                img_feats = feature_extractor(vgg_imgs)
                loss = loss + mse_loss(img_feats, target_feats)
            loss += z.pow(2).mean()
            loss.backward()
            optimizer.step()
            z_min, z_mean, z_max = float(z.min()), float(z.mean()), float(z.max())
            tqdm_iter.set_postfix(loss=float(loss), z_min=z_min, z_mean=z_mean, z_max=z_max)
            self.save_image(imgs, f'{self.args.output_prefix}_{i}.png')

    def setup_feature_extractor(self):
        self.feature_extractor = models.vgg16(pretrained=True).to(self.args.device)
        self.feature_extractor.requires_grad_(False)
        #self.feature_extractor.eval()
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

        def untanh(img):
            return (img + 1.) / 2.

        self.transform_tanh_to_vgg = transforms.Compose([
            untanh,
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @classmethod
    def add_args_to_parser(cls, p):
        super().add_args_to_parser(p)
        p.add_argument('target_image', help='Path to image to be found in G')
        p.add_argument('--max-steps', default=100000, type=int)
        p.add_argument('--num-samples', default=1, type=int)
        p.add_argument('--lr', default=0.001, type=float)
        p.add_argument('--vgg-layers', default=(9, 16, 23), type=int, nargs='+')
        p.add_argument('--optimizer', default='sgd')


if __name__ == '__main__':
    app = FindImage.create_from_cli()
    app.run()
