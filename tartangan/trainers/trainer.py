import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import os
import random
import string

import numpy as np
from PIL import Image
import torch
import torch.utils.data as data_utils
from torchvision import transforms
import tqdm

from tartangan.image_bytes_dataset import ImageBytesDataset
from tartangan.image_folder_dataset import ImageFolderDataset
from tartangan.trainers.components.metrics import (
    FIDComponent, KatibMetricsComponent, KubeflowMetricsComponent,
    TensorboardComponent
)
from tartangan.utils.cli import save_cli_arguments, type_or_none
from .components.container import ComponentContainer
from .components.model_checkpoint import ModelCheckpointComponent
from .components.image_sampler import ImageSamplerComponent
from .tqdm_newlines import TqdmNewLines
from .utils import set_device_from_args


class Trainer:
    def __init__(self, args):
        self.args = args

        if args.run_id is None:
            self.run_id = self._generate_run_id()
        else:
            self.run_id = args.run_id

        os.makedirs(self.output_root, exist_ok=True)
        self._save_cli_arguments()

        self.components = ComponentContainer()
        self.components.trainer = self

        self.steps = 0
        self.epoch = 1

    def build_models(self):
        pass

    def setup_components(self):
        self.components.add_components(
            ImageSamplerComponent(), ModelCheckpointComponent(),
        )

        if self.args.inception_moments:
            self.components.add_components(FIDComponent())

        if self.args.metrics_collector:
            metrics_collector_class = {
                'katib': KatibMetricsComponent,
                'kubeflow': KubeflowMetricsComponent,
                'tensorboard': TensorboardComponent,
            }[self.args.metrics_collector]
            metrics_collector = metrics_collector_class(
                self.args.metrics_path
            )
            self.components.add_components(metrics_collector)

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
                transforms.ToPILImage(),  # stored as ndarray which is incompatible w/ RandomCrop
                transforms.RandomCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            dataset = ImageBytesDataset.from_path(
                self.args.data_path, transform=transform
            )
        return dataset

    def train(self):
        self.build_models()
        print(f'Preparing dataset from {self.args.data_path}')
        self.dataset = self.prepare_dataset()
        train_loader = data_utils.DataLoader(
            self.dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=True
        )
        self.setup_components()
        logs = defaultdict(list)
        try:
            self.components.invoke('train_begin', self.steps, logs)
            while self.epoch <= self.args.epochs:
                print(f'Starting epoch {self.epoch}')
                self.components.invoke('epoch_begin', self.steps, self.epoch, logs)
                loader_iter = self.tqdm_class()(train_loader, **self.tqdm_kwargs())
                for batch_i, images in enumerate(loader_iter):
                    self.components.invoke('batch_begin', self.steps, logs)
                    training_metrics = self.train_batch(images)
                    for name, value in training_metrics.items():
                        logs[name].append(value)
                    self.components.invoke('batch_end', self.steps, logs)
                    # update progress bar
                    pretty_training_metrics = {
                        k: round(v, 4) for k, v in training_metrics.items()
                    }
                    loader_iter.set_postfix(refresh=False, **pretty_training_metrics)
                    self.steps += 1

                self.components.invoke('epoch_end', self.steps, self.epoch, logs)
                # TODO: extract dataset cacher to component
                if self.epoch == 1 and self.args.cache_dataset:
                    if hasattr(self.dataset, 'save_cache'):
                        self.dataset.save_cache(self.dataset_cache_path(self.g.max_size))
                self.epoch += 1
        except KeyboardInterrupt:
            pass  # Graceful interrupt
        self.components.invoke('train_end', self.steps, logs)

    def dataset_cache_path(self, size):
        # TODO: extract to component
        root_hash = hashlib.md5(self.dataset.root.encode('utf-8')).hexdigest()
        return self.args.dataset_cache.format(
            root=root_hash,
            size=size
        )

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

    def tqdm_class(self):
        if self.args.log_progress_newlines:
            return TqdmNewLines
        else:
            return tqdm.tqdm

    def tqdm_kwargs(self):
        if not self.args.quiet_logs:
            return {}
        else:
            return dict(
                mininterval=0, maxinterval=np.inf, miniters=self.args.log_iters
            )

    def get_state(self):
        return dict(
            epoch=self.epoch,
            steps=self.steps,
        )

    def set_state(self, state):
        for key, value in state.items():
            setattr(self, key, value)

    def _save_cli_arguments(self):
        save_cli_arguments(f'{self.output_root}/config.args')

    def _generate_run_id(self, suffix_len=6):
        now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        random_suffix = ''.join(random.sample(string.ascii_letters, suffix_len))
        return f'{now}_{random_suffix}'

    @property
    def device(self):
        return self.args.device

    @property
    def output_root(self):
        return f'{self.args.output}/{self.run_id}'

    @classmethod
    def create_from_cli(cls):
        args = cls.parse_cli_args()
        set_device_from_args(args)
        print(f'Using device "{args.device}"')
        return cls(args)

    @classmethod
    def parse_cli_args(cls):
        p = argparse.ArgumentParser(
            description='TartanGAN trainer', fromfile_prefix_chars='@'
        )
        cls.add_args_to_parser(p)
        return p.parse_args()

    @classmethod
    def add_args_to_parser(cls, p):
        p.add_argument('data_path')
        p.add_argument('--batch-size', type=int, default=128)
        p.add_argument('--gen-freq', type=int, default=200,
                       help='Output samples every N batches')
        p.add_argument('--lr-g', type=float, default=1e-4,
                       help='Learning rate for the generator')
        p.add_argument('--lr-d', type=float, default=4e-4,
                       help='Learning rate for the discriminator')
        p.add_argument('--lr-target-g', type=float, default=1e-3,
                       help='Exponential moving average factor for the target generator')
        p.add_argument('--no-cuda', action='store_true')
        p.add_argument('--epochs', type=int, default=10000)
        p.add_argument('--output', default='output',
                       help='Root of output locations. '
                            'A path segment unique to the run will be appended.')
        p.add_argument('--checkpoint-freq', type=int, default=100000,
                       help='Output a checkpoint every N batches')
        p.add_argument('--dataset-cache', default='cache/{root}_{size}.pkl',
                       help='Location of dataset cache when using ImageFolderDataset')
        p.add_argument('--grad-penalty', type=float, default=5.,
                       help='Gradient penalty weight for discriminator on real data')
        p.add_argument('--config', default='64',
                       help='Id of configuration to use. See pluggan.py for details.')
        p.add_argument('--model-scale', type=float, default=1.,
                       help='Multiple the output dimensionality of all layers by this factor')
        p.add_argument('--cache-dataset', action='store_true',
                       help='Enable dataset caching with ImageFolderDataset')
        p.add_argument('--g-base', default='mlp',
                       help='Method generator uses to prepare the inputted latent code')
        p.add_argument('--norm', default='bn',
                       help='Layer normalization method. Either "bn" '
                            '(batchnorm) or "id" (identity)')
        p.add_argument('--activation', default='relu',
                       help='Activation function, either "relu" or "selu"')
        p.add_argument('--quiet-logs', action='store_true', help='Reduce log output')
        p.add_argument('--log-iters', type=int, default=1000,
                       help='Progress logging frequency when --quiet-logs are enabled')
        p.add_argument('--log-progress-newlines', action='store_true',
                       help='Log progress updates one per line')
        p.add_argument('--test-freq', default=10000, type=int,
                       help='Calculate test metrics every N batches')
        p.add_argument('--inception-moments', type=type_or_none(str), default=None,
                       help='Path to pre-calculated inception moments')
        p.add_argument('--n-inception-imgs', default=1000, type=int)
        p.add_argument('--metrics-path', default=None,
                       help='Where to output a file containing run metrics')
        p.add_argument('--metrics-collector', default=None,
                       help='Which metric collector to use (katib, kubeflow, tensorflow)')
        p.add_argument('--resume-training-step', type=type_or_none(int), default=None,
                       help='Resume training from the checkpoint corresponding to this step '
                       'found in the output path specified by the --run-id option.')
        p.add_argument('--run-id', type=type_or_none(str), default=None,
                       help='Explicitly set a run id. Otherwise, one will '
                       'be generated automatically.')
        p.add_argument('--cleanup-inception-model', action='store_true',
                       help='Delete pretrained inception model used for FID metric.')


if __name__ == '__main__':
    trainer = Trainer.create_from_cli()
    trainer.train()
