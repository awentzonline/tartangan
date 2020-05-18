import argparse
import functools
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import torchvision
from torchvision import transforms
import tqdm

from tartangan.models.blocks import (
    DiscriminatorInput, DiscriminatorBlock,
    GeneratorBlock,
    GeneratorInputMLP1d, TiledZGeneratorInput,
    GeneratorOutput, DiscriminatorOutput,
    ResidualDiscriminatorBlock, ResidualGeneratorBlock,
)
from tartangan.models.losses import (
    discriminator_hinge_loss, generator_hinge_loss, gradient_penalty
)
from tartangan.models.pluggan import Discriminator, Generator, GAN_CONFIGS
from tartangan.models.text import SkipGram
from tartangan.text_dataset import TextDataset
from .components.model_checkpoint import ModelCheckpointComponent
from .components.text_sampler import TextSamplerComponent
from .trainer import Trainer
from .utils import set_device_from_args, toggle_grad


class TextCNNTrainer(Trainer):
    def build_models(self):
        self.gan_config = GAN_CONFIGS[self.args.config]
        self.gan_config = self.gan_config.scale_model(self.args.model_scale)
        self.gan_config = self.gan_config._replace(
            data_dims=self.args.embedding_dims
        )
        g_norm_factory = {
            'id': nn.Identity,
            'bn': nn.BatchNorm1d,
        }[self.args.norm]
        d_norm_factory = g_norm_factory  # nn.Identity
        g_input_factory = {
            'mlp': GeneratorInputMLP1d,
        }[self.args.g_base]
        activation_factory = {
            'relu': functools.partial(nn.LeakyReLU, 0.2),
            'selu': nn.SELU,
            'elu': nn.ELU,
        }[self.args.activation]

        g_input_factory = functools.partial(
            g_input_factory, activation_factory=activation_factory
        )
        d_input_factory = functools.partial(
            DiscriminatorInput, activation_factory=activation_factory,
            conv_factory=nn.Conv1d,
        )
        g_block_factory = functools.partial(
            ResidualGeneratorBlock, norm_factory=g_norm_factory,
            activation_factory=activation_factory,
            conv_factory=nn.Conv1d,
        )
        d_block_factory = functools.partial(
            ResidualDiscriminatorBlock, norm_factory=d_norm_factory,
            activation_factory=activation_factory,
            conv_factory=nn.Conv1d,
            avg_pool_factory=nn.AvgPool1d,
            interpolate=functools.partial(
                F.interpolate, scale_factor=0.5, mode='linear', align_corners=False
            )
        )
        g_output_factory = functools.partial(
            GeneratorOutput, norm_factory=g_norm_factory,
            activation_factory=activation_factory,
            conv_factory=nn.Conv1d,
            output_activation_factory=nn.Identity,
        )
        d_output_factory = functools.partial(
            DiscriminatorOutput, norm_factory=d_norm_factory,
            activation_factory=activation_factory
        )
        self.g = Generator(
            self.gan_config,
            input_factory=g_input_factory,
            block_factory=g_block_factory,
            output_factory=g_output_factory,
        ).to(self.device)
        self.target_g = Generator(
            self.gan_config,
            input_factory=g_input_factory,
            block_factory=g_block_factory,
            output_factory=g_output_factory,
        ).to(self.device)

        self.d = Discriminator(
            self.gan_config,
            input_factory=d_input_factory,
            block_factory=d_block_factory,
            output_factory=d_output_factory,
        ).to(self.device)
        self.optimizer_g = torch.optim.Adam(self.g.parameters(), lr=self.args.lr_g, betas=(0., 0.999))
        self.optimizer_d = torch.optim.Adam(self.d.parameters(), lr=self.args.lr_d, betas=(0., 0.999))
        self.d_loss = discriminator_hinge_loss
        self.g_loss = generator_hinge_loss
        self.bce_loss = nn.BCEWithLogitsLoss()
        # self.bce_loss = nn.BCELoss()
        print(self.g)
        print(self.d)
        if self.args.activation == 'selu':
            self.init_params_selu(self.g.parameters())
            self.init_params_selu(self.d.parameters())
        self.update_target_generator(1.)  # copy weights
        self.pretraining_embedding = self.args.pretrain_embedding

    def init_params_selu(self, params):
        for p in params:
            d = p.data
            if len(d.shape) == 1:
                d.zero_()
                # d.normal_(std=1e-8)
            else:
                in_dims, _ = nn.init._calculate_fan_in_and_fan_out(d)
                d.normal_(std=np.sqrt(1. / in_dims))

    def setup_components(self):
        self.components.add_components(
            ModelCheckpointComponent(), TextSamplerComponent(),
        )

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
        max_doc_size = self.g.max_size
        self.dataset = TextDataset.from_path(
            self.args.data_path, doc_len=max_doc_size
        )
        self.embedding = SkipGram(
            len(self.dataset.vocab), self.args.embedding_dims, sparse=True,
            padding_idx=self.dataset.vocab.stoi['<PAD>']
        )
        self.embedding = self.embedding.to(self.device)
        self.optimizer_embedding = torch.optim.SGD(
            self.embedding.parameters(), lr=self.args.lr_d,
        )
        return self.dataset

    def train_batch(self, input_indexes):
        input_indexes = input_indexes.long().to(self.device)

        # train embedding
        self.embedding.train()
        toggle_grad(self.embedding, True)
        self.optimizer_embedding.zero_grad()
        # extract random windows from the input docs
        window_size = self.args.context * 2 + 1
        offsets = np.random.randint(0, window_size, len(input_indexes))
        windows = torch.stack([
            input_indexes[i, ..., offset:offset + window_size]
            for i, offset in enumerate(offsets)
        ])
        # get a list of pivot words and their contexts from the windows
        words = windows[..., self.args.context]
        contexts = torch.cat([
            windows[..., :self.args.context],
            windows[..., self.args.context + 1:]
        ], dim=-1)
        # get the loss
        embedding_loss = self.embedding.loss(words, contexts)
        embedding_loss.backward()
        self.optimizer_embedding.step()

        self.pretraining_embedding = max(self.pretraining_embedding - 1, 0)
        if not self.pretraining_embedding:
            inputs = self.embedding(input_indexes).permute((0, 2, 1)).detach()
            self.g.train()
            self.d.train()
            # train discriminator
            toggle_grad(self.embedding, False)
            toggle_grad(self.g, False)
            toggle_grad(self.d, True)
            self.optimizer_d.zero_grad()
            batch_inputs, labels = self.make_adversarial_batch(inputs)
            real, fake = batch_inputs[:self.args.batch_size], batch_inputs[self.args.batch_size:]
            if self.args.grad_penalty:
                real.requires_grad_()
            p_labels_real = self.d(real)
            p_labels_fake = self.d(fake.detach())
            p_labels = torch.cat([p_labels_real, p_labels_fake], dim=0)
            d_loss = self.bce_loss(p_labels, labels)
            d_grad_penalty = 0.
            if self.args.grad_penalty:
                d_grad_penalty = self.args.grad_penalty * gradient_penalty(p_labels_real, real)
                d_loss += d_grad_penalty
            d_loss.backward()
            self.optimizer_d.step()

            # train generator
            toggle_grad(self.g, True)
            toggle_grad(self.d, False)
            self.optimizer_g.zero_grad()
            batch_inputs, labels = self.make_generator_batch(inputs)
            p_labels = self.d(batch_inputs)
            g_loss = self.bce_loss(p_labels, labels)
            g_loss.backward()
            self.optimizer_g.step()
        else:
            g_loss = d_loss = d_grad_penalty = 0.

        self.update_target_generator()

        return dict(
            g_loss=float(g_loss), d_loss=float(d_loss),
            gp=float(d_grad_penalty),
            embedding_loss=float(embedding_loss)
        )

    def make_adversarial_batch(self, real_data, **g_kwargs):
        generated_data = self.sample_g(len(real_data), **g_kwargs)
        batch = torch.cat([real_data, generated_data], dim=0)
        labels = torch.zeros(len(batch), 1).to(self.device)
        labels[:len(labels) // 2] = 1  # first half is real
        return batch, labels

    @torch.no_grad()
    def update_target_generator(self, lr=None):
        if lr is None:
            lr = self.args.lr_target_g
        for g_p, target_g_p in zip(self.g.parameters(), self.target_g.parameters()):
            target_g_p.add_(
                (g_p - target_g_p) * self.args.lr_target_g
            )

    @classmethod
    def add_args_to_parser(cls, p):
        super().add_args_to_parser(p)
        p.add_argument('--embedding-dims', type=int, default=64)
        p.add_argument('--context', type=int, default=3)
        p.add_argument('--pretrain-embedding', type=int, default=10000)


def main():
    trainer = TextCNNTrainer.create_from_cli()
    trainer.train()


if __name__ == '__main__':
    main()
