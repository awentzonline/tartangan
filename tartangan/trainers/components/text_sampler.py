import os
import textwrap

import smart_open
import torch
import torchvision

from tartangan.utils.fs import maybe_makedirs
from tartangan.utils.slerp import slerp_grid
from .base import TrainerComponent


class TextSamplerComponent(TrainerComponent):
    def on_train_begin(self, steps, logs):
        maybe_makedirs(os.path.dirname(self.sample_root + '/'), exist_ok=True)
        self.progress_samples = self.trainer.sample_z(32)

    def on_train_end(self, steps, logs):
        self.output_samples(f'{self.sample_root}/sample_{steps}.txt')

    def on_batch_end(self, steps, logs):
        if steps % self.trainer.args.gen_freq == 0:
            self.output_samples(f'{self.sample_root}/sample_{steps}.txt')

    def output_samples(self, filename, n=None):
        with torch.no_grad():
            # Render some random samples
            generated = self.trainer.g(self.progress_samples)[:16]
            generated = self.trainer.embedding.lookup(generated)
            v = self.trainer.dataset.vocab
            with smart_open.open(filename, 'w') as outfile:
                for result in generated:
                    decoded = list(map(v.itos.__getitem__, result))
                    doc = ' '.join(decoded)
                    outfile.writelines([
                        s + '\n' for s in textwrap.wrap(doc, 70)
                    ])
                    outfile.write('-' * 40 + '\n')

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
