import os

import smart_open
import torch

from .base import TrainerComponent


class ModelCheckpointComponent(TrainerComponent):
    """Saves the models at regular intervals."""

    def on_train_begin(self, steps=0):
        os.makedirs(
            os.path.dirname(self.checkpoint_root + '/'), exist_ok=True)

        if self.trainer.args.resume_training_id:
            self.load_checkpoint('TODO')

    def on_batch_end(self, steps):
        if steps and steps % self.trainer.args.checkpoint_freq == 0:
            self.save_checkpoint(steps)

    def on_train_end(self, steps):
        self.save_checkpoint(steps)

    def save_checkpoint(self, steps):
        filename = f'{self.checkpoint_root}/checkpoint_{steps}'
        print(f'saving checkpoint to {filename}')
        model_filenames = (
            (self.trainer.g, f'{filename}_g.pt'),
            (self.trainer.target_g, f'{filename}_g_target.pt'),
            (self.trainer.d, f'{filename}_d.pt')
        )
        for model, filename in model_filenames:
            with smart_open.open(filename, 'wb') as outfile:
                torch.save(model, outfile)

    def load_checkpoint(self, filename):
        print('loading checkpoint...')
        # TODO: fix --resume-training
        raise NotImplementedError('Need to re-add arguments to support resuming training.')

        model_filenames = (
            ('g', f'{filename}_g.pt'),
            ('target_g', f'{filename}_g_target.pt'),
            ('d', f'{filename}_d.pt')
        )
        for model_name, model_filename in model_filenames:
            with smart_open.open(model_filename, 'rb') as infile:
                model = torch.load(infile)
                setattr(self.trainer, model_name, model)

    @property
    def checkpoint_root(self):
        return f'{self.trainer.output_root}/checkpoints'
