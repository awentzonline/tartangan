import json
import os

import smart_open
import torch

from tartangan.utils.fs import maybe_mkdirs
from .base import TrainerComponent


class ModelCheckpointComponent(TrainerComponent):
    """Saves the models at regular intervals."""

    def on_train_begin(self, steps, logs):
        if self.trainer.args.resume_training_step:
            self.load_checkpoint()

    def on_batch_end(self, steps, logs):
        if steps and steps % self.trainer.args.checkpoint_freq == 0:
            self.save_checkpoint(steps)

    def on_train_end(self, steps, logs):
        self.save_checkpoint(steps)

    def save_checkpoint(self, steps):
        maybe_mkdirs(self.checkpoint_root)
        print(f'saving checkpoint to {self.checkpoint_root}')
        model_filenames = (
            (self.trainer.g, 'g.pt'),
            (self.trainer.target_g, 'g_target.pt'),
            (self.trainer.d, 'd.pt'),
            (self.trainer.optimizer_d, 'opt_d.pt'),
            (self.trainer.optimizer_g, 'opt_g.pt'),
        )
        for model, filename in model_filenames:
            full_filename = f'{self.checkpoint_root}/{filename}'
            with smart_open.open(full_filename, 'wb') as outfile:
                torch.save(model, outfile)
        # trainer state
        state_filename = f'{self.checkpoint_root}/trainer.json'
        with smart_open.open(state_filename, 'w') as outfile:
            state = self.trainer.get_state()
            json.dump(state, outfile)

    def load_checkpoint(self):
        # This must come first
        self.trainer.steps = self.trainer.args.resume_training_step
        print(f'resuming from checkpoint {self.checkpoint_root}')

        model_filenames = (
            ('g', 'g.pt'),
            ('target_g', 'g_target.pt'),
            ('d', 'd.pt'),
            ('optimizer_d', 'opt_d.pt'),
            ('optimizer_g', 'opt_g.pt'),
        )
        for model_name, model_filename in model_filenames:
            full_filename = f'{self.checkpoint_root}/{model_filename}'
            with smart_open.open(full_filename, 'rb') as infile:
                cp_model = torch.load(infile)
                model = getattr(self.trainer, model_name)
                model.load_state_dict(cp_model.state_dict())

        # trainer state
        state_filename = f'{self.checkpoint_root}/trainer.json'
        with smart_open.open(state_filename, 'r') as infile:
            state = json.load(infile)
        self.trainer.set_state(state)

    @property
    def checkpoint_root(self):
        return f'{self.trainer.output_root}/checkpoints/{self.trainer.steps}'
