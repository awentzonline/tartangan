import json
import os

import smart_open
import torch

from tartangan.utils.fs import maybe_makedirs, smart_ls
from .base import TrainerComponent


class ModelCheckpointComponent(TrainerComponent):
    """Saves the models at regular intervals."""

    def on_train_begin(self, steps, logs):
        self._loaded_from = None
        if self.trainer.args.resume_training_step:
            self.trainer.steps = self.trainer.args.resume_training_step
            self.load_checkpoint()
        elif self.trainer.args.resume_training_latest:
            self.resume_training_from_latest()

    def on_batch_end(self, steps, logs):
        if steps and steps % self.trainer.args.checkpoint_freq == 0:
            # Prevent immediate re-saving of checkpoint
            if self._loaded_from != steps:
                self.save_checkpoint(steps)

    def on_train_end(self, steps, logs):
        self.save_checkpoint(steps)

    def save_checkpoint(self, steps):
        maybe_makedirs(self.checkpoint_root)
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
        print(f'resuming from checkpoint {self.checkpoint_root}')
        self._loaded_from = self.trainer.steps

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

    def resume_training_from_latest(self):
        latest_id = self.latest_checkpoint_id()
        if latest_id is not None:
            self.trainer.steps = latest_id
            self.load_checkpoint()
        else:
            print('No checkpoints found to resume.')

    def latest_checkpoint_id(self):
        """Find the latest checkpoint id in the checkpoints directory."""
        subdirs = smart_ls(self.all_checkpoints_root)
        int_dirs = []
        for key in subdirs:
            try:
                key = int(key)
            except ValueError:
                pass
            else:
                int_dirs.append(key)

        if not int_dirs:
            return None
        latest = list(sorted(int_dirs))[-1]
        return latest

    @property
    def checkpoint_root(self):
        return f'{self.all_checkpoints_root}/{self.trainer.steps}'

    @property
    def all_checkpoints_root(self):
        return f'{self.trainer.output_root}/checkpoints'
