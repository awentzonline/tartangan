try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print('Tensorboard not available.')

from tartangan.trainers.components.base import TrainerComponent


class TensorboardComponent(TrainerComponent):
    def __init__(self, output_path=None, whitelist=None):
        self.output_path = output_path
        self.whitelist = whitelist

    def on_train_begin(self, steps, logs):
        if self.output_path:
            self.output_path = f'{self.output_path}/{self.trainer.run_id}'
        self.summary_writer = SummaryWriter(self.output_path)

    def on_batch_end(self, steps, logs):
        scalars = {
            key: float(values[-1]) for key, values in logs.items()
            if not self.whitelist or key in self.whitelist
        }
        self.summary_writer.add_scalars(self.main_tag, scalars, steps)

    @property
    def main_tag(self):
        return self.trainer.__class__.__name__
