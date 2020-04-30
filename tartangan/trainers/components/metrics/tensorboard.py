try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print('Tensorboard not available.')

from .base import FileBasedMetricsComponent


class TensorboardComponent(FileBasedMetricsComponent):
    whitelist = None

    def on_train_begin(self, steps, logs):
        metrics_path = f'{self.args.metrics_path}/{self.trainer.run_id}'
        self.summary_writer = SummaryWriter(metrics_path)

    def on_batch_end(self, steps, logs):
        scalars = {
            key: float(values[-1]) for key, values in logs.items()
            if not self.whitelist or key in self.whitelist
        }
        self.summary_writer.add_scalars(self.main_tag, scalars, steps)

    @property
    def main_tag(self):
        return self.trainer.__class__.__name__
