import os
import shutil
import tempfile

from tartangan.trainers.components.base import TrainerComponent
from tartangan import inception_utils


class FIDComponent(TrainerComponent):
    """Calculate Frechet Inception Distance"""

    def on_train_begin(self, steps, logs):
        if self.trainer.args.cleanup_inception_model:
            self.model_path = tempfile.mkdtemp()
            os.environ['TORCH_HOME'] = self.model_path
            print(f'Setting $TORCH_HOME to {self.model_path}')

        print('Preparing inception metrics...')
        self.get_inception_metrics = inception_utils.prepare_inception_metrics(
            self.trainer.args.inception_moments, self.trainer.device, False
        )

    def on_train_end(self, steps, logs):
        if self.trainer.args.cleanup_inception_model:
            print(f'Cleaning up $TORCH_HOME = {self.model_path}')
            shutil.rmtree(self.model_path)

    def on_batch_end(self, steps, logs):
        """Calculate inception metrics"""
        if steps and steps % self.trainer.args.test_freq == 0:
            print('Calculating inception metrics...')
            is_mean, is_std, fid = self._calculate()
            logs['fid'].append(fid)
            logs['inception_score_mean'].append(is_mean)
            logs['inception_score_std'].append(is_std)

    def _calculate(self):
        is_mean, is_std, fid = self.get_inception_metrics(
            self.trainer.sample_g, self.trainer.args.n_inception_imgs,
            num_splits=5
        )
        print('Inception Score is %3.3f +/- %3.3f' % (is_mean, is_std))
        print('FID is %5.4f' % (fid,))
        return is_mean, is_std, fid
