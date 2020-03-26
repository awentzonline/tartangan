import json
import os

import smart_open

from tartangan.trainers.components.base import TrainerComponent
from tartangan.utils.kubeflow import key_to_kf_name


class KubeflowMetricsComponent(TrainerComponent):
    """Output metrics in a format suitable for Kubeflow."""

    def __init__(self, output_path, whitelist=None):
        self.output_path = output_path
        self.whitelist = whitelist

    def on_train_end(self, steps, logs):
        """Outputs the final value for each metric."""
        output = dict(
            metrics=[
                dict(name=key_to_kf_name(key), numberValue=float(values[-1]))
                for key, values in logs.items()
                if not self.whitelist or key in self.whitelist
            ]
        )
        dirname = os.path.dirname(self.output_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with smart_open.open(self.output_path, 'w') as outfile:
            json.dump(output, outfile)
