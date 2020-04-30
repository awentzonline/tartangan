import json
import os

import smart_open

from tartangan.utils.fs import maybe_makedirs
from tartangan.utils.kubeflow import key_to_kf_name
from .base import FileBasedMetricsComponent


class KubeflowMetricsComponent(FileBasedMetricsComponent):
    """Output metrics in a format suitable for Kubeflow."""
    whitelist = None

    def on_train_end(self, steps, logs):
        """Outputs the final value for each metric."""
        output = dict(
            metrics=[
                dict(name=key_to_kf_name(key), numberValue=float(values[-1]))
                for key, values in logs.items()
                if not self.whitelist or key in self.whitelist
            ]
        )
        dirname = os.path.dirname(self.args.metrics_path)
        if dirname:
            maybe_makedirs(dirname, exist_ok=True)
        with smart_open.open(self.args.metrics_path, 'w') as outfile:
            json.dump(output, outfile)
