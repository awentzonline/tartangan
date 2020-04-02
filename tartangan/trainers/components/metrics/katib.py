import configparser
import os

import smart_open

from tartangan.trainers.components.base import TrainerComponent
from tartangan.utils.fs import maybe_makedirs
from tartangan.utils.kubeflow import key_to_kf_name


class KatibMetricsComponent(TrainerComponent):
    """Output metrics in a format suitable for Katib.
    ```
    ...experiment yaml...
    metricsCollectorSpec:
    collector:
      kind: File
    source:
      filter:
        metricsFormat:
        - "([\\w|-]+) = ((-?\\d+)(\\.\\d+)?)"
      fileSystemPath:
        path: "/path/to/the/metrics.ini"
        kind: File
    ...rest of experiment yaml...
    ```
    """

    def __init__(self, output_path, whitelist=None):
        self.output_path = output_path
        self.whitelist = whitelist

    def on_train_end(self, steps, logs):
        """Outputs the final value for each metric."""
        output = {
            key_to_kf_name(key): float(values[-1])
            for key, values in logs.items()
            if not self.whitelist or key in self.whitelist
        }
        config = configparser.ConfigParser()
        config['metrics'] = output
        dirname = os.path.dirname(self.output_path)
        if dirname:
            maybe_makedirs(dirname, exist_ok=True)
        with smart_open.open(self.output_path, 'w') as outfile:
            config.write(outfile)
