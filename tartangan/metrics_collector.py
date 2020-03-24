import abc
import configparser
import json
import os

import smart_open


class MetricsCollector(abc.ABC):

    @abc.abstractmethod
    def add_scalar(self, key, value):
        raise NotImplementedError

    def flush(self):
        pass


class KubeflowMetricsCollector(MetricsCollector):
    """Output metrics in a format suitable for Kubeflow."""

    def __init__(self, output_path):
        self.values = {}
        self.output_path = output_path

    def add_scalar(self, key, value):
        self.values[key] = value

    def flush(self):
        output = dict(
            metrics=[
                dict(name=key_to_kf_metric_name(key), numberValue=float(value))
                for key, value in self.values.items()
            ]
        )
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with smart_open.open(self.output_path, 'w') as outfile:
            json.dump(output, outfile)


class KatibMetricsCollector(MetricsCollector):
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

    def __init__(self, output_path):
        self.values = {}
        self.output_path = output_path

    def add_scalar(self, key, value):
        self.values[key] = value

    def flush(self):
        output = {
            key_to_kf_metric_name(key): float(value)
            for key, value in self.values.items()
        }
        config = configparser.ConfigParser()
        config['metrics'] = output
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with smart_open.open(self.output_path, 'w') as outfile:
            config.write(outfile)


def key_to_kf_metric_name(k):
    """Convert a name to something Kubeflow likes."""
    k = k.replace('_', '-').lower()
    return k
