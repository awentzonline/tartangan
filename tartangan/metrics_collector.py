import abc
from collections import defaultdict
import json

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
        with smart_open.open(self.output_path, 'w') as outfile:
            json.dump(output, outfile)


def key_to_kf_metric_name(k):
    """Convert a name to something Kubeflow likes."""
    k = k.replace('_', '-').lower()
    return k
