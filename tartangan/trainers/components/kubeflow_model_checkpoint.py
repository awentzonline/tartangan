import json

from kubeflow.metadata import metadata
import smart_open

from tartangan.kubeflow.metadata_mixin import MetadataMixin
from .base import TrainerComponent


class KubeflowModelCheckpointComponent(TrainerComponent, MetadataMixin):
    """Saves the models at regular intervals."""
    def on_train_begin(self, steps, logs):
        self._loaded_from = None
        if self.trainer.args.kubeflow_metadata:
            self.load_from_metadata()

    def _setup_kubeflow_metadata(self):
        self.create_metadata_store()
        self.create_metadata_workspace(name='tartangan')  # TODO: param for name

    def load_from_metadata(self):
        # pull model metadata for this workspace/model
        models_md = self.find_metadata_models_by_name(self.model_name)
        if not models_md:
            print('No model metadata found.')
            return

        model_md = models_md[-1]  # grab latest
        # set up trainer state via model.uri
        trainer_state_uri = f"{model_md['uri']}/trainer.json"
        with smart_open.open(trainer_state_uri, 'r') as infile:
            state = json.load(infile)
            self.trainer.set_state(state)
        # load up the models
        self.load_checkpoint()

    def on_train_end(self, steps, logs):
        super().on_train_end(steps, logs)
        self.save_checkpoint_metadata()

    def save_checkpoint_metadata(self):
        exec = metadata.Execution(
            'train', workspace=self.metadata_workspace
        )
        model_md = metadata.Model(
            name=self.model_name,
            uri=self.checkpoint_root,
            version='0'
        )
        exec.log_output(model_md)

    @property
    def model_name(self):
        return self.trainer.run_id
