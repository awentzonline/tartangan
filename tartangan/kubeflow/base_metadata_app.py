import os

from kubeflow.metadata import metadata

from tartangan.utils.app import App


class BaseMetadataApp(App):
    def run(self):
        self.create_metadata_store()
        self.create_workspace()

    def create_metadata_store(self):
        self.store = metadata.Store(
            grpc_host=os.getenv('METADATA_STORE_HOST', 'metadata-grpc-service.kubeflow'),
            grpc_port=int(os.getenv('METADATA_STORE_PORT', '8080'))
        )
        return self.store

    def create_workspace(self):
        self.workspace = metadata.Workspace(
            store=self.store,
            name=self.args.workspace
        )

    @classmethod
    def add_args_to_parser(cls, p):
        super().add_args_to_parser(p)
        p.add_argument('--workspace', default='debug')
