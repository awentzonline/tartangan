from tartangan.utils.app import App
from .metadata_mixin import MetadataMixin


class BaseMetadataApp(App, MetadataMixin):
    def run(self):
        self.create_metadata_store()
        self.create_metadata_workspace(self.args.workspace)

    @classmethod
    def add_args_to_parser(cls, p):
        super().add_args_to_parser(p)
        p.add_argument('--workspace', default='debug')
