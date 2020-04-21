from kubeflow.metadata import metadata

from .base_metadata_app import BaseMetadataApp
from .metadata_mixin import MetadataMixin


class AssignDatasetMetadata(BaseMetadataApp, MetadataMixin):
    def run(self):
        super().run()
        exec = metadata.Execution(
            'assign-dataset-metadata', workspace=self.metadata_workspace
        )
        ds = metadata.DataSet(
            name=self.args.dataset_name,
            uri=self.args.dataset_uri,
            version='0'
        )
        exec.log_output(ds)

    @classmethod
    def add_args_to_parser(cls, p):
        super().add_args_to_parser(p)
        p.add_argument('dataset_name', help='Name of metadata entity')
        p.add_argument('dataset_uri', help='Location of the target dataset')
        p.add_argument('--version', default='0')


def main():
    AssignDatasetMetadata.run_from_cli()


if __name__ == '__main__':
    main()
