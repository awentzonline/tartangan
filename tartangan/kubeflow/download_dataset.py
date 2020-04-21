import os

from kubeflow.metadata import metadata
import smart_open

from .base_metadata_app import BaseMetadataApp


class DownloadDatasetMetadata(BaseMetadataApp):
    def run(self):
        super().run()
        datasets = self.find_metadata_datasets_by_name(self.args.dataset_name)
        # TODO: need to sort to get latest?
        dataset = datasets[-1]
        with smart_open.open(dataset['uri'], 'rb') as infile:
            with smart_open.open(self.args.output_path, 'wb') as outfile:
                outfile.write(infile.read())

    @classmethod
    def add_args_to_parser(cls, p):
        super().add_args_to_parser(p)
        p.add_argument('dataset_name', help='Name of metadata entity')
        p.add_argument('output_path', help='Where the corresponding files go')


def main():
    DownloadDatasetMetadata.run_from_cli()


if __name__ == '__main__':
    main()
