import os

from kubeflow.metadata import metadata


class MetadataMixin:
    def create_metadata_store(self):
        self.metadata_store = metadata.Store(
            grpc_host=os.getenv('METADATA_STORE_HOST', 'metadata-grpc-service.kubeflow'),
            grpc_port=int(os.getenv('METADATA_STORE_PORT', '8080'))
        )
        return self.metadata_store

    def create_metadata_workspace(self, name):
        self.metadata_workspace = metadata.Workspace(
            store=self.metadata_store,
            name=name
        )
        return self.metadata_workspace

    def find_metadata_datasets_by_name(self, name):
        return self.find_metadata_artifacts_by_name(metadata.DataSet, name)

    def find_metadata_models_by_name(self, name):
        return self.find_metadata_artifacts_by_name(metadata.Model, name)

    def find_metadata_artifacts_by_name(self, artifact_class, name):
        objs = self.metadata_workspace.list(artifact_class.ARTIFACT_TYPE_NAME)
        matches = [
            obj for obj in objs
            if obj['name'] == name
        ]
        return matches
