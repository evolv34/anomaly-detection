from .source import Source
import tfx


class ParquetSource(Source):

    def __init__(self, folder_path, label):
        print(f"reading data from path {folder_path}")
        self.folder_path = folder_path
        self.label = label

    def load(self, pipeline: tfx.dsl.Pipeline) -> tfx.dsl.Pipeline:
        return pipeline
