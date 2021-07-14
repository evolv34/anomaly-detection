from .source import Source
import tfx


class CSVSource(Source):

    def __init__(self, folder_path, label):
        print(f"reading data from path {folder_path}")
        self.folder_path = folder_path
        self.label = label

    def load(self, batch_size=32, num_epochs=None, shuffle=True) -> tfx.dsl.Pipeline:
        return None
