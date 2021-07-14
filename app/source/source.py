import tfx


class Source(object):
    def load(self, pipeline: tfx.dsl.Pipeline) -> tfx.dsl.Pipeline:
        pass
