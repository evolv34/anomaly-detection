# docker run --rm --name=beam_job_server --net=host -e EXECUTOR_MEMORY=4500M -e EXECUTOR_CORES=2 -e EXECUTOR_JAVA_OPTIONS='-XX:+UseG1GC' apache/beam_spark_job_server:latest --spark-master-url=spark://localhost:7077

import os
from typing import Optional, List

import absl
import tfx.v1 as tfx
from ml_metadata.proto import metadata_store_pb2
from tfx.components.base import executor_spec
from tfx.components.example_gen.component import FileBasedExampleGen
from tfx.components.example_gen.custom_executors import parquet_executor
from tfx.proto import example_gen_pb2, transform_pb2

from preprocessors import *

try:
    _parallelism = 3
    # multiprocessing.cpu_count()
except NotImplementedError:
    _parallelism = 1

_beam_pipeline_args_by_runner = {
    'DirectRunner': [
        '--direct_running_mode=in_memory',
        '--direct_num_workers=%d' % _parallelism,
    ],
    'SparkRunner': [
        '--runner=PortableRunner',
        '--job_endpoint=localhost:8099',
        '--environment_type=LOOPBACK',
        '--cache_disabled',
    ]
}


def create_pipeline(
        pipeline_name: Text,
        pipeline_root: Text,
        data_path: Text,
        enable_cache: bool,
        metadata_connection_config: Optional[
            metadata_store_pb2.ConnectionConfig] = None,
        beam_pipeline_args: Optional[List[Text]] = None):
    components = []

    splits_arr = list()
    split_names = list()
    for index, file in enumerate(os.listdir(data_path)):
        if file.endswith(".parquet"):
            split_name = f"source_{index}"
            split_names.append(split_name)
            splits_arr.append(example_gen_pb2.Input.Split(name=split_name,
                                                          pattern=file))

    input_config = example_gen_pb2.Input(splits=[
        example_gen_pb2.Input.Split(name="source", pattern="*.snappy.parquet")
    ])
    # input_config = input,
    # output_config = example_gen_pb2.Output(
    #     split_config=example_gen_pb2.SplitConfig(splits=[
    #         example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=3),
    #         example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1)
    #     ]))
    # output_config = output_config,
    # input_config=input_config,
    example_gen = FileBasedExampleGen(input_base=data_path,
                                      input_config=input_config,
                                      custom_executor_spec=executor_spec.ExecutorClassSpec(parquet_executor.Executor))
    stats_gen = tfx.components.StatisticsGen(examples=example_gen.outputs.examples)
    infer_schema = tfx.components.SchemaGen(statistics=stats_gen.outputs.statistics)

    splits_config = transform_pb2.SplitsConfig(analyze=split_names,
                                               transform=split_names)

    transform_gen = tfx.components.Transform(examples=example_gen.outputs.examples,
                                             schema=infer_schema.outputs.schema,
                                             module_file="preprocessors/drop_columns.py")

    components.append(example_gen)
    components.append(stats_gen)
    components.append(infer_schema)
    components.append(transform_gen)

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=enable_cache,
        metadata_connection_config=metadata_connection_config,
        beam_pipeline_args=beam_pipeline_args,
    )


def run_pipeline(_beam_pipeline_args=None):
    my_pipeline = create_pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=PIPELINE_ROOT,
        data_path=SRC_FOLDER,
        enable_cache=ENABLE_CACHE,
        metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(METADATA_PATH),
        beam_pipeline_args=_beam_pipeline_args
    )

    tfx.orchestration.LocalDagRunner().run(my_pipeline)


def show(path):
    raw_dataset = tf.data.TFRecordDataset(path, compression_type="GZIP")

    for record in raw_dataset.take(10):
        record_numpy = record.numpy()
        example = tf.train.Example()
        example.ParseFromString(record_numpy)
        print(example)


def count(path):
    raw_dataset = tf.data.TFRecordDataset(path, compression_type="GZIP")
    count_val = 0
    for record in raw_dataset:
        count_val += 1

    print(count_val)
    return count_val


tfx.dsl.Pipeline
if __name__ == '__main__':
    absl.logging.set_verbosity(absl.logging.INFO)
    print("Fraud Detection preprocessor pipeline")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # for gpu in tf.config.list_physical_devices("GPU"):
    #     print(f"setting memory for {gpu}")
    #     tf.config.experimental.set_memory_growth(gpu, True)
    #     tf.config.experimental.set_virtual_device_configuration(gpu, [
    #         tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])

    PIPELINE_NAME = 'my_pipeline'
    PIPELINE_ROOT = os.path.join('.', 'resources/fraud_detection_pipeline')
    METADATA_PATH = os.path.join('.', 'resources/tfx_metadata', PIPELINE_NAME, 'metadata.db')
    ENABLE_CACHE = False
    SRC_FOLDER = ""

    # _beam_pipeline_args=_beam_pipeline_args_by_runner['DirectRunner']
    run_pipeline(_beam_pipeline_args=_beam_pipeline_args_by_runner['DirectRunner'])
