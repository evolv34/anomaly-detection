# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Penguin example using TFX."""

import multiprocessing
import os
import socket
import sys
from typing import List, Optional, Text
import absl
from absl import flags

import tensorflow_model_analysis as tfma
from tfx import v1 as tfx

flags.DEFINE_enum(
    'runner', 'DirectRunner', ['DirectRunner', 'FlinkRunner', 'SparkRunner'],
    'The Beam runner to execute Beam-powered components. '
    'For FlinkRunner or SparkRunner, first run setup/setup_beam_on_flink.sh '
    'or setup/setup_beam_on_spark.sh, respectively.')

flags.DEFINE_enum(
    'model_framework', 'keras', ['keras', 'flax_experimental'],
    'The modeling framework.')

# This example assumes that penguin data is stored in ~/penguin/data and the
# utility function is in ~/penguin. Feel free to customize as needed.
_penguin_root = 'penguin'
_data_root = os.path.join(_penguin_root, 'data')

# Directory and data locations.  This example assumes all of the
# example code and metadata library is relative to $HOME, but you can store
# these files anywhere on your local filesystem.
_tfx_root = os.path.join(os.environ['HOME'], 'tfx')

# Pipeline arguments for Beam powered Components.
# LINT.IfChange
try:
    _parallelism = multiprocessing.cpu_count()
except NotImplementedError:
    _parallelism = 1
# LINT.ThenChange(setup/setup_beam_on_flink.sh)

# Common pipeline arguments used by both Flink and Spark runners.
_beam_portable_pipeline_args = [
    # The runner will instruct the original Python process to start Beam Python
    # workers.
    '--environment_type=LOOPBACK',
    # Start Beam Python workers as separate processes as opposed to threads.
    '--experiments=use_loopback_process_worker=True',
    '--sdk_worker_parallelism=%d' % _parallelism,

    # Setting environment_cache_millis to practically infinity enables
    # continual reuse of Beam SDK workers, improving performance.
    '--environment_cache_millis=1000000',

    # TODO(b/183057237): Obviate setting this.
    '--experiments=pre_optimize=all',
]

# Pipeline arguments for Beam powered Components.
# Arguments differ according to runner.
_beam_pipeline_args_by_runner = {
    'DirectRunner': [
        '--direct_running_mode=multi_processing',
        # 0 means auto-detect based on on the number of CPUs available
        # during execution time.
        '--direct_num_workers=0',
    ],
    'SparkRunner': [
                       '--runner=SparkRunner',
                       '--spark_submit_uber_jar',
                       '--spark_rest_url=http://%s:6066' % socket.gethostname(),
                   ] + _beam_portable_pipeline_args,
    'FlinkRunner': [
                       '--runner=FlinkRunner',
                       # LINT.IfChange
                       '--flink_version=1.12',
                       # LINT.ThenChange(setup/setup_beam_on_flink.sh)
                       '--flink_submit_uber_jar',
                       '--flink_master=http://localhost:8081',
                       '--parallelism=%d' % _parallelism,
                   ] + _beam_portable_pipeline_args
}

# Configs for ExampleGen and SpansResolver, e.g.,
#
# This will match the <input_base>/day3/* as ExampleGen's input and generate
# Examples artifact with Span equals to 3.
#   examplegen_input_config = tfx.proto.Input(splits=[
#       tfx.proto.Input.Split(name='input', pattern='day{SPAN}/*'),
#   ])
#   examplegen_range_config = tfx.proto.RangeConfig(
#       static_range=tfx.proto.StaticRange(
#           start_span_number=3, end_span_number=3))
#
# This will get the latest 2 Spans (Examples artifacts) from MLMD for training.
#   resolver_range_config = tfx.proto.RangeConfig(
#       rolling_range=tfx.proto.RollingRange(num_spans=2))
_examplegen_input_config = None
_examplegen_range_config = None
_resolver_range_config = None


def _create_pipeline(
        pipeline_name: Text,
        pipeline_root: Text,
        data_root: Text,
        module_file: Text,
        accuracy_threshold: float,
        serving_model_dir: Text,
        metadata_path: Text,
        user_provided_schema_path: Optional[Text],
        enable_tuning: bool,
        enable_bulk_inferrer: bool,
        examplegen_input_config: Optional[tfx.proto.Input],
        examplegen_range_config: Optional[tfx.proto.RangeConfig],
        resolver_range_config: Optional[tfx.proto.RangeConfig],
        beam_pipeline_args: List[Text],
) -> tfx.dsl.Pipeline:
    """Implements the penguin pipeline with TFX.

    Args:
      pipeline_name: name of the TFX pipeline being created.
      pipeline_root: root directory of the pipeline.
      data_root: directory containing the penguin data.
      module_file: path to files used in Trainer and Transform components.
      accuracy_threshold: minimum accuracy to push the model.
      serving_model_dir: filepath to write pipeline SavedModel to.
      metadata_path: path to local pipeline ML Metadata store.
      user_provided_schema_path: path to user provided schema file.
      enable_tuning: If True, the hyperparameter tuning through KerasTuner is
        enabled.
      enable_bulk_inferrer: If True, the generated model will be used for a
        batch inference.
      examplegen_input_config: ExampleGen's input_config.
      examplegen_range_config: ExampleGen's range_config.
      resolver_range_config: SpansResolver's range_config. Specify this will
        enable SpansResolver to get a window of ExampleGen's output Spans for
        transform and training.
      beam_pipeline_args: list of beam pipeline options for LocalDAGRunner. Please
        refer to https://beam.apache.org/documentation/runners/direct/.

    Returns:
      A TFX pipeline object.
    """

    # Brings data into the pipeline or otherwise joins/converts training data.
    example_gen = tfx.components.CsvExampleGen(
        input_base=os.path.join(data_root, 'labelled'),
        input_config=examplegen_input_config,
        range_config=examplegen_range_config)

    # statistics_gen,
    # example_validator,
    # transform,
    # trainer,
    # model_resolver,
    # evaluator,
    # pusher,

    components_list = [
        example_gen
    ]
    # if user_provided_schema_path:
    #     components_list.append(schema_importer)
    # else:
    #     components_list.append(schema_gen)
    # if resolver_range_config:
    #     components_list.append(examples_resolver)
    # if enable_tuning:
    #     components_list.append(tuner)
    # if enable_bulk_inferrer:
    #     components_list.append(example_gen_unlabelled)
    #     components_list.append(bulk_inferrer)

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components_list,
        enable_cache=True,
        metadata_connection_config=tfx.orchestration.metadata
            .sqlite_metadata_connection_config(metadata_path),
        beam_pipeline_args=beam_pipeline_args)


# To run this pipeline from the python CLI:
#   $python penguin_pipeline_local.py [--model_framework=flax_experimental]
if __name__ == '__main__':
    # absl.logging.set_verbosity(absl.logging.INFO)
    absl.flags.FLAGS(sys.argv)

    _pipeline_name = f'penguin_local_{flags.FLAGS.model_framework}'

    # Python module file to inject customized logic into the TFX components. The
    # Transform, Trainer and Tuner all require user-defined functions to run
    # successfully.
    _module_file_name = f'penguin_utils_{flags.FLAGS.model_framework}.py'
    _module_file = os.path.join(_penguin_root, _module_file_name)
    # Path which can be listened to by the model server.  Pusher will output the
    # trained model here.
    _serving_model_dir = os.path.join(_penguin_root, 'serving_model',
                                      _pipeline_name)
    # Pipeline root for artifacts.
    _pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)
    # Sqlite ML-metadata db path.
    _metadata_path = os.path.join(_tfx_root, 'metadata', _pipeline_name,
                                  'metadata.db')
    tfx.orchestration.LocalDagRunner().run(
        _create_pipeline(
            pipeline_name=_pipeline_name,
            pipeline_root=_pipeline_root,
            data_root=_data_root,
            module_file=_module_file,
            accuracy_threshold=0.6,
            serving_model_dir=_serving_model_dir,
            metadata_path=_metadata_path,
            user_provided_schema_path=None,
            # TODO(b/180723394): support tuning for Flax.
            enable_tuning=(flags.FLAGS.model_framework == 'keras'),
            enable_bulk_inferrer=True,
            examplegen_input_config=_examplegen_input_config,
            examplegen_range_config=_examplegen_range_config,
            resolver_range_config=_resolver_range_config,
            beam_pipeline_args=_beam_pipeline_args_by_runner[flags.FLAGS.runner]))
