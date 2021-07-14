import os
from collections import OrderedDict
from datetime import datetime

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import feature_column

from app.models import FraudDetectDenseModel, FraudDetectAutoEncoder
from source import CSVSource
from utils import PreprocessorUtils, FeatureUtils
from constants import Metrics

dest_folder = ""
is_repartition = False


def is_not_fraud_filter(ds):
    print(ds['isFraud'] == 0)
    return tf.reduce_all(tf.equal(ds['isFraud'], 0))


def repartition_feature_files() -> tf.data.Dataset:
    files = os.listdir(dest_folder)
    for file in files:
        os.remove(os.path.join(dest_folder, file))
    os.rmdir(dest_folder)
    os.mkdir(dest_folder)

    csv_source = CSVSource("", None)
    dataset = csv_source.load(batch_size=1,
                              num_epochs=1,
                              shuffle=False)

    dataset = dataset.filter(is_not_fraud_filter)
    dataset = dataset.batch(10000)
    count = 0
    dataset_iter = dataset.as_numpy_iterator()

    for ds in dataset_iter:
        for key in ds.keys():
            ds[key] = tf.reshape(ds[key], (ds[key].shape[0],))
        df = pd.DataFrame.from_records(data=ds)

        # TODO: revisit to implement utf-8 conversion in tensorflow
        for col, dtype in df.dtypes.items():
            if dtype == np.object:  # Only process byte object columns.
                df[col] = df[col].apply(lambda x: x.decode("utf-8"))

        df.to_csv(os.path.join(dest_folder, f"{count}.csv"), index=False)
        count += 1
    print(f"create {count} files")
    return dataset


def load_csv_dataset(path: str,
                     label_col: str,
                     batch_size: int,
                     num_epochs: int = None,
                     shuffle: bool = True) -> tf.data.Dataset:
    csv_source = CSVSource(path, label_col)
    dataset = csv_source.load(batch_size,
                              num_epochs=num_epochs,
                              shuffle=shuffle)
    return dataset


# def compile_model(metrics):
#     model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
#                   loss='mae')


if __name__ == '__main__':
    for gpu in tf.config.list_physical_devices("GPU"):
        print(f"setting memory for {gpu}")
        tf.config.experimental.set_virtual_device_configuration(gpu, [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])

    pre_processor_utils = PreprocessorUtils()
    feature_utils = FeatureUtils()
    metrics = Metrics()

    if is_repartition:
        repartition_feature_files()
    path = os.path.join(dest_folder, "*.csv")

    # total batches = 2100
    # num elements in a batch = 100
    dataset = load_csv_dataset(path, "isFraud", 30, shuffle=True)

    for ds in dataset.take(1):
        print(ds[0]['C5'])
    # train_ds, val_ds, test_ds = pre_processor_utils.time_series_train_val_test_split(dataset,
    #                                                                                  total_batches=2100,
    #                                                                                  train_split_percent=0.70)
    #
    # # cls_weights = feature_utils.class_weights(train_ds)
    # # print(cls_weights)
    #
    # # feature pre processing layers
    # feature_columns = feature_utils.to_feature_columns(train_ds)
    # print(f"total feature columns = [{len(feature_columns)}]")
    #
    # # Model building layer
    # model = FraudDetectAutoEncoder(feature_columns, len(feature_columns))
    # METRICS = [
    #     metrics.TRUE_POSITIVE,
    #     metrics.FALSE_POSITIVE,
    #     metrics.TRUE_NEGATIVE,
    #     metrics.FALSE_NEGATIVE,
    #     metrics.BINARY_ACCURACY,
    #     metrics.PRECISION,
    #     metrics.RECALL,
    #     metrics.AUC
    # ]
    # compile_model([metrics.BINARY_ACCURACY])
    # log_dir = f"logs/fit/{datetime.now()}"
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    #
    # # class_weight=cls_weights,
    # # callbacks=[tensorboard_callback]
    # model.fit(x=train_ds,
    #           validation_data=val_ds,
    #           epochs=5)

# if __name__ == '__main__':
#     cluster_spec = tf.train.ClusterSpec(
#         {
#             'workers': ['localhost:2222', 'localhost:2223', 'localhost:2224']
#         }
#     )
#     for i in [0, 1, 2]:
#         print(f"index = {i}")
#         server = tf.distribute.Server(cluster_spec, task_index=i)
#         server.start()
#
#     while True:
#         print("Server started")
#         time.sleep(60)
