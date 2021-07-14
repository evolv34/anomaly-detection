from .utils import Utils
from collections import OrderedDict
from tensorflow import dtypes, feature_column, data
import tensorflow as tf


class FeatureUtils(Utils):
    def __init__(self):
        super(FeatureUtils).__init__()

    def to_feature_columns(self, train_ds: data.Dataset) -> list:
        features_dataset = train_ds.take(1)
        features_dataset_iter = iter(features_dataset)
        feature_columns_dict = features_dataset_iter.get_next()[0]
        return self.features(feature_columns_dict)

    def features(self, row: OrderedDict,
                 hash_bucket_size=200) -> list:
        keys = row.keys()
        feature_columns = list()
        for key in keys:
            col_values = row[key]
            datatype = col_values.dtype
            print(f"name = {key} datatype={datatype}")
            if datatype == dtypes.string:
                str_hash_col = feature_column.categorical_column_with_hash_bucket(key, hash_bucket_size)
                col_indicator = feature_column.indicator_column(str_hash_col)
                feature_columns.append(col_indicator)
            if datatype == dtypes.int8 or datatype == dtypes.int16 or datatype == dtypes.int32 or datatype == dtypes.int64:
                col_indicator = feature_column.numeric_column(key)
                feature_columns.append(col_indicator)
            if datatype == dtypes.float16 or datatype == dtypes.float32 or datatype == dtypes.float64:
                col_indicator = feature_column.numeric_column(key)
                feature_columns.append(col_indicator)

        return feature_columns

    def class_weights(self, dataset: data.Dataset) -> OrderedDict:
        stacked_count_list = list()
        prev_shape = None
        for ds in dataset:
            sort_tensor = tf.sort(ds[1])
            y, _, count = tf.unique_with_counts(sort_tensor)
            if prev_shape is not None and prev_shape != count.shape:
                count = tf.concat((count, tf.zeros(shape=(tf.abs(prev_shape[0] - count.shape[0]),), dtype=tf.int32)),
                                  axis=0)
            stacked_count_list.append(count)
            prev_shape = count.shape
        counts = tf.reduce_sum(tf.stack(stacked_count_list), axis=0)
        total_counts = tf.reduce_sum(counts).numpy()
        counts_dict = OrderedDict()
        for index, count_value in enumerate(counts):
            counts_dict[index] = total_counts / count_value.numpy()
        return counts_dict
