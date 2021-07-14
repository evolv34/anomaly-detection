# from .preprocessor import Preprocessor
#
#
# class StandardScalarPreprocessor(Preprocessor):
#     def __init__(self):
#         super(StandardScalarPreprocessor, self).__init__()
#
from typing import Dict, Text, Any
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.python.framework import sparse_tensor

_CATEGORICAL_VARIABLES = ['ProductCD']
_VOCAB_SIZE = 1000
_OOV_SIZE = 10


def _transform_key(x: Text) -> Text:
    return f"{x}_transform"


def _fill_in_missing(x):
    """Replace missing values in a SparseTensor.
    Fills in missing values of `x` with '' or 0, and converts to a dense tensor.
    Args:
      x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
        in the second dimension.
    Returns:
      A rank 1 tensor where missing values of `x` have been filled in.
    """
    if not isinstance(x, tf.sparse.SparseTensor):
        return x

    default_value = '' if x.dtype == tf.string else 0
    return tf.squeeze(
        tf.sparse.to_dense(
            tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
            default_value),
        axis=1)


def preprocessing_fn(inputs: Dict[Text, Any]) -> Dict[Text, Any]:
    outputs = inputs.copy()
    for key in outputs.keys():
        tensor = outputs[key]
        print(f"key = {key} data-type={tensor.dtype} class={type(tensor)}")
        if tensor.dtype is not tf.string:
            print(f"key = {key}")
            if isinstance(tensor, sparse_tensor.SparseTensor):
                if tensor.dtype == tf.bfloat16 or tensor.dtype == tf.float16 or tensor.dtype == tf.float32 or tensor.dtype == tf.float64:
                    print(f"key = {key} = values{tf.math.is_finite(tf.sparse.to_dense(tensor))}")
            # else:
            #     print(f"key = {key} = values{tf.math.(tensor)}")
        # outputs[_transform_key(cat_key)] = tft.compute_and_apply_vocabulary(
        #     _fill_in_missing(inputs[cat_key]),
        #     top_k=_VOCAB_SIZE,
        #     num_oov_buckets=_OOV_SIZE)
    return outputs
