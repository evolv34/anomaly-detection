import math

from .utils import Utils
from tensorflow import data


class PreprocessorUtils(Utils):

    def __init__(self):
        super(PreprocessorUtils).__init__()

    def time_series_train_val_test_split(self,
                                         dataset: data.Dataset,
                                         total_batches=2100,
                                         train_split_percent=0.70) -> (data.Dataset, data.Dataset, data.Dataset):
        total_train_batches = int(math.floor(total_batches * train_split_percent))
        train_batches = int(math.floor(total_train_batches * 0.8))
        val_batches = int(math.floor(total_batches * train_split_percent) - train_batches)
        test_batches = int(total_batches - total_train_batches)

        print(f"total_train_batches {total_train_batches}")
        print(f"train_batches {train_batches}")
        print(f"val_batches {val_batches}")
        print(f"test_batches {test_batches}")

        train_ds = dataset.take(total_batches)
        val_ds = dataset.skip(train_batches).take(val_batches)
        test_ds = dataset.skip(total_train_batches).take(test_batches)

        return train_ds, val_ds, test_ds
