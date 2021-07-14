# from .preprocessor import PreprocessorPreprocessor
#
#
# class Imputer(Preprocessor):
#     def __init__(self):
#         super(Imputer, self).__init__()
#

from typing import Dict, Text, Any


def preprocessing_fn(inputs: Dict[Text, Any]) -> Dict[Text, Any]:
    print(f"input from process fn imputer {inputs}")
    return {}