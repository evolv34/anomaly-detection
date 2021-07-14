import math

from tensorflow.python.keras.models import Model
from tensorflow.keras import layers


class FraudDetectDenseModel(Model):

    def __init__(self, feature_columns, num_classes=2):
        super(FraudDetectDenseModel, self).__init__()
        self.feature_layer = layers.DenseFeatures(feature_columns)
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(.1)
        self.classifier = layers.Dense(math.ceil(num_classes / 2), activation='sigmoid')

    def call(self, inputs, training=False, mask=None):
        x = self.feature_layer(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dropout(x)
        return self.classifier(x)
