from tensorflow.python.keras.models import Model
from tensorflow import keras
from tensorflow.keras import layers


class FraudDetectAutoEncoder(Model):

    def __init__(self, feature_columns, num_features: int):
        super(FraudDetectAutoEncoder, self).__init__()
        self.encoder = keras.Sequential()
        self.encoder.add(layers.DenseFeatures(feature_columns))
        self.encoder.add(layers.Dense(128, activation='relu'))
        self.encoder.add(layers.Dense(64, activation='relu'))
        self.encoder.add(layers.Dense(32, activation='relu'))
        self.encoder.add(layers.Dense(16, activation='relu'))
        self.encoder.add(layers.Dense(8, activation='relu'))

        self.decoder = keras.Sequential()
        self.decoder.add(layers.Dense(32, activation='relu')),
        self.decoder.add(layers.Dense(64, activation='relu')),
        self.decoder.add(layers.Dense(128, activation='relu')),
        self.decoder.add(layers.Dense(num_features, activation='sigmoid'))

    def call(self, inputs, training=False, mask=None):
        encoded = self.encoder(inputs)
        x = self.decoder(encoded)
        return x
