from tensorflow.keras import layers, models, optimizers
import tensorflow as tf
import numpy as np

class NetworkB:
    def __init__(self, learning_rate=1e-3):
        self.model = models.Sequential([
            layers.Input(shape=(6, 7, 1)),
            layers.Conv2D(128, (4, 4), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])
        self.optimizer = optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer, metrics=['accuracy'])

    def predict(self, boards):
        boards = boards[..., np.newaxis]
        return self.model(boards, training=False).numpy()

    def train_on_batch(self, boards, targets):
        boards = boards[..., np.newaxis]
        return self.model.train_on_batch(boards, targets)

    def save(self, filepath="networkB.h5"):
        self.model.save(filepath)

    def load(self, filepath="networkB.h5"):
        self.model = tf.keras.models.load_model(filepath)
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer, metrics=['accuracy'])
