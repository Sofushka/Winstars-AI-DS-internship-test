from .interface import MnistClassifierInterface
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np



class FeedForwardNeuralNetworkMnist(MnistClassifierInterface):
    """
    Feed-Forward Neural Network implementation for MNIST dataset.
    This class implements MnistClassifierInterface and provides train
    and predict methods.
    """
    def __init__(self):
        """
        Initialize Feed-Forward Neural Network instance
        """
        self.nn = keras.models.Sequential([
            layers.Flatten(input_shape=(28, 28)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(10)
        ])

    def train(self, X, y, epochs=5, batch_size=64):
        """
        Compile the model with loss function and optimizer
        Train Feed-Forward Neural Network

        :param X: Training features, shape (num_samples, 28, 28)
        :param y: Training labels, shape (num_samples)
        :param Epochs: Number of training epochs
        :param Batch_size: Batch size for training
        """
        self.nn.compile(
            optimizer='adam',
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        self.nn.fit(X, y, epochs=epochs, batch_size=batch_size)


    def predict(self, X):
        """
        Make prediction

        :param X: Testing features, shape (num_samples, 28, 28)
        :return: Numpy.ndarray with predicted labels
        """
        logits = self.nn.predict(X)
        probs = tf.nn.softmax(logits).numpy()
        #Returns class indices with maximum probability
        return np.argmax(probs, axis=1)