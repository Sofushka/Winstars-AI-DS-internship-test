from .interface import MnistClassifierInterface
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np



class ConvolutionalNeuralNetworkMnist(MnistClassifierInterface):
    """
    Convolutional Neural Network implementation for MNIST dataset.
    This class implements MnistClassifierInterface and provides train
    and predict methods.
    """
    def __init__(self):
        """
        Initialize Convolutional Neural Network instance
        """
        self.cnn = keras.models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPool2D((2,2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPool2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10)
        ])

    def train(self, X, y, epochs=5, batch_size=64):
        """
        Compile the model and train CNN.

        :param X: Training features, shape (num_samples, 28, 28, 1)
        :param y: Training labels, shape (num_samples,)
        :param epochs: Number of training epochs
        :param batch_size: Batch size for training
        """
        self.cnn.compile(
            optimizer='adam',
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        self.cnn.fit(X, y, epochs=epochs, batch_size=batch_size)


    def predict(self, X):
        """
        Make prediction

        :param X: Testing features, shape (num_samples, 28, 28, 1)
        :return: Numpy.ndarray with predicted labels
        """
        logits = self.cnn.predict(X)
        probs = tf.nn.softmax(logits).numpy()
        #Returns class indices with maximum probability
        return np.argmax(probs, axis=1)