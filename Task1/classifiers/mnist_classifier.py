from .nn import FeedForwardNeuralNetworkMnist
from .cnn import ConvolutionalNeuralNetworkMnist
from .rf import RandomForestMnistClassifier

class MnistClassifier:
    """
    Wrapper class that provides a unified interface
    for different MNIST classification algorithms.
    """
    def __init__(self, algorithm):
        """
        Choose the model to work with
        :param algorithm: str, one of ['rf', 'nn', 'cnn']
        """
        if algorithm == 'nn':
            self.model = FeedForwardNeuralNetworkMnist()
        elif algorithm == 'cnn':
            self.model = ConvolutionalNeuralNetworkMnist()
        elif algorithm == 'rf':
            self.model = RandomForestMnistClassifier()
        else:
            raise ValueError('Invalid algorythm type. Please choose "rf", "nn" or "cnn".')

    def train(self, X, y, **kwargs):
        """
        Train the chosen model.

        :param X: Training features
        :param y: Training labels
        :param kwargs: Additional parameters for model's train method, e.g., epochs, batch_size
        """
        return self.model.train(X, y, **kwargs)


    def predict(self, X):
        """
        Make predictions using the chosen model.

        :param X: Testing features
        :return: Numpy.ndarray with predicted labels
        """
        return self.model.predict(X)