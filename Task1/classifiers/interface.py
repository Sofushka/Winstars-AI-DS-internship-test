from abc import abstractmethod, ABC


class MnistClassifierInterface(ABC):
    """
    Abstract base class that defines the interface
    for all MNIST classifiers
    """
    @abstractmethod
    def train(self, X, y):
        """
        Train the classifier on the provided dataset

        :param X: Training features
        :param y: Training labels
        :return: None(the trained model is stored inside the class)
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predict labels for new data

        :param X: Testing features
        :return: Array containing predicted labels
        """
        pass