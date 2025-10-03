from .interface import MnistClassifierInterface
from sklearn.ensemble import RandomForestClassifier


class RandomForestMnistClassifier(MnistClassifierInterface):
    """
    Random Forest classifier implementation for MNIST dataset.
    """
    def __init__(self):
        """
        Initialize RandomForestClassifier instance
        """
        self.rf = RandomForestClassifier()

    def train(self, X, y):
        """
        Train Random Forest Classifier

        :param X: Training features
        :param y: Training labels
        """
        self.rf.fit(X, y)


    def predict(self, X):
        """
        Make prediction

        :param X: Testing features
        :return: Numpy.ndarray with predicted labels
        """
        return self.rf.predict(X)
