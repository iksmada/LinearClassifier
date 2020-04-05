from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
import numpy as np


class LinearClassifier(BaseEstimator, ClassifierMixin):
    weights_ = None

    def __init__(self, gamma=1):
        self.gamma = gamma

    def fit(self, X, y):
        self.classes_ = unique_labels(y)
        # create one hot encoded matrix
        y_encoded = np.zeros((len(y), len(self.classes_)))
        # put on correct columns 1 value
        np.put_along_axis(y_encoded, y.reshape(-1, 1), 1, axis=1)
        #  w = (x_t*x + gamma*I)^-1 * x_t*y
        self.weights_ = np.linalg.inv(X.transpose().dot(X) + self.gamma * np.identity(X.shape[1])).dot(X.transpose().dot(y_encoded))
        return self

    def predict(self, X):
        y = X.dot(self.weights_)
        #  need to convert hot encoded to labels
        return y.argmax(axis=1)

    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"gamma": self.gamma}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
