from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np


class LinearClassifier(BaseEstimator, ClassifierMixin):
    weights_ = None

    def __init__(self, gamma=1):
        self.gamma = gamma

    def fit(self, X, y: np.uint8):
        # label validation
        y = check_array(y, dtype='uint8', ensure_2d=False)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        # get classes
        self.classes_ = list(range(0, y.max()+1))
        # Check that X and y have correct shape
        X, y = check_X_y(X, y, y_numeric=True, multi_output=True)
        # create one hot encoded matrix
        y_encoded = np.zeros((len(y), len(self.classes_)))
        # put on correct columns 1 value
        np.put_along_axis(y_encoded, y, 1, axis=1)
        # add bias to X
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        #  w = (x_t*x + gamma*I)^-1 * x_t*y
        self.weights_ = np.linalg.inv(X.transpose().dot(X) + self.gamma * np.identity(X.shape[1])).dot(X.transpose().dot(y_encoded))
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ["weights_"])
        # Input validation
        X = check_array(X, ensure_min_features=self.weights_.shape[0]-1)
        # add bias to X and remove colums in order to match lines in W
        X = np.hstack((np.ones((X.shape[0], 1)), X[:, :self.weights_.shape[0]-1]))
        # x * w
        y = X.dot(self.weights_)
        #  need to convert hot encoded to labels
        return y.argmax(axis=1)
