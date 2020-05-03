from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np


class ExtremeLearningMachine(BaseEstimator, ClassifierMixin):
    weights_ = None
    V_ = None

    def __init__(self, gamma=1, neurons=500, seed=None):
        self.gamma = gamma
        self.neurons = neurons
        self.seed = seed

    def fit(self, X, y: np.uint8):
        # label validation
        y = check_array(y, dtype='uint8', ensure_2d=False)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        # get classes
        self.classes_ = list(range(0, y.max() + 1))
        # Check that X and y have correct shape
        X, y = check_X_y(X, y, y_numeric=True, multi_output=True)
        # create one hot encoded matrix
        y_encoded = np.zeros((len(y), len(self.classes_)))
        # put on correct columns 1 value
        np.put_along_axis(y_encoded, y, 1, axis=1)
        # add bias to X
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        # generate random matrix if it does not exist or it is a different size
        rng = np.random.default_rng(seed=self.seed)
        self.V_ = rng.normal(scale=0.2, size=(X.shape[1], self.neurons))
        # calculate activator for mid-layer
        H = np.tanh(X.dot(self.V_))
        # add bias to H
        H = np.hstack((np.ones((H.shape[0], 1)), H))
        # if n <= N
        if self.neurons <= X.shape[0]:
            #  w = (h_t*h + gamma*I)^-1 * h_t*y
            self.weights_ = np.linalg.inv(H.transpose().dot(H) + self.gamma * np.identity(H.shape[1])).dot(
                H.transpose().dot(y_encoded))
        else:
            # n > N
            #  w = h_t(h*h_t + gamma*I)^-1 * y
            self.weights_ = H.transpose().dot(
                np.linalg.inv(H.dot(H.transpose()) + self.gamma * np.identity(H.shape[1]))).dot(y_encoded)
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ["weights_", "V_"])
        # Input validation
        X = check_array(X, ensure_min_features=self.V_.shape[0] - 1)
        # add bias to X and remove colums in order to match lines in W
        X = np.hstack((np.ones((X.shape[0], 1)), X[:, :self.V_.shape[0] - 1]))
        # calculate activation funcion:
        H = np.tanh(X.dot(self.V_))
        # add bias to H
        H = np.hstack((np.ones((H.shape[0], 1)), H))
        # h * w
        y = H.dot(self.weights_)
        #  need to convert hot encoded to labels
        return y.argmax(axis=1)
