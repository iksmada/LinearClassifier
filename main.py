#!/usr/bin/env python3
from numpy import ndarray
import numpy as np
import scipy.io
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder


def calculate_weights(x: ndarray, y: ndarray, gamma: float = 1.0) -> ndarray:
    #  inverse(x_t*x + gamma*I)*x_t*y
    return np.linalg.inv(x.transpose().dot(x) + gamma * np.identity(x.shape[1])).dot(x.transpose().dot(y))


def predict(x: ndarray, w: ndarray) -> ndarray:
    return x.dot(w)


if __name__ == '__main__':
    # loas MATLAB matrix
    data = scipy.io.loadmat('data.mat')
    test = scipy.io.loadmat('test.mat')

    # training set, divide data (X) and solution (Y)
    X = data['X']
    Y = data['S']

    # test set, divide data (X) and solution (Y)
    Xt = test['Xt']
    Yt = test['St']

    W = calculate_weights(X, Y)
    y_pred = predict(Xt, W)

    #  need to convert regression to labels
    y_pred = y_pred.argmax(axis=1)
    #  one hot encoding
    onehot_encoder = OneHotEncoder(categories=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], sparse=False, dtype="uint8")
    y_pred_encoded = onehot_encoder.fit_transform(y_pred.reshape(-1, 1))

    print(classification_report(Yt.argmax(axis=1), y_pred, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
