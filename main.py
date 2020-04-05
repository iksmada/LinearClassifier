#!/usr/bin/env python3
import scipy.io
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from LinearClassifier import LinearClassifier

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

    best_model = GridSearchCV(
        cv=5,
        estimator=LinearClassifier(),
        param_grid={'gamma': list(2**x for x in range(-10, 11))},
        scoring=('accuracy', 'neg_mean_squared_error'),
        refit='neg_mean_squared_error'
                              )

    best_model.fit(X, Y.argmax(axis=1))
    y_pred = best_model.predict(Xt)

    print(classification_report(Yt.argmax(axis=1), y_pred, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
