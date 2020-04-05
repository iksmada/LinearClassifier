#!/usr/bin/env python3
import scipy.io
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from LinearClassifier import LinearClassifier

if __name__ == '__main__':
    # loads MATLAB matrix
    data = scipy.io.loadmat('data.mat')
    test = scipy.io.loadmat('test.mat')

    # training set, divide data (X) and solution (Y)
    X = data['X']
    Y = data['S']

    # test set, divide data (X) and solution (Y)
    Xt = test['Xt']
    Yt = test['St']

    # Grid Search on gamma hyperparameter
    search = GridSearchCV(
        # default 4 folds, 3 train 1 test
        cv=4,
        # customized clissfier
        estimator=LinearClassifier(),
        # 2^-10 to 2^10
        param_grid={'gamma': list(2 ** x for x in range(-10, 11, 2))},
        # accuracy and mean square error scoring methods
        scoring=('accuracy', 'neg_mean_squared_error'),
        # to use best model returned, we need to set a tiebreaker criteria
        refit='neg_mean_squared_error',
        #  -1 means using all processors
        n_jobs=-1
    ).fit(X, Y.argmax(axis=1))

    # Print results
    print("Best parameters set found on development set:")
    print()
    print(search.best_params_)
    print()
    print("Grid Accuracy on development set:")
    print()
    means_acc = search.cv_results_['mean_test_accuracy']
    stds_acc = search.cv_results_['std_test_accuracy']
    for mean, std, params in zip(means_acc, stds_acc, search.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    print("Grid mean squared error on development set:")
    print()
    means_acc = search.cv_results_['mean_test_neg_mean_squared_error']
    stds_acc = search.cv_results_['std_test_neg_mean_squared_error']
    for mean, std, params in zip(means_acc, stds_acc, search.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    y_pred = search.predict(Xt)

    print(classification_report(Yt.argmax(axis=1), y_pred,
                                target_names=['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']))
