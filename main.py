#!/usr/bin/env python3
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_confusion_matrix

from LinearClassifier import LinearClassifier

import matplotlib.pyplot as plt
import numpy as np
import scipy.io


def run_gridsearch_and_plot(X, Y, param_grid: list, name: str):
    # Grid Search on gamma hyperparameter
    search = GridSearchCV(
        # default 4 folds, 3 train 1 test
        cv=4,
        # customized clissfier
        estimator=LinearClassifier(),
        # 2^-10 to 2^10
        param_grid={'gamma': param_grid},
        # accuracy and mean square error scoring methods
        scoring=('accuracy', 'neg_mean_squared_error'),
        # to use best model returned, we need to set a tiebreaker criteria
        refit=False,
        #  -1 means using all processors
        n_jobs=-1,
        verbose=1
    ).fit(X, Y)

    # Print results
    fig, ax1 = plt.subplots()
    plt.title(name)
    plt.xscale("log")
    plt.grid()
    gamma_params = list(param['gamma'] for param in search.cv_results_['params'])

    print("########################### %s ###########################" % name)
    print()
    print("Best parameters set found on development set:")
    print()
    print('Best Acc for %r' % search.cv_results_['params'][search.cv_results_['rank_test_accuracy'][0] - 1])
    print('Best MSE for %r' % search.cv_results_['params'][search.cv_results_['rank_test_neg_mean_squared_error'][0] - 1])
    print()
    print("Grid Accuracy on development set:")
    print()
    means_acc = search.cv_results_['mean_test_accuracy']
    stds_acc = search.cv_results_['std_test_accuracy']
    for mean, std, params in zip(means_acc, stds_acc, search.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    # add info to graph
    color = 'tab:red'
    ax1.set_xlabel('Coeficiente de Regularização')
    ax1.set_ylabel('Acurácia', color=color)
    ax1.plot(gamma_params, means_acc, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    print()
    print("Grid mean squared error on development set:")
    print()
    means_mse = search.cv_results_['mean_test_neg_mean_squared_error']
    stds_mse = search.cv_results_['std_test_neg_mean_squared_error']
    for mean, std, params in zip(means_mse, stds_mse, search.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    # add more info to graph
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Erro quadrático médio', color=color)  # we already handled the x-label with ax1
    ax2.plot(gamma_params, means_mse, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    return search.cv_results_['params'][search.cv_results_['rank_test_accuracy'][0]-1]['gamma'], search.cv_results_['params'][search.cv_results_['rank_test_neg_mean_squared_error'][0]-1]['gamma']


if __name__ == '__main__':
    # loads MATLAB matrix
    data = scipy.io.loadmat('data.mat')
    test = scipy.io.loadmat('test.mat')

    # training set, divide data (X) and solution (Y)
    X = data['X']
    Y = data['S']
    Y_train = Y.argmax(axis=1)

    # test set, divide data (X) and solution (Y)
    Xt = test['Xt']
    Yt = test['St']
    Y_test = Yt.argmax(axis=1)

    param_list = list(2 ** x for x in range(-10, 12))
    best_param = run_gridsearch_and_plot(X, Y_train, param_list, '1st Step Grid search')
    # check if [1] is bigger
    if np.argmax(best_param) == 0:
        best_param = (best_param[1], best_param[0])
    # get the power
    best_param = np.log2(best_param)

    # add margin for second search for params
    margin = 0.5
    if best_param[0] == best_param[1]:
        margin = 1
    best_param[0] = best_param[0] - margin
    best_param[1] = best_param[1] + margin

    # generate new param list among the best results
    param_list = list(2 ** x for x in np.linspace(best_param[0], best_param[1], 11))
    best_acc, beat_mse = run_gridsearch_and_plot(X, Y_train, param_list, '2nd Step Grid search')

    # train best gamma Acc
    clf = LinearClassifier(gamma=best_acc).fit(X, Y_train)
    y_pred = clf.predict(Xt)

    # print results for best classifier
    print(classification_report(Y_test, y_pred,
                                target_names=['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']))
    print('Acc for gamma %f was %f' % (clf.gamma, accuracy_score(Y_test, y_pred)))
    print('MSE for gamma %f was %f' % (clf.gamma, mean_squared_error(Y_test, y_pred)))

    # display confusion matrix
    disp = plot_confusion_matrix(clf, Xt, Y_test,
                                 display_labels=['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
                                 cmap=plt.cm.Blues,
                                 normalize='true',
                                 values_format='.2f')

    disp.ax_.set_title("Matrix de Confusão")
    plt.show()

    # save weights
    np.savetxt("weights.txt", clf.weights_)

