#!/usr/bin/env python3
import scipy.io
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from LinearClassifier import LinearClassifier

import matplotlib.pyplot as plt


def run_gridsearch_and_plot(param_grid: list):
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
    ).fit(X, Y.argmax(axis=1))

    # Print results
    fig, ax1 = plt.subplots()
    plt.title('1st Step Grid search')
    plt.xscale("log")
    plt.grid()
    gamma_params = list(param['gamma'] for param in search.cv_results_['params'])

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

    # test set, divide data (X) and solution (Y)
    Xt = test['Xt']
    Yt = test['St']

    best_acc_gamma, best_mse_gamma = run_gridsearch_and_plot(list(2 ** x for x in range(-10, 12, 10)))
    # ordena os dois bests, adiciona um pra cada lado, e faz grid search
    #best_acc_gamma, best_mse_gamma = run_gridsearch_and_plot(list(2 ** x for x in range(, 12, 10)))

    #y_pred = search.predict(Xt)

    #print(classification_report(Yt.argmax(axis=1), y_pred,
    #                            target_names=['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']))
