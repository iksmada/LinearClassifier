import argparse
import timeit

import numpy as np
import scipy.io

from ExtremeLearningMachine import ExtremeLearningMachine
from LinearClassifier import LinearClassifier

if __name__ == '__main__':
    classifiers = ['LinearClassifier', 'ExtremeLearningMachine']
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gamma", type=float, help="regularization value for linear classifier", default=1)
    parser.add_argument("-s", "--seed", type=int, help="Seed for random matrix generation", default=0)
    args = vars(parser.parse_args())

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

    stmt = timeit.Timer('clf.fit(X, Y_train); clf.predict(Xt)', globals=globals())

    clf = ExtremeLearningMachine(seed=args['seed'], gamma=args['gamma'])
    time_table = stmt.repeat(repeat=10, number=1)
    print("%0.3f s (+/-%0.03f) for LinearClassifier"
          % (np.mean(time_table), np.std(time_table) * 2))

    clf = LinearClassifier(gamma=args['gamma'])
    time_table = stmt.repeat(repeat=10, number=1)
    print("%0.3f s (+/-%0.03f) for ExtremeLearningMachine"
          % (np.mean(time_table), np.std(time_table) * 2))
