import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import load_boston

TRAIN_TEST_SPLIT = 0.33


def main(X_train, X_test, y_train, y_test):
    models = [
    {'name': 'MLP', 'model': MLPRegressor(), 'params': {'hidden_layer_sizes': [(100, ), (200,), (300,)], 'activation': ['logistic', 'tanh', 'relu'], 
        'solver': ['lbfgs', 'sgd','adam'], 'alpha': [0.01, 0.001, 0.0001]}}
    ]
    for model in models:
        parameters = model['params']
        gb_clf = model['model']
        clf = GridSearchCV(gb_clf, parameters, scoring = 'neg_mean_squared_error', cv=5, verbose=0, n_jobs=4)
        clf.fit(X_train, y_train)
        print('MODEL - {} '.format(model['name']))
        print('Best model: {}'.format(clf.best_estimator_))
        print('Best Score: {}'.format(clf.best_score_))


if __name__ == '__main__':
    data = load_boston()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=TRAIN_TEST_SPLIT, random_state=42)
    main(X_train, X_test, y_train, y_test)