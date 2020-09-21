from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from collections import Counter
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
TRAIN_TEST_SPLIT = 0.33


def main(X_train, X_test, y_train, y_test):
    models = [
    {'name':'XGB' , 'model': XGBClassifier(), 'params': {'colsample_bytree': [0.3, 0.4, 0.5], 'learning_rate':[0.1, 0.05, 0.001],
         'max_depth':[4, 5, 6], 'n_estimators':[10,20,100]}},            
    {'name':'SVM' , 'model': svm.SVC(probability=True), 'params': {'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}},
    {'name': 'Gradient Boosting', 'model': GradientBoostingClassifier(), 'params': {'loss': ['deviance', 'exponential'], 'learning_rate': [0.01, 0.001], 
                'n_estimators': [750, 1000], 'subsample': [0.7, 0.8], 'max_leaf_nodes': [None, 5, 7]}},
    {'name': 'MLP', 'model': MLPClassifier(), 'params': {'hidden_layer_sizes': [(100, ), (200,), (300,)], 'activation': ['logistic', 'tanh', 'relu'], 
        'solver': ['lbfgs', 'sgd','adam'], 'alpha': [0.01, 0.001, 0.0001]}},
    {'name': 'AdaBoost', 'model': AdaBoostClassifier(), 'params': {'n_estimators':[40, 50, 75, 100], 'learning_rate': [0.5, 1, 1.5, 2]}},
    {'name': 'DecisionTrees', 'model': DecisionTreeClassifier(), 'params': {'criterion': ['gini', 'entropy'], 'max_depth': [None, 5, 7], 'min_samples_split': [1,2,3]}},
    {'name': 'ExtraTreesClassifier', 'model': ExtraTreesClassifier(), 'params': {'criterion': ['gini', 'entropy'], 'max_depth': [None, 5, 7], 'min_samples_split': [1,2,3]}},
    {'name': 'RandomForest', 'model': RandomForestClassifier(), 'params': {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 7], 'min_samples_split': [1 ,2, 3]}},    
    ]
    for model in models:
        parameters = model['params']
        gb_clf = model['model']
        clf = GridSearchCV(gb_clf, parameters, scoring = 'roc_auc', cv=5, verbose=0, n_jobs=4)
        clf.fit(X_train, y_train)
        print('MODEL - {} '.format(model['name']))
        print('Best model: {}'.format(clf.best_estimator_))
        print('Best Score: {}'.format(clf.best_score_))
        y_score = clf.predict_proba(X_test)[:,1]
        print('Test Score: {}'.format(roc_auc_score(y_test, y_score)))


if __name__ == '__main__':
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=TRAIN_TEST_SPLIT, random_state=42)
    main(X_train, X_test, y_train, y_test)