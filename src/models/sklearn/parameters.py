#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 2021
@author: Simon Pelletier
"""

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier, \
    RandomForestRegressor, BaggingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.svm import LinearSVC, SVC, LinearSVR, SVR
from sklearn.neighbors import KNeighborsClassifier

params_sgd = {
    'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
    'penalty': ['elasticnet'],
    'l1_ratio': [0, 0.15, 0.5, 0.85, 1.0],
    'max_iter': [10000],
    'class_weight': ['balanced']
}

param_grid_rf = {
    'max_depth': [300],
    'max_features': [100],
    'min_samples_split': [300],
    'n_estimators': [100],
    'criterion': ['entropy'],
    'min_samples_leaf': [3],
    'oob_score': [False],
    'class_weight': ['balanced']
}
param_grid_rfr = {
    'max_depth': [300],
    'max_features': [100],
    'min_samples_split': [300],
    'n_estimators': [100],
    'min_samples_leaf': [3],
    'oob_score': [False],
}
param_grid_lda = {
}
param_grid_qda = {
}
param_grid_logreg = {
    # 'max_iter': [10000],
    'solver': ['saga'],
    'penalty': ['l1', 'l2'],
    'class_weight': ['balanced']
}
param_grid_linsvc = {
    'max_iter': [10000],
    'C': [1]
}
param_grid_svc = {
    'max_iter': [10000],
    'C': [1],
    'kernel': ['linear'],
    'probability': [True]
}
param_grid_ada = {
    'base_estimator': [LinearSVC(max_iter=10000)],
    'learning_rate': (1)
}
param_grid_bag = {
    'base_estimator': [
        LinearSVC(max_iter=1000)],
    'n_estimators': [10]
}
param_grid_knn = {
    'n_neighbors': [5, 10, 50, 100],
    'weights': ['distance', 'uniform']
}
param_grid_voting = {
    'voting': ('soft', 'hard'),
}
rf = RandomForestClassifier(max_depth=300, max_features=100, min_samples_split=300, n_estimators=100)
gnb = GaussianNB()
cnb = CategoricalNB()
lr = LogisticRegression(max_iter=4000)
lsvc = LinearSVC(max_iter=10000)
estimators_list = [('rf', rf),
                   ('lr', lr),
                   ('lsvc', lsvc),
                   ('gnb', gnb),
                   ]
