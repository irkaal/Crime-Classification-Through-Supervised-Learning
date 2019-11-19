import os
import time
import random
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV

train_data = pd.read_csv('data/clean_centered.csv')
y_train = train_data['Category']
X_train = train_data.drop('Category', axis = 1)

param_grid = {
    'loss': ['log'],
    'penalty': ['l1', 'l2', 'elasticnet'], 
    'alpha': [1e-1, 1e0, 1e1], 
    'l1_ratio': [0.15], 
    'fit_intercept': [True, False], 
    'max_iter': [1000], 
    'tol': [0.001], 
    'shuffle': [True], 
    'verbose': [0], 
    'epsilon': [0.1], 
    'n_jobs': [None], 
    'random_state': [None], 
    'learning_rate': ['optimal'],
    'eta0': [0.0], 
    'power_t': [0.5], 
    'early_stopping': [False], 
    'validation_fraction': [0.1], 
    'n_iter_no_change': [5], 
    'class_weight': [None], 
    'warm_start': [0],
    'average': [True, False]
}

grid_search = GridSearchCV(
    estimator = linear_model.SGDClassifier(), 
    param_grid = param_grid,
    scoring = 'neg_log_loss',
    n_jobs = -1, 
    cv = 2, 
    refit = False,
    verbose = 2)

grid_search.fit(X_train, y_train)

pd.DataFrame(grid_search.cv_results_).describe()
