import os
import time
import random
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV

train_data = pd.read_csv('data/clean_centered_final.csv')
y_train = train_data['Category']
X_train = train_data.drop('Category', axis = 1)

# First hyperparameter search (penalty, alpha, class_weight)

param_grid = {
    'loss': ['modified_huber'],
    'penalty': ['l1', 'l2', 'elasticnet'], 
    'alpha': [0.075, 0.1, 0.125], 
    'l1_ratio': [0.15], 
    'fit_intercept': [False], 
    'max_iter': [1000], 
    'tol': [1e-3], 
    'epsilon': [0.1], 
    'random_state': [2019], 
    'learning_rate': ['optimal'],
    'eta0': [0], 
    'power_t': [0.5], 
    'class_weight': ['balanced', None]
}
grid_search = GridSearchCV(
    estimator = linear_model.SGDClassifier(), 
    param_grid = param_grid,
    scoring = 'neg_log_loss',
    n_jobs = -1, 
    cv = 3, 
    refit = False,
    verbose = 2)
grid_search.fit(X_train, y_train)
grid_search.best_params_

pd.DataFrame(grid_search.cv_results_).to_csv('results.csv', index = False)

# Second hyperparameter search (smaller alpha range, penalty, elasticnet + l1_ratio)

param_grid = {
    'loss': ['modified_huber'],
    'penalty': ['l2'], 
    'alpha': [0.075, 0.1, 0.125], 
    'l1_ratio': [0.15], 
    'fit_intercept': [False], 
    'max_iter': [3000], 
    'tol': [1e-2], 
    'shuffle': [True], 
    'epsilon': [0.1], 
    'random_state': [2019], 
    'learning_rate': ['optimal'],
    'eta0': [0.0], 
    'power_t': [0.5], 
    'class_weight': ['balanced', None]
}

grid_search = GridSearchCV(
    estimator = linear_model.SGDClassifier(), 
    param_grid = param_grid,
    scoring = 'neg_log_loss',
    n_jobs = 4, 
    cv = 3, 
    refit = False,
    verbose = 2)

grid_search.fit(X_train, y_train)
pd.DataFrame(grid_search.cv_results_).to_csv('results2.csv', index = False)

# Third hyperparameter search (alpha, balanced/None, and smaller alpha range)

param_grid = {
    'loss': ['modified_huber'],
    'penalty': ['l2'], 
    'alpha': [0.075, 0.1, 0.125], 
    'l1_ratio': [0.15], 
    'fit_intercept': [False], 
    'max_iter': [3000], 
    'tol': [1e-2], 
    'shuffle': [True], 
    'epsilon': [0.1], 
    'random_state': [2019], 
    'learning_rate': ['optimal'],
    'eta0': [0.0], 
    'power_t': [0.5], 
    'class_weight': ['balanced', None]
}

grid_search = GridSearchCV(
    estimator = linear_model.SGDClassifier(), 
    param_grid = param_grid,
    scoring = 'neg_log_loss',
    n_jobs = -1, 
    cv = 5, 
    refit = False,
    verbose = 2)

grid_search.fit(X_train, y_train)
pd.DataFrame(grid_search.cv_results_).to_csv('results2.csv', index = False)