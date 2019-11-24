import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss

# Load Data
train_data = pd.read_csv('data/clean_centered_final.csv')
y_train = train_data['Category'].astype('category')
X_train = train_data.drop('Category', axis = 1)

# repeatedcv for validation score
skf = RepeatedStratifiedKFold(n_splits = 3, n_repeats = 5, random_state = 2019)

# To obtain log loss using predict_proba:
# y_pred_proba = grid_search.best_estimator_.predict_proba(X_train)
# log_loss(y_train, y_pred_proba)

# Hinge Loss with calibrated cv
parameters = {
    'base_estimator__loss': ['hinge'],
    'base_estimator__penalty': ['l2'],
     'base_estimator__l1_ratio': [0.15],
    'base_estimator__alpha': [0.001], 
    'base_estimator__max_iter': [5000], 
    'base_estimator__fit_intercept': [True],
    'base_estimator__tol': [1e-3], 
    'base_estimator__random_state': [2019], 
    'base_estimator__class_weight': [None]
}
grid_search = GridSearchCV(estimator = CalibratedClassifierCV(base_estimator = SGDClassifier(), method = 'sigmoid', cv = 3), 
                           param_grid = parameters, scoring = 'neg_log_loss', cv = skf, n_jobs = -1,  refit = False, verbose = 2)
grid_search.fit(X_train, y_train)
pd.set_option('display.max_columns', 30)
pd.DataFrame(grid_search.cv_results_)

# Logistic Loss
parameters = {
    'loss': ['log'],
    'penalty': ['l2'], 
    'alpha': [0.1], 
    'fit_intercept': [True], 
    'max_iter': [5000], 
    'tol': [1e-3], 
    'random_state': [2019], 
    'class_weight': [None]
}
grid_search2 = GridSearchCV(estimator = SGDClassifier(), param_grid = parameters, scoring = 'neg_log_loss',
                            cv = skf, n_jobs = -1, refit = False, verbose = 2)
grid_search2.fit(X_train, y_train)
pd.set_option('display.max_columns', 30)
pd.DataFrame(grid_search2.cv_results_)

# Modified Huber Loss
parameters = {
    'loss': ['modified_huber'],
    'penalty': ['l2'], 
    'alpha': [0.01], 
    'fit_intercept': [False], 
    'max_iter': [5000], 
    'tol': [1e-3], 
    'random_state': [2019], 
    'class_weight': [None]
}
grid_search3 = GridSearchCV(estimator = SGDClassifier(), param_grid = parameters, scoring = 'neg_log_loss',
                            cv = skf, n_jobs = -1, refit = False, verbose = 2)
grid_search3.fit(X_train, y_train)
pd.set_option('display.max_columns', 30)
pd.DataFrame(grid_search3.cv_results_)

# Perceptron with calibrated cv
parameters = {
    'base_estimator__loss': ['perceptron'],
    'base_estimator__penalty': ['l2'],
    'base_estimator__alpha': [0.01], 
    'base_estimator__max_iter': [1000], 
    'base_estimator__fit_intercept': [False],
    'base_estimator__tol': [1e-3], 
    'base_estimator__random_state': [2019], 
    'base_estimator__class_weight': [None]
}
grid_search4 = GridSearchCV(estimator = CalibratedClassifierCV(base_estimator = SGDClassifier(), method = 'sigmoid', cv = 3), 
                            param_grid = parameters, scoring = 'neg_log_loss', cv = skf, n_jobs = 7,  refit = False, verbose = 2)
grid_search4.fit(X_train, y_train)
pd.set_option('display.max_columns', 30)
pd.DataFrame(grid_search4.cv_results_)

# Squared Hinge Loss with calibrated cv
parameters = {
    'base_estimator__loss': ['squared_hinge'],
    'base_estimator__penalty': ['l2'],
    'base_estimator__alpha': [0.01], 
    'base_estimator__max_iter': [1000], 
    'base_estimator__fit_intercept': [False],
    'base_estimator__tol': [1e-3], 
    'base_estimator__random_state': [2019], 
    'base_estimator__class_weight': [None]
}
grid_search5 = GridSearchCV(estimator = CalibratedClassifierCV(base_estimator = SGDClassifier(), method = 'sigmoid', cv = 3), 
                            param_grid = parameters, scoring = 'neg_log_loss', cv = skf, n_jobs = 7,  refit = False, verbose = 2)
grid_search5.fit(X_train, y_train)
pd.set_option('display.max_columns', 30)
pd.DataFrame(grid_search5.cv_results_)
