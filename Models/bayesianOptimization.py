from hyperopt import STATUS_OK
import pandas as pd
import numpy as np
import lightgbm as lgb
import data_cleaning
from hyperopt import hp
from hyperopt import tpe
from hyperopt import Trials
import csv
from hyperopt import fmin
import os

# from skopt import gp_minimize

train_data_clean = pd.read_csv('clean_num.csv')

label = train_data_clean['Category']
data = train_data_clean.drop('Category', axis=1)
train_data = lgb.Dataset(data, label=label)

# Advanced (save model as object)
# train_data.save_binary('train.bin')
# 'objective': 'softmax',
#         'num_classes': 39,
#         'metric': 'multi_logloss',
# 'class_weight': [None, 'balanced'],

# Hyperparameter grid
# param = {'objective': 'softmax',
#         'num_classes': 39,
#         'metric': 'multi_logloss'}

# bst = lgb.train(param, train_data)
# ypred = bst.predict(data)

# temp = lgb.cv(param, train_data, nfold=2,
#              num_boost_round=1, metrics='multi_logloss', seed=2019)


N_FOLDS = 3

# Create lightGBM dataset
train_data = lgb.Dataset(data, label=label)

# Define the objective function (minimize validation error)
ITERATION = 0


def objective(params, n_folds=N_FOLDS):
    global ITERATION

    ITERATION += 1
    # Make sure parameters that need to be integers are integers
    for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:
        params[parameter_name] = int(params[parameter_name])

    params['objective'] = 'softmax'
    params['num_classes'] = 39
    params['metric'] = 'multi_logloss'
    params['min_data_in_leaf'] = 15
    # n_fold cross validation with hyperparameters
    # Use early stopping and evaluate based on softmax

    # cv_results = lgb.cv(params, train_data, nfold=n_folds, num_boost_round=1000,
    #                   early_stopping_rounds=100, metrics='multi_logloss', seed=2019)

    cv_results = lgb.cv(params, train_data, nfold=n_folds, num_boost_round=200,
                        early_stopping_rounds=50, metrics='multi_logloss', seed=2019)

    # Find the best logloss
    best_logloss = max(cv_results['multi_logloss-mean'])

    # Loss
    loss = best_logloss

    # Boosting round with best cv score
    n_estimators = int(np.argmin(cv_results['multi_logloss-mean'])+1)

    # # Write to the csv file ('a' means append)
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, params, ITERATION, n_estimators])
    of_connection.close()

    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, "estimators": n_estimators, 'status': STATUS_OK}


num_leaves_distrib = hp.quniform('num_leaves', 30, 150, 1)
learning_rate_distrib = hp.loguniform(
    'learning_rate', np.log(0.01), np.log(0.2))
subsample_for_bin_distrib = hp.quniform(
    'subsample_for_bin', 20000, 300000, 20000)
min_child_samples_distrib = hp.quniform('min_child_samples', 20, 500, 5)
reg_alpha_distrib = hp.uniform('reg_alpha', 0.0, 1.0)
reg_lambda_distrib = hp.uniform('reg_lambda', 0.0, 1.0)
colsample_by_tree_distrib = hp.uniform('colsample_by_tree', 0.6, 1.0)

# Define the search space (Omitted boosting_type dart and goss)
space = {
    'num_leaves': num_leaves_distrib,
    'learning_rate': learning_rate_distrib,
    'subsample_for_bin': subsample_for_bin_distrib,
    'min_child_samples': min_child_samples_distrib,
    'reg_alpha': reg_alpha_distrib,
    'reg_lambda': reg_lambda_distrib,
    'colsample_bytree': colsample_by_tree_distrib
}

# Bayesian optimization algorithm (Tree Parzen Estimator)
tpe_algorithm = tpe.suggest

# Trials object to track progress
bayes_trials = Trials()

# File to save first results
out_file = 'gbm_trials.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# # Write the headers to the file
writer.writerow(['loss', 'params', 'iteration',
                 'estimators'])
of_connection.close()


MAX_EVALS = 100

# Optimize
best = fmin(fn=objective, space=space, algo=tpe.suggest,
            max_evals=MAX_EVALS, trials=bayes_trials,
            return_argmin=False)
