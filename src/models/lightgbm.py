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
from zipfile import ZipFile

# Load clean data
train_data_clean = pd.read_csv('data/clean_final.csv')
# Get the labels from the data
label = train_data_clean['Category']
# Drop the labels from the training data
data = train_data_clean.drop('Category', axis=1)
X_train = data; 
#Convert to lgb dataset to be used by lightGBM
train_data = lgb.Dataset(data, label=label)

category = pd.factorize(train_data_clean['Category'], sort = True)
y_train = category[0]


N_FOLDS = 3

# Create lightGBM dataset
#train_data = lgb.Dataset(data, label=label)

# Define the objective function (minimize validation error)
ITERATION = 0


def objective(params, n_folds=N_FOLDS):
    global ITERATION

    ITERATION += 1
    # Make sure parameters that need to be integers are integers
    for parameter_name in ['subsample_for_bin', 'min_child_samples']:
        params[parameter_name] = int(params[parameter_name])

    params['objective'] = 'softmax'
    params['num_classes'] = 39
    params['metric'] = 'multi_logloss'
    #params['min_data_in_leaf'] = 15
    # n_fold cross validation with hyperparameters
    # Use early stopping and evaluate based on softmax

    # cv_results = lgb.cv(params, train_data, nfold=n_folds, num_boost_round=1000,
    #                   early_stopping_rounds=100, metrics='multi_logloss', seed=2019)

    cv_results = lgb.cv(params, train_data, nfold=n_folds, num_boost_round=1000,
                        early_stopping_rounds=100, metrics='multi_logloss', seed=2019)

    # Find the best logloss
    best_logloss = min(cv_results['multi_logloss-mean'])

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


#num_leaves_distrib = hp.choice('num_leaves', np.arange(7, 4096, dtype = int)) #Better one
#max_depth_distrib = hp.quniform('max_depth', 2, 63,1)
max_depth_distrib = hp.choice('max_depth', np.arange(1, 20, dtype = int))

learning_rate_distrib = hp.quniform(
    'learning_rate', 0.1, 0.3, 0.01)
subsample_for_bin_distrib = hp.quniform(
    'subsample_for_bin', 20000, 300000, 20000)
min_child_samples_distrib = hp.quniform('min_child_samples', 20, 250, 5)
reg_alpha_distrib = hp.quniform('reg_alpha', 0.0, 10, 1)
reg_lambda_distrib = hp.uniform('reg_lambda', 0, 10)

colsample_by_tree_distrib = hp.uniform('colsample_by_tree', 0.6, 1.0)
min_child_weight_distrib = hp.quniform('min_child_weight_distrib', 1, 20, 1)

#learning_rate_distrib = hp.loguniform(learning_rate', np.log(0.01), np.log(0.2))

# Define the search space (Omitted boosting_type dart and goss)
space = {
    'colsample_bytree': colsample_by_tree_distrib,
    'max_depth': max_depth_distrib, 
    'learning_rate': learning_rate_distrib,
    'subsample_for_bin': subsample_for_bin_distrib,
    'min_child_samples': min_child_samples_distrib,
    'reg_alpha': reg_alpha_distrib,
    'reg_lambda': reg_lambda_distrib,
    'min_child_weight': min_child_weight_distrib
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


MAX_EVALS = 7

# Optimize
best = fmin(fn=objective, space=space, algo=tpe.suggest,
            max_evals=MAX_EVALS, return_argmin=False)


# My max 2.58
params = {'colsample_bytree': 0.8728331295897298, 'learning_rate': 0.19994149039861425, 
'min_child_samples': 25, 'num_leaves': 150, 'reg_alpha': 0.6158151936471097, 
'reg_lambda': 0.6643855739060315, 'subsample_for_bin': 180000, 'objective': 
'softmax', 'num_classes': 39, 'metric': 'multi_logloss', 'min_data_in_leaf': 15}

results = lgb.cv(params, train_data, nfold=3, num_boost_round=1000,
                        early_stopping_rounds = 100, metrics='multi_logloss', seed=2019)

min(results['multi_logloss-mean'])

# Predict using final model
X_test = pd.read_csv(ZipFile('data/sf-crime.zip').open('test.csv'))
crime_id = X_test['Id']
X_test = X_test.drop('Id', axis = 1)

params['objective'] = 'softmax'
params['num_classes'] = 39
params['metric'] = 'multi_logloss'
d_train = lgb.Dataset(X_train, label=y_train)

# Create Classifier with best parameters
clf = lgb.LGBMClassifier(colsample_bytree= 0.8728331295897298, learning_rate= 0.19994149039861425, 
min_child_samples= 25, num_leaves= 150, reg_alpha= 0.6158151936471097, 
reg_lambda= 0.6643855739060315, subsample_for_bin= 180000, objective= 'softmax', 
num_classes= 39, metric= 'multi_logloss', min_data_in_leaf= 15)

# Fit classifier on training data 
clf.fit(X_train, y_train)

pred_proba = clf.predict_proba(X_test)


# Save results
result = pd.DataFrame(pred_proba)
result.insert(0, 'Id', crime_id)
column_names = np.insert(category[1], 0, 'Id')
result.columns = column_names
result.to_csv('lgbm_submission.csv', index = False)
