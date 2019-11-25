import csv
import numpy as np
import pandas as pd
import xgboost as xgb
from hyperopt import hp, fmin, tpe, STATUS_OK
from zipfile import ZipFile

# Load Data
train_data = pd.read_csv(ZipFile('data/processed/train.zip').open('train.csv'))
dtrain = xgb.DMatrix(data = train_data.drop('Category', axis = 1), 
                     label = train_data['Category'].values)

# Bayesian Optimization
out_file = 'xgb_trials.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)
writer.writerow(['loss', 'params', 'iteration', 'estimators'])
of_connection.close()
ITERATION = 0

def score(params):
    nbr = int(params['n_estimators'])
    del params['n_estimators']
    best = xgb.cv(params, dtrain, num_boost_round = nbr, nfold = 2, stratified = True, metrics = 'mlogloss', seed = 2019)
    loss = best['test-mlogloss-mean'].tail(1)
    # Save result
    global ITERATION
    ITERATION += 1
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, params, ITERATION, nbr])
    of_connection.close()
    return {'loss': loss, 'status': STATUS_OK}

def optimize(max_evals):
    space = {
        'n_estimators': hp.quniform('n_estimators', 500, 1000, 1),
        'tree_method': 'hist',
        'eta': hp.quniform('eta', 0.05, 0.2, 0.025),
        'max_depth':  hp.choice('max_depth', np.arange(3, 13, dtype=int)),
        'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
        'subsample': hp.quniform('subsample', 0.8, 1, 0.05),
        'gamma': hp.choice('gamma', np.array([0, 1, 5])),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.3, 1, 0.05),
        'alpha' : hp.quniform('alpha', 0, 10, 1),
        'lambda': hp.quniform('lambda', 1, 2, 0.1),
        'objective': 'multi:softmax', 
        'num_class': 39, 
        'seed': 2019
    }
    best_param = fmin(score, space, algo = tpe.suggest, max_evals = max_evals)
    return best_param

best_param = optimize(max_evals = 50)

# params = {
#     'alpha': 5.0,
#     'colsample_bytree': 0.8,
#     'eta': 0.05,
#     'gamma': 0.75,
#     'lambda': 1.8,
#     'max_depth': 12,
#     'min_child_weight': 3.0,
#     'n_estimators': 536.0,
#     'subsample': 0.75
# }