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
out_file = 'rf_trials.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)
writer.writerow(['loss', 'params', 'iteration'])
of_connection.close()
ITERATION = 0

def score(params):
    best = xgb.cv(params, dtrain, num_boost_round = 1, nfold = 2, stratified = True, metrics = 'mlogloss', seed = 2019)
    loss = best['test-mlogloss-mean'].tail(1)
    # Save result
    global ITERATION
    ITERATION += 1
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, params, ITERATION])
    of_connection.close()
    return {'loss': loss, 'status': STATUS_OK}

def optimize(max_evals):
    space = {
        'tree_method': 'hist',
        'num_parallel_tree': hp.quniform('num_parallel_tree', 200, 500, 100),
        'subsample': hp.quniform('subsample', 0.6, 0.8, 0.1),
        'colsample_bynode': hp.quniform('colsample_bynode', 0.6, 0.8, 0.1),
        'max_depth':  hp.choice('max_depth', np.arange(3, 13, dtype=int)),
        'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
        'gamma': hp.choice('gamma', np.array([0, 1, 5])),
        'alpha' : hp.quniform('alpha', 0, 10, 1),
        'lambda': hp.quniform('lambda', 1, 2, 0.1),
        'objective': 'multi:softmax', 
        'num_class': 39, 
        'eta': 1,
        'seed': 2019
    }
    best_param = fmin(score, space, algo = tpe.suggest, max_evals = max_evals)
    return best_param

best_param = optimize(max_evals = 50)
