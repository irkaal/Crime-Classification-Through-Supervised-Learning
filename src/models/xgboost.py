import csv
import numpy as np
import pandas as pd
import xgboost as xgb
from hyperopt import hp, fmin, tpe, STATUS_OK
from zipfile import ZipFile


# Load Data
train_data = pd.read_csv(ZipFile('data/processed/train.zip').open('train.csv'))
X_train = train_data.drop('Category', axis = 1)
category = pd.factorize(train_data['Category'], sort = True)
y_train = category[0]
dtrain = xgb.DMatrix(data = X_train, 
                     label = y_train)


# Bayesian Optimization
out_file = 'xgb_trials.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)
writer.writerow(['loss', 'params', 'iteration'])
of_connection.close()
ITERATION = 0

def score(params):
    best = xgb.cv(params, dtrain, num_boost_round = 700, early_stopping_rounds = 50, 
                  nfold = 2, stratified = True, metrics = 'mlogloss', seed = 2019)
    loss = best['test-mlogloss-mean'].values[-1]
    global ITERATION
    ITERATION += 1
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, params, ITERATION])
    of_connection.close()
    return {'loss': loss, 'status': STATUS_OK}

def optimize(max_evals):
    space = {
        'max_depth': hp.choice('max_depth', [11, 12, 13, 14]),
        'subsample': hp.quniform('subsample', 0.6, 0.8, 0.05),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.7, 0.9, 0.05),
        # 'tree_method': 'hist',
        'eta': hp.quniform('eta', 0.04, 0.06, 0.01),
        'min_child_weight': hp.quniform('min_child_weight', 2, 4, 1),
        'alpha' : hp.quniform('alpha', 4, 6, 1),
        'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
        'lambda': hp.quniform('lambda', 1.5, 2, 0.1),
        'objective': 'multi:softmax', 
        'num_class': 39, 
        'seed': 2019
    }
    best_param = fmin(score, space, algo = tpe.suggest, max_evals = max_evals)
    return best_param

best_param = optimize(max_evals = 50)


# Predict using final model
X_test = pd.read_csv(ZipFile('data/processed/test.zip').open('test.csv'))
crime_id = X_test['Id']
X_test = X_test.drop('Id', axis = 1)
clf = xgb.XGBClassifier(max_depth=12, learning_rate=0.05, n_estimators=505, objective='multi:softmax', gamma=0.75, verbosity=2, 
                        min_child_weight=3, subsample=0.75, colsample_bytree=0.8, reg_alpha=5, reg_lambda=1.8, seed=2019, n_jobs=-1)
clf.fit(X_train, y_train)
pred_proba = clf.predict_proba(X_test)


# Save results
result = pd.DataFrame(pred_proba)
result.insert(0, 'Id', crime_id)
column_names = np.insert(category[1], 0, 'Id')
result.columns = column_names
result.to_csv('submission.csv', index = False)
