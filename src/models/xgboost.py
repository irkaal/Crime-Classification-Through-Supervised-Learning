import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from hyperopt import hp, fmin, tpe, STATUS_OK

# Load train data
train_data = pd.read_csv('data/clean_final.csv')
y_train = train_data['Category'].astype('category')
X_train = train_data.drop('Category', axis = 1)
dtrain = xgb.DMatrix(data = X_train, label = y_train)


# Hyperparameter search with Bayesian Optimization

# out_file = 'gbm_trials.csv'
# of_connection = open(out_file, 'w')
# writer = csv.writer(of_connection)
# writer.writerow(['loss', 'params', 'iteration', 'estimators'])
# of_connection.close()

def score(params):
    num_boost = int(params['n_estimators'])
    del params['n_estimators']
    best = xgb.cv(params, dtrain, num_boost_round = num_boost, 
                  nfold = 2, stratified = True, metrics = 'mlogloss')
    loss = best['test-mlogloss-mean'].tail(1)
    
    # Save result
    # of_connection = open(out_file, 'a')
    # writer = csv.writer(of_connection)
    # writer.writerow([loss, params, ITERATION, n_estimators])
    # of_connection.close()
    
    return {'loss': loss, 'status': STATUS_OK}
 
def optimize(max_evals):
    space = {
        'n_estimators': hp.quniform('n_estimators', 200, 600, 1),
        'eta': hp.quniform('eta', 0.025, 0.25, 0.025),
        'max_depth':  hp.choice('max_depth', np.arange(1, 14, dtype=int)),
        'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
        'subsample': hp.quniform('subsample', 0.7, 1, 0.05),
        'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.7, 1, 0.05),
        'alpha' :  hp.quniform('alpha', 0, 10, 1),
        'lambda': hp.quniform('lambda', 1, 2, 0.1),
        'objective': 'multi:softmax', 
        'num_class': 39, 
        'seed': 2019
    }
    best = fmin(score, space, algo = tpe.suggest, max_evals = max_evals)
    return best
best_param = optimize(max_evals = 10)
print(best_param)


# Hyperparameter grid search
skf = RepeatedStratifiedKFold(n_splits = 3, n_repeats = 5, random_state = 2019)
parameters = {
    'n_estimators': [100],
    'objective': ['multi:softmax'],
    'verbosity': [1],
    'n_jobs': [-1],
    'random_state': [2019]
}
grid_search = GridSearchCV(estimator = xgb.XGBClassifier(), param_grid = parameters, scoring = 'neg_log_loss',
                            cv = skf, n_jobs = -1, refit = False, verbose = 1)                            
grid_search.fit(X_train, y_train)
pd.set_option('display.max_columns', 30)
print(pd.DataFrame(grid_search.cv_results_))
