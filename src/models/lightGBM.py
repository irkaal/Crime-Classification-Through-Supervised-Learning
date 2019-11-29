import pandas as pd
import numpy as np
import lightgbm as lgb
import data_cleaning
import csv
import os
import random
# Load clean data
train_data_clean = pd.read_csv('data/clean_final.csv')
# Get the labels from the data
label = train_data_clean['Category']
# Drop the labels from the training data
data = train_data_clean.drop('Category', axis=1)
#Convert to lgb dataset to be used by lightGBM
train_data = lgb.Dataset(data, label=label)



N_FOLDS = 3
MAX_EVALS = 20

# X_X_X_X
# NFOLDS_MAXEVALS_BOOSTROUNDS_EARLY 

# Create lightGBM dataset
train_data = lgb.Dataset(data, label=label)

#cv_results = lgb.cv(param_grid, train_data, nfold=N_FOLDS, num_boost_round=100,
#    early_stopping_rounds=10, metrics='multi_logloss', seed=2019)

# Find the best logloss
#best_logloss = max(cv_results['multi_logloss-mean'])

# Loss
#loss = best_logloss

# Boosting round with best cv score
#n_estimators = int(np.argmin(cv_results['multi_logloss-mean'])+1)


#model = lgb.LGBMClassifier()
#model.get_params()

# Random Search 
{'boosting_type': 'gbdt', 
'class_weight': None, 
'colsample_bytree': 1.0, 
'importance_type': 'split', 
'learning_rate': 0.1, 
'max_depth': -1, 
'min_child_samples': 20, 
'min_child_weight': 0.001, 
'min_split_gain': 0.0, 
'n_estimators': 100, 
'n_jobs': -1, 
'num_leaves': 31, 
'objective': None, 
'random_state': None, 
'reg_alpha': 0.0, 
'reg_lambda': 0.0, 
'silent': True, 
'subsample': 1.0, 
'subsample_for_bin': 200000, 
'subsample_freq': 0}

param_grid = {
    #'boosting_type': 'gbdt',
    #'objective': 'softmax',
   # 'metric': 'multi_logloss',
    #'num_classes': 39,
    'num_leaves': list(range(20, 150)),
    'learning_rate': list(np.logspace(np.log10(0.005), np.log10(0.2), base = 10, num = 1000)),
    'subsample_for_bin': list(range(20000, 300000, 20000)),
    'min_child_samples': list(range(20, 500, 5)),
    'reg_alpha': list(np.linspace(0, 1)),
    'reg_lambda': list(np.linspace(0, 1)),
    'colsample_bytree': list(np.linspace(0.6, 1, 10)),
    'subsample': list(np.linspace(0.7, 1, 100))
}

def objective(hyperparameters, iteration):
    """Objective function for grid and random search. Returns
       the cross validation score from a set of hyperparameters."""
    
    # Number of estimators will be found using early stopping
    if 'n_estimators' in hyperparameters.keys():
        del hyperparameters['n_estimators']
    
     # Perform n_folds cross validation
    cv_results = lgb.cv(hyperparameters, train_data, num_boost_round = 1000, nfold = N_FOLDS, 
                        early_stopping_rounds = 100, metrics = 'multi_logloss', seed = 42)
    
    # results to retun
    best_logloss = min(cv_results['multi_logloss-mean'])
    estimators = len(cv_results['multi_logloss-mean'])
    hyperparameters['n_estimators'] = estimators 
    
    return [best_logloss, hyperparameters, iteration]

def random_search(param_grid, max_evals = MAX_EVALS):
    """Random search for hyperparameter optimization"""
    
    # Dataframe for results
    results = pd.DataFrame(columns = ['best_logloss', 'params', 'iteration'],
                                  index = list(range(MAX_EVALS)))
    
    # Keep searching until reach max evaluations
    for i in range(MAX_EVALS):
        
        # Choose random hyperparameters
        hyperparameters = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
        hyperparameters['objective'] = 'softmax'
        hyperparameters['num_classes'] = 39
        hyperparameters['metric'] = 'multi_logloss'
        hyperparameters['boosting_type'] ='gbdt'
        


        # Evaluate randomly selected hyperparameters
        eval_results = objective(hyperparameters, i)
        
        results.loc[i, :] = eval_results
    
    # open connection (append option) and write results
        of_connection = open(out_file, 'a')
        writer = csv.writer(of_connection)
        writer.writerow(eval_results)
        
        # make sure to close connection
        of_connection.close()

    # Sort with best score on top
    results.sort_values('best_logloss', ascending = False, inplace = True)
    results.reset_index(inplace = True)
    return results 

#Run 
# Create file and open connection
out_file = 'random_search_trials.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write column names
headers = ['score', 'hyperparameters', 'iteration']
writer.writerow(headers)
of_connection.close()

random_results = random_search(param_grid)



random_params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}








num_leaves_distrib = hp.choice('num_leaves', np.arange(7, 4096, dtype = int))
max_depth_distrib = hp.choice('max_depth', np.arange(2, 64, dtype = int))
learning_rate_distrib = hp.uniform(
    'learning_rate', 0.001, 0.03)
subsample_for_bin_distrib = hp.quniform(
    'subsample_for_bin', 20000, 300000, 20000)
min_child_samples_distrib = hp.quniform('min_child_samples', 20, 500, 5)
reg_alpha_distrib = hp.uniform('reg_alpha', 0.0, 1.0)
reg_lambda_distrib = hp.uniform('reg_lambda', 0.0, 1.0)

space = {
    'num_leaves': num_leaves_distrib,
    'max_depth': max_depth_distrib, 
    'learning_rate': learning_rate_distrib,
    'subsample_for_bin': subsample_for_bin_distrib,
    'min_child_samples': min_child_samples_distrib,
    'reg_alpha': reg_alpha_distrib,
    'reg_lambda': reg_lambda_distrib
}
 



