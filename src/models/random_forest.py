import csv
import numpy as np
import pandas as pd
from hyperopt import hp, fmin, tpe, STATUS_OK
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from zipfile import ZipFile


# Load Data
train_data = pd.read_csv(ZipFile('data/processed/train.zip').open('train.csv'))
X_train = train_data.drop(['Category'], axis = 1)
category = pd.factorize(train_data['Category'], sort = True)
y_train = category[0]


# Bayesian Optimization
out_file = 'rf_trials.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)
writer.writerow(['loss', 'params'])
of_connection.close()

def score(params):
    clf = RandomForestClassifier(**params)
    clf.fit(X_train, y_train)
    # Calculate oob log loss estimate
    y_pred = clf.oob_decision_function_
    oob_indices = np.unique([i[0] for i in np.argwhere(~np.isnan(y_pred))])
    loss = log_loss(y_train[oob_indices], y_pred[oob_indices])
    
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, params])
    of_connection.close()
    return {'loss': loss, 'status': STATUS_OK}

def optimize(max_evals):
    space = {
        'max_depth': hp.choice('max_depth', [11, 12, 13, 14]), 
        'max_features': 'auto', 
        'max_leaf_nodes': None, 
        'min_impurity_decrease': 0.0, 
        'min_impurity_split': None, 
        'min_samples_leaf': 1, 
        'min_samples_split': 2, 
        'min_weight_fraction_leaf': 0.0, 
        'n_estimators': 100, 
        'n_jobs': 1, 
        'oob_score': True, 
        'random_state': 2019, 
        'verbose': 0, 
        'warm_start': False
    }
    best_param = fmin(score, space, algo = tpe.suggest, max_evals = max_evals)
    return best_param

best_param = optimize(max_evals = 2)


# Predict using final model
X_test = pd.read_csv(ZipFile('data/processed/test.zip').open('test.csv'))
crime_id = X_test['Id']
X_test = X_test.drop(['Id'], axis = 1)
clf = RandomForestClassifier(max_depth = 10, max_features = 'auto', max_leaf_nodes = None, 
                             min_impurity_decrease = 0.0, min_impurity_split = None, 
                             min_samples_leaf = 1, min_samples_split = 2, min_weight_fraction_leaf = 0.0, 
                             n_estimators = 100, n_jobs = -1, random_state = 2019)
clf.fit(X_train, y_train)
pred_proba = clf.predict_proba(X_test)
result = pd.DataFrame(pred_proba)
result.insert(0, 'Id', crime_id)
column_names = np.insert(category[1], 0, 'Id')
result.columns = column_names
result.to_csv('submission.csv', index = False)
