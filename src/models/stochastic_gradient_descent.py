import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from zipfile import ZipFile


# Load Data
train_data = pd.read_csv(ZipFile('data/processed/train.zip').open('train.csv'))
X_train = train_data.drop(['Category', 
                           '2010-2012', '0600-1759', 
                           'PdDistrict_TARAVAL', 'Patrol_Division',
                           'Polar_Rho', 'Polar_Phi', 'X_R30', 'Y_R30', 'X_R60', 'Y_R60', 'XY_PCA1', 'XY_PCA2'], axis = 1)
X_train_columns = X_train.columns
scaler = StandardScaler().fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train), columns = X_train_columns)
category = pd.factorize(train_data['Category'], sort = True)
y_train = category[0]

pd.set_option('display.max_columns', 30)
skf = RepeatedStratifiedKFold(n_splits = 2, n_repeats = 3, random_state = 2019)


# Logistic Loss
parameters = {
    'loss': ['log'],
    'penalty': ['l2', 'l1'],
    'fit_intercept': [True, False], 
    'alpha': [1e-4, 1e-3, 1e-2, 1e-1],  
    'max_iter': [10000], 
    'random_state': [2019]
}

# Log loss measure
grid_search = GridSearchCV(estimator = SGDClassifier(), param_grid = parameters, scoring = 'neg_log_loss',
                           cv = skf, n_jobs = -1, refit = False, verbose = 2)
grid_search.fit(X_train, y_train)
print(pd.DataFrame(grid_search.cv_results_))



# Modified Huber Loss
parameters = {
    'loss': ['modified_huber'],
    'penalty': ['l2'], 
    'alpha': [0.01], 
    'fit_intercept': [False], 
    'max_iter': [5000],
    'random_state': [2019]
}

# Log loss measure
grid_search = GridSearchCV(estimator = SGDClassifier(), param_grid = parameters, scoring = 'neg_log_loss',
                           cv = skf, n_jobs = -1, refit = False, verbose = 2)
grid_search.fit(X_train, y_train)
print(pd.DataFrame(grid_search.cv_results_))



# Prediction
X_test = pd.read_csv(ZipFile('data/processed/test.zip').open('test.csv'))
crime_id = X_test['Id']
X_test = X_test.drop(['Id', 
                     '2010-2012', '0600-1759', 
                     'PdDistrict_TARAVAL', 'Patrol_Division',
                     'Polar_Rho', 'Polar_Phi', 'X_R30', 'Y_R30', 'X_R60', 'Y_R60', 'XY_PCA1', 'XY_PCA2'], axis = 1)
X_test_columns = X_test.columns
X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test_columns)

# Logistic loss
clf = SGDClassifier(loss='log', alpha=1e-1, max_iter=5000, random_state=2019, verbose=2, n_jobs=-1)
clf.fit(X_train, y_train)
pred_proba = clf.predict_proba(X_test)
result = pd.DataFrame(pred_proba)
result.insert(0, 'Id', crime_id)
column_names = np.insert(category[1], 0, 'Id')
result.columns = column_names
result.to_csv('submission.csv', index = False)

# Modified huber loss
clf = SGDClassifier(loss='modified_huber', alpha=1e-2, fit_intercept=False, max_iter=5000, random_state=2019, verbose=2, n_jobs=-1)
clf.fit(X_train, y_train)
pred_proba = clf.predict_proba(X_test)
result = pd.DataFrame(pred_proba)
result.insert(0, 'Id', crime_id)
column_names = np.insert(category[1], 0, 'Id')
result.columns = column_names
result.to_csv('submission.csv', index = False)
