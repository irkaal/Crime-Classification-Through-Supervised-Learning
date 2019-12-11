import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from zipfile import ZipFile

# Load Data
train_data = pd.read_csv(ZipFile('data/processed/train.zip').open('train.csv'))
X_train = train_data.drop(['Category', '2010-2012', '0600-1759', 'PdDistrict_TARAVAL', 'Patrol_Division',
                           'Polar_Rho', 'Polar_Phi', 'X_R30', 'Y_R30', 'X_R60', 'Y_R60', 'XY_PCA1', 'XY_PCA2'], axis = 1)
category = pd.factorize(train_data['Category'], sort = True)
y_train = category[0]

# Cross Validation
rskf = RepeatedStratifiedKFold(n_splits = 2, n_repeats = 1, random_state = 2019)
pipe = make_pipeline(StandardScaler(), 
                     SGDClassifier())
calibrated_pipe = make_pipeline(StandardScaler(), 
                                CalibratedClassifierCV(base_estimator = SGDClassifier(), 
                                                       method = 'sigmoid', 
                                                       cv = 2))

# Logistic Loss
parameters = {
    'sgdclassifier__loss': ['log'],
    'sgdclassifier__penalty': ['l2', 'l1'],
    'sgdclassifier__fit_intercept': [True, False], 
    'sgdclassifier__alpha': [1e-4, 1e-3, 1e-2, 1e-1],  
    'sgdclassifier__max_iter': [10000], 
    'sgdclassifier__random_state': [2019]
}
grid_search = GridSearchCV(estimator = pipe, 
                           param_grid = parameters, 
                           scoring = 'neg_log_loss',
                           cv = rskf, 
                           refit = False,
                           n_jobs = -1, 
                           verbose = 2)
grid_search.fit(X_train, y_train)
result = pd.DataFrame(grid_search.cv_results_)
print(result[['params', 'mean_test_score', 'rank_test_score']])

# Modified Huber Loss
parameters = {
    'sgdclassifier__loss': ['modified_huber'],
    'sgdclassifier__penalty': ['l2'], 
    'sgdclassifier__alpha': [0.01], 
    'sgdclassifier__fit_intercept': [False], 
    'sgdclassifier__max_iter': [5000],
    'sgdclassifier__random_state': [2019]
}
pipe = make_pipeline(StandardScaler(), 
                     SGDClassifier())
grid_search = GridSearchCV(estimator = pipe, 
                           param_grid = parameters, 
                           scoring = 'neg_log_loss',
                           cv = rskf, 
                           refit = False,
                           n_jobs = -1, 
                           verbose = 2)
grid_search.fit(X_train, y_train)
result = pd.DataFrame(grid_search.cv_results_)
print(result[['params', 'mean_test_score', 'rank_test_score']])

# Hinge Loss
parameters = {
    'calibratedclassifiercv__base_estimator__loss': ['hinge'],
    'calibratedclassifiercv__base_estimator__alpha': [1e0], 
    'calibratedclassifiercv__base_estimator__penalty': ['l2'],
    'calibratedclassifiercv__base_estimator__tol': [1e-3],
    'calibratedclassifiercv__base_estimator__fit_intercept': [True], 
    'calibratedclassifiercv__base_estimator__random_state': [2019],
    'calibratedclassifiercv__base_estimator__max_iter': [5000]
}

grid_search = GridSearchCV(estimator = calibrated_pipe, 
                           param_grid = parameters, 
                           scoring = 'neg_log_loss',
                           cv = rskf, 
                           refit = False,
                           n_jobs = -1,
                           verbose = 2)
grid_search.fit(X_train, y_train)
result = pd.DataFrame(grid_search.cv_results_)
print(result[['params', 'mean_test_score', 'rank_test_score']])

# Prediction 
scaler = StandardScaler().fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train), columns = X_train.columns)

X_test = pd.read_csv(ZipFile('data/processed/test.zip').open('test.csv'))
crime_id = X_test['Id']
X_test = X_test.drop(['Id', '2010-2012', '0600-1759', 'PdDistrict_TARAVAL', 'Patrol_Division',
                      'Polar_Rho', 'Polar_Phi', 'X_R30', 'Y_R30', 'X_R60', 'Y_R60', 'XY_PCA1', 'XY_PCA2'], axis = 1)
X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)

# Logistic Loss
logistic_clf = SGDClassifier(loss = 'log', 
                             alpha = 1e-1, 
                             max_iter = 5000, 
                             random_state = 2019, 
                             verbose = 2, 
                             n_jobs = -1)
logistic_clf.fit(X_train, y_train)
pred_proba = logistic_clf.predict_proba(X_test)
result = pd.DataFrame(pred_proba)
result.insert(0, 'Id', crime_id)
column_names = np.insert(category[1], 0, 'Id')
result.columns = column_names
result.to_csv('logistic_sgd_submission.csv', index = False)

# Modified Huber Loss
huber_clf = SGDClassifier(loss = 'modified_huber', 
                          alpha = 1e-2, 
                          fit_intercept = False, 
                          max_iter = 5000, 
                          random_state = 2019, 
                          verbose = 2, 
                          n_jobs = -1)
huber_clf.fit(X_train, y_train)
pred_proba = huber_clf.predict_proba(X_test)
result = pd.DataFrame(pred_proba)
result.insert(0, 'Id', crime_id)
column_names = np.insert(category[1], 0, 'Id')
result.columns = column_names
result.to_csv('huber_sgd_submission.csv', index = False)

# Hinge Loss
clf = CalibratedClassifierCV(SGDClassifier(loss = 'hinge', 
                                           penalty = 'l2', 
                                           tol = 1e-3, 
                                           alpha = 1e0, 
                                           max_iter = 5000, 
                                           random_state = 2019,
                                           verbose = 2, 
                                           n_jobs = -1), 
                             method = 'sigmoid', 
                             cv = 2)
clf.fit(X_train, y_train)
pred_proba = clf.predict_proba(X_test)
result = pd.DataFrame(pred_proba)
result.insert(0, 'Id', crime_id)
column_names = np.insert(category[1], 0, 'Id')
result.columns = column_names
result.to_csv('hinge_sgd_submission.csv', index = False)
