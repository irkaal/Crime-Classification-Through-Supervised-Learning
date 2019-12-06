import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from zipfile import ZipFile


# Load Data
train_data = pd.read_csv(ZipFile('data/processed/train.zip').open('train.csv'))
pd.set_option('display.max_columns', 30)

# Calibrated Classifiers
calibrated_LinearSVC = CalibratedClassifierCV(base_estimator = LinearSVC(), method = 'sigmoid', cv = 2)
calibrated_LinearSVC_SGD = CalibratedClassifierCV(base_estimator = SGDClassifier(), method = 'sigmoid', cv = 2)


# Hinge Loss
X_train = train_data.drop(['Category',
                           '2010-2012', '0600-1759', 
                           'PdDistrict_TARAVAL', 'Patrol_Division',
                           'Polar_Rho', 'Polar_Phi', 'X_R30', 'Y_R30', 'X_R60', 'Y_R60', 'XY_PCA1', 'XY_PCA2'], axis = 1)
X_train_columns = X_train.columns
scaler = StandardScaler().fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train), columns = X_train_columns)
category = pd.factorize(train_data['Category'], sort = True)
y_train = category[0]

skf = RepeatedStratifiedKFold(n_splits = 2, n_repeats = 1, random_state = 2019)
parameters = {
    'base_estimator__loss': ['hinge'],
    'base_estimator__penalty': ['l2'],
    'base_estimator__C': [1e-2, 1e-1],
    'base_estimator__class_weight': [None, 'balanced'],
    'base_estimator__fit_intercept': [True, False], 
    'base_estimator__random_state': [2019],
    'base_estimator__max_iter': [1000000] 
}
grid_search = GridSearchCV(estimator = calibrated_LinearSVC, param_grid = parameters, scoring = 'neg_log_loss',
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
clf = CalibratedClassifierCV(LinearSVC(loss = 'hinge', penalty = 'l2', C = 0.1, fit_intercept = False, class_weight='balanced', 
                                       verbose = 2, random_state = 2019, max_iter = 1000000), method = 'sigmoid', cv = 2)
clf.fit(X_train, y_train)
pred_proba = clf.predict_proba(X_test)
result = pd.DataFrame(pred_proba)
result.insert(0, 'Id', crime_id)
column_names = np.insert(category[1], 0, 'Id')
result.columns = column_names
result.to_csv('submission.csv', index = False)



# Squared Hinge Loss
X_train = train_data.drop(['Category',
                           '2010-2012', '0600-1759', 
                           'PdDistrict_TARAVAL', 'Patrol_Division',
                           'Polar_Rho', 'Polar_Phi', 'X_R30', 'Y_R30', 'X_R60', 'Y_R60', 'XY_PCA1', 'XY_PCA2'], axis = 1)
X_train_columns = X_train.columns
scaler = StandardScaler().fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train), columns = X_train_columns)
category = pd.factorize(train_data['Category'], sort = True)
y_train = category[0]

skf = RepeatedStratifiedKFold(n_splits = 2, n_repeats = 1, random_state = 2019)
calibrated_LinearSVC = CalibratedClassifierCV(base_estimator = LinearSVC(), method = 'sigmoid', cv = 2)
parameters = {
    'base_estimator__loss': ['squared_hinge'],
    'base_estimator__penalty': ['l2'],
    'base_estimator__C': [1e-2],
    'base_estimator__class_weight': [None],
    'base_estimator__fit_intercept': [True], 
    'base_estimator__random_state': [2019],
    'base_estimator__max_iter': [1000000] 
}
grid_search = GridSearchCV(estimator = calibrated_LinearSVC, param_grid = parameters, scoring = 'neg_log_loss',
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
clf = CalibratedClassifierCV(LinearSVC(loss = 'squared_hinge', penalty = 'l2', C = 1e-2, fit_intercept = True, 
                                       verbose = 2, random_state = 2019, max_iter = 1000000), method = 'sigmoid', cv = 2)
clf.fit(X_train, y_train)
pred_proba = clf.predict_proba(X_test)
result = pd.DataFrame(pred_proba)
result.insert(0, 'Id', crime_id)
column_names = np.insert(category[1], 0, 'Id')
result.columns = column_names
result.to_csv('submission.csv', index = False)



# Hinge Loss with SGD
X_train = train_data.drop(['Category',
                           '2010-2012', '0600-1759', 
                           'PdDistrict_TARAVAL', 'Patrol_Division',
                           'Polar_Rho', 'Polar_Phi', 'X_R30', 'Y_R30', 'X_R60', 'Y_R60', 'XY_PCA1', 'XY_PCA2'], axis = 1)
X_train_columns = X_train.columns
scaler = StandardScaler().fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train), columns = X_train_columns)
category = pd.factorize(train_data['Category'], sort = True)
y_train = category[0]

skf = RepeatedStratifiedKFold(n_splits = 3, n_repeats = 2, random_state = 2019)
parameters = {
    'base_estimator__loss': ['hinge'],
    'base_estimator__alpha': [1e0], 
    'base_estimator__penalty': ['l2'],
    'base_estimator__tol': [1e-3],
    'base_estimator__fit_intercept': [True], 
    'base_estimator__random_state': [2019],
    'base_estimator__max_iter': [5000]
}
grid_search = GridSearchCV(estimator = calibrated_LinearSVC_SGD, param_grid = parameters, scoring = 'neg_log_loss',
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
clf = CalibratedClassifierCV(SGDClassifier(loss = 'hinge', penalty = 'l2', tol = 1e-3, alpha = 1e0, max_iter = 5000, 
                                           verbose = 2, random_state = 2019), method = 'sigmoid', cv = 2)
clf.fit(X_train, y_train)
pred_proba = clf.predict_proba(X_test)
result = pd.DataFrame(pred_proba)
result.insert(0, 'Id', crime_id)
column_names = np.insert(category[1], 0, 'Id')
result.columns = column_names
result.to_csv('submission.csv', index = False)
