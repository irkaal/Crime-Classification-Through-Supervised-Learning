import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.kernel_approximation import Nystroem
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

# RBF Kernel Approximation (Nystroem Approximation)
feature_map_nystroem = Nystroem(gamma = 1e-4,
                                n_components = 1000, 
                                random_state = 2019)

# Cross Validation
rskf = RepeatedStratifiedKFold(n_splits = 2, n_repeats = 1, random_state = 2019)
calibrated_LinearSVC_SGD = CalibratedClassifierCV(base_estimator = SGDClassifier(), 
                                                  method = 'sigmoid', 
                                                  cv = 2)

parameters = {
    'calibratedclassifiercv__base_estimator__alpha': [1e-4, 1e-3, 1e-2],  
    'calibratedclassifiercv__base_estimator__class_weight': [None, 'balanced'],
    'calibratedclassifiercv__base_estimator__fit_intercept': [True, False],
    'calibratedclassifiercv__base_estimator__max_iter': [100000], 
    'calibratedclassifiercv__base_estimator__random_state': [2019]
}
pipe = make_pipeline(StandardScaler(), 
                     feature_map_nystroem,
                     calibrated_LinearSVC_SGD)
grid_search = GridSearchCV(pipe,
                           param_grid = parameters, 
                           scoring = 'neg_log_loss',
                           cv = rskf, 
                           refit = False,
                           n_jobs = 2, 
                           verbose = 2)
grid_search.fit(X_train, y_train)
result = pd.DataFrame(grid_search.cv_results_)
print(result[['param_calibratedclassifiercv__base_estimator__alpha', 'mean_test_score', 'rank_test_score']])

# Prediction
scaler = StandardScaler()
X_train_transformed = scaler.fit_transform(X_train)
X_train_transformed = feature_map_nystroem.fit_transform(X_train_transformed)
X_train = pd.DataFrame(X_train_transformed)

X_test = pd.read_csv(ZipFile('data/processed/test.zip').open('test.csv'))
crime_id = X_test['Id']
X_test = X_test.drop(['Id', '2010-2012', '0600-1759', 'PdDistrict_TARAVAL', 'Patrol_Division',
                      'Polar_Rho', 'Polar_Phi', 'X_R30', 'Y_R30', 'X_R60', 'Y_R60', 'XY_PCA1', 'XY_PCA2'], axis = 1)
X_test_transformed = scaler.transform(X_test)
X_test_transformed = feature_map_nystroem.transform(X_test)
X_test = pd.DataFrame(X_test_transformed)

clf = CalibratedClassifierCV(SGDClassifier(loss = 'hinge', 
                                           alpha = 1e-3, 
                                           max_iter = 100000, 
                                           random_state = 2019, 
                                           verbose = 2, 
                                           n_jobs = -1), 
                             method = 'sigmoid', 
                             cv = 2)
clf.fit(X_train, y_train)
pred_proba = clf.predict_proba(X_test)
result = pd.DataFrame(pred_proba)
result.insert(0, 'Id', crime_id)
result.columns = np.insert(category[1], 0, 'Id')
result.to_csv('svm_rbf_submission.csv', index = False)
