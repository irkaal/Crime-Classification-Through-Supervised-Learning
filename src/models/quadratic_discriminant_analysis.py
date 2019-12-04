import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from zipfile import ZipFile


# Load Data
train_data = pd.read_csv(ZipFile('data/processed/train.zip').open('train.csv'))
X_train = train_data.drop(['Category', 
                           '2003-2005', '2013-2016', '00:00-05:59', '18:00-23:59', '06:00-17:59',
                           'Polar_Rho', 'Polar_Phi', 'XY_PCA1', 'XY_PCA2',
                           'Patrol_Division', 
                           'PdDistrict_BAYVIEW', 'PdDistrict_CENTRAL', 'PdDistrict_INGLESIDE', 'PdDistrict_MISSION', 'PdDistrict_NORTHERN', 
                           'PdDistrict_PARK', 'PdDistrict_RICHMOND', 'PdDistrict_SOUTHERN', 'PdDistrict_TARAVAL', 'PdDistrict_TENDERLOIN'], axis = 1)
X_train_columns = X_train.columns
scaler = StandardScaler().fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train), columns = X_train_columns)
category = pd.factorize(train_data['Category'], sort = True)
y_train = category[0]


# Cross validation
pd.set_option('display.max_columns', 30)
skf = RepeatedStratifiedKFold(n_splits = 2, n_repeats = 3, random_state = 2019)
parameters = {
    'tol': [1e-4]
}
grid_search = GridSearchCV(estimator = QuadraticDiscriminantAnalysis(), param_grid = parameters, scoring = 'neg_log_loss',
                            cv = skf, n_jobs = -1, refit = False, verbose = 2)
grid_search.fit(X_train, y_train)
print(pd.DataFrame(grid_search.cv_results_))


# Predict using final model
X_test = pd.read_csv(ZipFile('data/processed/test.zip').open('test.csv'))
crime_id = X_test['Id']
X_test = X_test.drop(['Id', 
                      '2003-2005', '2013-2016', '00:00-05:59', '18:00-23:59', '06:00-17:59',
                      'Polar_Rho', 'Polar_Phi', 'XY_PCA1', 'XY_PCA2',
                      'Patrol_Division', 
                      'PdDistrict_BAYVIEW', 'PdDistrict_CENTRAL', 'PdDistrict_INGLESIDE', 'PdDistrict_MISSION', 'PdDistrict_NORTHERN', 
                      'PdDistrict_PARK', 'PdDistrict_RICHMOND', 'PdDistrict_SOUTHERN', 'PdDistrict_TARAVAL', 'PdDistrict_TENDERLOIN'], axis = 1)
X_test_columns = X_test.columns
X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test_columns)
clf = QuadraticDiscriminantAnalysis()
clf.fit(X_train, y_train)
pred_proba = clf.predict_proba(X_test)


# Save results
result = pd.DataFrame(pred_proba)
result.insert(0, 'Id', crime_id)
column_names = np.insert(category[1], 0, 'Id')
result.columns = column_names
result.to_csv('submission.csv', index = False)
