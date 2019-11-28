import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from zipfile import ZipFile


# Load Data
train_data = pd.read_csv(ZipFile('data/processed/train.zip').open('train.csv'))
X_train = train_data.drop(['Category', 
                           '2010-2012', '06:00-17:59', 
                           'PdDistrict_TARAVAL', 'Patrol_Division',
                           'Polar_Rho', 'Polar_Phi', 'X_R30', 'Y_R30', 'X_R60', 'Y_R60', 'XY_PCA1', 'XY_PCA2'], axis = 1)
category = pd.factorize(train_data['Category'], sort = True)
y_train = category[0]


# Cross validation
pd.set_option('display.max_columns', 30)
skf = RepeatedStratifiedKFold(n_splits = 2, n_repeats = 3, random_state = 2019)
parameters = {
    'solver': ['svd'],
    'shrinkage': [None], 
    'n_components': [None],
    'tol': [1e-4]
}
# Log loss measure
grid_search = GridSearchCV(estimator = LinearDiscriminantAnalysis(), param_grid = parameters, scoring = 'neg_log_loss',
                            cv = skf, n_jobs = -1, refit = False, verbose = 2)
grid_search.fit(X_train, y_train)
print(pd.DataFrame(grid_search.cv_results_))
# Accuracy measure
grid_search = GridSearchCV(estimator = LinearDiscriminantAnalysis(), param_grid = parameters, scoring = 'accuracy',
                            cv = skf, n_jobs = -1, refit = False, verbose = 2)
grid_search.fit(X_train, y_train)
print(pd.DataFrame(grid_search.cv_results_))


# Predict using final model
X_test = pd.read_csv(ZipFile('data/processed/test.zip').open('test.csv'))
crime_id = X_test['Id']
X_test = X_test.drop(['Id', 
                     '2010-2012', '06:00-17:59', 
                     'PdDistrict_TARAVAL', 'Patrol_Division',
                     'Polar_Rho', 'Polar_Phi', 'X_R30', 'Y_R30', 'X_R60', 'Y_R60', 'XY_PCA1', 'XY_PCA2'], axis = 1)
clf = LinearDiscriminantAnalysis()
clf.fit(X_train, y_train)
pred_proba = clf.predict_proba(X_test)


# Save results
result = pd.DataFrame(pred_proba)
result.insert(0, 'Id', crime_id)
column_names = np.insert(category[1], 0, 'Id')
result.columns = column_names
result.to_csv('submission.csv', index = False)
