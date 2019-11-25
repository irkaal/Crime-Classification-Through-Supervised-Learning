import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from zipfile import ZipFile

# Load Data
train_data = pd.read_csv(ZipFile('data/processed/train_centered.zip').open('train.csv'))
y_train = train_data['Category'].astype('category')
X_train = train_data.drop(['Category', 
                           '2003-2005', '2013-2016', '00:00-05:59', '18:00-23:59', '06:00-17:59',
                           'Polar_Rho', 'Polar_Phi', 'PCA_X', 'PCA_Y',
                           'PatrolDivision', 
                           'PdDistrict_BAYVIEW', 'PdDistrict_CENTRAL', 'PdDistrict_INGLESIDE', 'PdDistrict_MISSION', 'PdDistrict_NORTHERN', 
                           'PdDistrict_PARK', 'PdDistrict_RICHMOND', 'PdDistrict_SOUTHERN', 'PdDistrict_TARAVAL', 'PdDistrict_TENDERLOIN'], axis = 1)

pd.set_option('display.max_columns', 30)
skf = RepeatedStratifiedKFold(n_splits = 2, n_repeats = 3, random_state = 2019)
parameters = {'tol': [1e-4]}

# Log loss measure
grid_search = GridSearchCV(estimator = QuadraticDiscriminantAnalysis(), param_grid = parameters, scoring = 'neg_log_loss',
                            cv = skf, n_jobs = -1, refit = False, verbose = 2)
grid_search.fit(X_train, y_train)
print(pd.DataFrame(grid_search.cv_results_))

# Accuracy measure
grid_search = GridSearchCV(estimator = QuadraticDiscriminantAnalysis(), param_grid = parameters, scoring = 'accuracy',
                            cv = skf, n_jobs = -1, refit = False, verbose = 2)
grid_search.fit(X_train, y_train)
print(pd.DataFrame(grid_search.cv_results_))
