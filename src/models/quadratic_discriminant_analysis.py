import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold

# Load Data
train_data = pd.read_csv('data/clean_centered_final.csv')
y_train = train_data['Category'].astype('category')
X_train = train_data.drop(['Category', 
                           'Polar_Rho', 'Polar_Phi',
                           '00:00-05:59', '18:00-23:59', '06:00-17:59',
                           'PCA_X', 'PCA_Y',
                           'PatrolDivision', 
                           '2003-2005', '2013-2016',
                           'PdDistrict_BAYVIEW', 'PdDistrict_CENTRAL', 'PdDistrict_INGLESIDE', 'PdDistrict_MISSION', 'PdDistrict_NORTHERN', 
                           'PdDistrict_PARK', 'PdDistrict_RICHMOND', 'PdDistrict_SOUTHERN', 'PdDistrict_TARAVAL', 'PdDistrict_TENDERLOIN'], axis = 1)

# repeatedcv for validation score
skf = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 5, random_state = 2019)

parameters = {
    'tol': [1e-4]
}
grid_search = GridSearchCV(estimator = QuadraticDiscriminantAnalysis(), param_grid = parameters, scoring = 'neg_log_loss',
                            cv = skf, n_jobs = -1, refit = False, verbose = 2)
grid_search.fit(X_train, y_train)
pd.set_option('display.max_columns', 30)
pd.DataFrame(grid_search.cv_results_)
