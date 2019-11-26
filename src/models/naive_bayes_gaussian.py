import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from zipfile import ZipFile

# Load Data
train_data = pd.read_csv(ZipFile('data/processed/train.zip').open('train.csv'))
y_train = train_data['Category'].astype('category')
X_train = train_data.drop(['Category', 
                           '2010-2012', '06:00-17:59', 
                           'PdDistrict_TARAVAL', 'PatrolDivision',
                           'Polar_Rho', 'Polar_Phi', 'R30_X', 'R30_Y', 'R60_X', 'R60_Y', 'PCA_X', 'PCA_Y'], axis = 1)

pd.set_option('display.max_columns', 30)
skf = RepeatedStratifiedKFold(n_splits = 2, n_repeats = 3, random_state = 2019)
parameters = {'priors' : [None]}

# Log loss measure 
grid_search = GridSearchCV(estimator = GaussianNB(), param_grid = parameters, scoring = 'neg_log_loss',
                            cv = skf, n_jobs = -1, refit = False, verbose = 2)
grid_search.fit(X_train, y_train)
print(pd.DataFrame(grid_search.cv_results_))

# Accuracy measure 
grid_search = GridSearchCV(estimator = GaussianNB(), param_grid = parameters, scoring = 'accuracy',
                            cv = skf, n_jobs = -1, refit = False, verbose = 2)
grid_search.fit(X_train, y_train)
print(pd.DataFrame(grid_search.cv_results_))
