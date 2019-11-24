import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold

# Load Data
train_data = pd.read_csv('data/clean_final.csv')
y_train = train_data['Category'].astype('category')
X_train = train_data.drop('Category', axis = 1)

# repeatedcv for validation score
skf = RepeatedStratifiedKFold(n_splits = 3, n_repeats = 5, random_state = 2019)

parameters = {
    'solver': ['svd'],
    'shrinkage': [None], 
    'n_components': [None],
    'tol': [1e-4]
}
grid_search = GridSearchCV(estimator = LinearDiscriminantAnalysis(), param_grid = parameters, scoring = 'neg_log_loss',
                            cv = skf, n_jobs = -1, refit = False, verbose = 2)
grid_search.fit(X_train, y_train)
pd.set_option('display.max_columns', 30)
pd.DataFrame(grid_search.cv_results_)
