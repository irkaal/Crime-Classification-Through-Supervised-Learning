import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, RepeatedStratifiedKFold

# Load Data
train_data = pd.read_csv('data/clean_final.csv')
y_train = train_data['Category'].astype('category')
X_train = train_data.drop('Category', axis = 1)
skf = RepeatedStratifiedKFold(n_splits = 3, n_repeats = 5, random_state = 2019)

# Hyperparameter random search
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 5)]
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
parameters = {'n_estimators': n_estimators,
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf}
random_search = RandomizedSearchCV(estimator = RandomForestClassifier(n_jobs = 2), param_distributions = parameters, n_iter = 2, 
                                   cv = skf, verbose = 2, random_state = 2019, n_jobs = 1)
random_search.fit(X_train, y_train)

# Hyperparameter grid search
parameters = {
    'n_estimators': [100],
    # 'max_depth': [None],
    # 'min_samples_split': [2],
    # 'min_samples_leaf': [1],
    # 'min_weight_fraction_leaf': [0.0], 
    # 'max_leaf_nodes': [None], 
    # 'min_impurity_decrease': [0.0], 
    # 'random_state': [2019], 
    # 'class_weight': [None],
    'verbose': [1],
    'n_jobs': [2]
}
grid_search2 = GridSearchCV(estimator = RandomForestClassifier(), param_grid = parameters, scoring = 'neg_log_loss',
                            cv = skf, n_jobs = 1, refit = False, verbose = 2)                            
grid_search2.fit(X_train, y_train)
pd.set_option('display.max_columns', 30)
pd.DataFrame(grid_search2.cv_results_)
