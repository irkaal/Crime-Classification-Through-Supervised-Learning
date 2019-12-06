import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from zipfile import ZipFile


# Load data
train_data = pd.read_csv(ZipFile('data/processed/train.zip').open('train.csv'))
X_train = train_data.drop(['Category'], axis = 1)
X_train_columns = X_train.columns
scaler = StandardScaler().fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train), columns = X_train_columns)
category = pd.factorize(train_data['Category'], sort = True)
y_train = category[0]


# Grid search
pd.set_option('display.max_columns', 30)
skf = RepeatedStratifiedKFold(n_splits = 2, n_repeats = 1, random_state = 2019)
parameters = {
    'penalty': ['l2'],
    'C': [0.1],
    'solver': ['sag'], 
    'multi_class': ['multinomial'],
    'random_state': [2019],
    'max_iter': [5000]
}
grid_search2 = GridSearchCV(estimator = LogisticRegression(), param_grid = parameters, scoring = 'neg_log_loss',
                           cv = skf, n_jobs = -1, refit = False, verbose = 2)
grid_search2.fit(X_train, y_train)
print(pd.DataFrame(grid_search2.cv_results_))


# Prediction
X_test = pd.read_csv(ZipFile('data/processed/test.zip').open('test.csv'))
crime_id = X_test['Id']
X_test = X_test.drop(['Id'], axis = 1)
X_test_columns = X_test.columns
X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test_columns)
clf = LogisticRegression(solver = 'sag', tol = 1e-4, C = 0.1, max_iter = 5000, multi_class = 'multinomial', verbose = 2, random_state = 2019, n_jobs=-1)
clf.fit(X_train, y_train)
pred_proba = clf.predict_proba(X_test)
result = pd.DataFrame(pred_proba)
result.insert(0, 'Id', crime_id)
column_names = np.insert(category[1], 0, 'Id')
result.columns = column_names
result.to_csv('submission.csv', index = False)
