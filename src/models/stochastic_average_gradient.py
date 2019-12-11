import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from zipfile import ZipFile

# Load data
train_data = pd.read_csv(ZipFile('data/processed/train.zip').open('train.csv'))
X_train = train_data.drop(['Category'], axis = 1)
category = pd.factorize(train_data['Category'], sort = True)
y_train = category[0]

# Cross Validation
rskf = RepeatedStratifiedKFold(n_splits = 2, n_repeats = 1, random_state = 2019)
parameters = {
    'logisticregression__penalty': ['l2'],
    'logisticregression__C': [1e-2, 1e-1, 1e0],
    'logisticregression__solver': ['sag'], 
    'logisticregression__multi_class': ['multinomial'],
    'logisticregression__random_state': [2019],
    'logisticregression__max_iter': [5000]
}
pipe = make_pipeline(StandardScaler(), 
                     LogisticRegression())
grid_search = GridSearchCV(estimator = pipe,
                           param_grid = parameters, 
                           scoring = 'neg_log_loss',
                           cv = rskf, 
                           refit = False,
                           n_jobs = -1, 
                           verbose = 2)
grid_search.fit(X_train, y_train)
result = pd.DataFrame(grid_search.cv_results_)
print(result[['params', 'mean_test_score', 'rank_test_score']])

# Prediction
scaler = StandardScaler().fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train), columns = X_train.columns)

X_test = pd.read_csv(ZipFile('data/processed/test.zip').open('test.csv'))
crime_id = X_test['Id']
X_test = X_test.drop(['Id'], axis = 1)
X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)

clf = LogisticRegression(solver = 'sag', 
                         tol = 1e-4, 
                         C = 0.1, 
                         max_iter = 5000, 
                         multi_class = 'multinomial', 
                         verbose = 2, 
                         random_state = 2019, 
                         n_jobs=-1)
clf.fit(X_train, y_train)
pred_proba = clf.predict_proba(X_test)
result = pd.DataFrame(pred_proba)
result.insert(0, 'Id', crime_id)
column_names = np.insert(category[1], 0, 'Id')
result.columns = column_names
result.to_csv('logistic_sag_submission.csv', index = False)
