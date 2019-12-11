import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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

# Cross validation
rskf = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 5, random_state = 2019)
lda_clf = make_pipeline(StandardScaler(), 
                        GridSearchCV(estimator = LinearDiscriminantAnalysis(), 
                                     param_grid = {'solver': ['svd']}, 
                                     scoring = 'neg_log_loss',
                                     cv = rskf, 
                                     n_jobs = -1, 
                                     verbose = 2))
lda_clf.fit(X_train, y_train)
result = pd.DataFrame(lda_clf._final_estimator.cv_results_)
print(result[['mean_test_score', 'rank_test_score']])

# Prediction
X_train_columns = X_train.columns
scaler = StandardScaler().fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train), columns = X_train_columns)

X_test = pd.read_csv(ZipFile('data/processed/test.zip').open('test.csv'))
crime_id = X_test['Id']
X_test = X_test.drop(['Id', '2010-2012', '0600-1759', 'PdDistrict_TARAVAL', 'Patrol_Division',
                      'Polar_Rho', 'Polar_Phi', 'X_R30', 'Y_R30', 'X_R60', 'Y_R60', 'XY_PCA1', 'XY_PCA2'], axis = 1)
X_test_columns = X_test.columns
X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test_columns)

clf = LinearDiscriminantAnalysis().fit(X_train, y_train)
pred_proba = clf.predict_proba(X_test)
result = pd.DataFrame(pred_proba)
result.insert(0, 'Id', crime_id)
column_names = np.insert(category[1], 0, 'Id')
result.columns = column_names
result.to_csv('lda_submission.csv', index = False)
