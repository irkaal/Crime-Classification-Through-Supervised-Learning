import zipfile
import numpy as np
import pandas as pd
import xgboost as xgb
import src.utilities.data_cleaning as dc
import src.utilities.outlier_handler as oh

# Load the train data 
train_zip = zipfile.ZipFile('data/raw/sf-crime.zip').open('train.csv')
train_data = pd.read_csv(train_zip)
pd.set_option('display.max_columns', 37)

# Exploratory Data Analysis
train_data.describe()
train_data = train_data.drop(columns = ['Descript', 'Resolution'])

# Handle X and Y outliers
# handle_outliers() returns the processed dataframe and calculated mean coord. by district
result_tuple = oh.handle_outliers(train_data)
train_data = result_tuple[0]
train_data.describe()

# Data pre-processing and feature Engineering
train_data = dc.main_clean(train_data)

# Prepare data for training
category = pd.factorize(train_data['Category'], sort = True)
y_train = pd.Series(category[0]).astype('category')
X_train = train_data.drop(columns = 'Category', axis = 1)

# Train the best model
clf = xgb.XGBClassifier(max_depth=12, learning_rate=0.05, n_estimators=1, 
                        objective='multi:softmax', gamma=0.75, verbosity=1, 
                        min_child_weight=3, subsample=0.75, colsample_bytree=0.8, 
                        reg_alpha=5, reg_lambda=1.8, seed=2019, n_jobs=-1)
clf.fit(X_train, y_train)

# Load and pre-process test data
test_zip = zipfile.ZipFile('data/raw/sf-crime.zip').open('test.csv')
test_data = pd.read_csv(test_zip)
test_data.describe()

# Handle X and Y outliers using the mean coordinates obtained from train data
avg_XY = result_tuple[1]
test_data = oh.handle_outliers(test_data, avg_XY)[0]
test_data = dc.main_clean(test_data)
test_data.describe()

# Prediction
X_test = test_data.drop('Id', axis = 1)
y_pred = clf.predict_proba(X_test)

# Save results
result = pd.DataFrame(y_pred)
result.insert(0, 'Id', test_data['Id'])
column_names = np.insert(category[1], 0, 'Id')
result.columns = column_names
# result.to_csv('submission.csv', index = False)
print(result)