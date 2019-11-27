import numpy as np
import pandas as pd
import zipfile
import src.utilities.data_cleaning as dc
import src.utilities.outlier_handler as oh


# Load the raw train data obtained from Kaggle
train_zip = zipfile.ZipFile('data/raw/sf-crime.zip').open('train.csv')
train_data = pd.read_csv(train_zip)
pd.set_option('display.max_columns', 37)


# Exploratory Data Analysis
train_data.describe()
train_data = train_data.drop(columns = ['Descript', 'Resolution'])

# Handle X and Y outliers
# handle_outliers() also returns the calculated mean coordinates of the train dataset that will be used
# for handling outliers in the test dataset 
result_tuple = oh.handle_outliers(train_data)
train_data = result_tuple[0]
avg_XY = result_tuple[1]
train_data.describe()


# Data pre-processing and feature Engineering
train_data = dc.main_clean(train_data, center_scale = False)


# Model Fitting and Prediction

# Prepare data for training
categories = pd.factorize(train_data['Category'], sort = True)
y_train = pd.Series(categories[0]).astype('category')
X_train = train_data.drop(columns = 'Category', axis = 1)

# TODO: Train the best model


# Prediction

# Load and pre-process test data
test_zip = zipfile.ZipFile('data/raw/sf-crime.zip').open('test.csv')
test_data = pd.read_csv(test_zip)
test_data.describe()
test_data = oh.handle_outliers(test_data, avg_XY)[0]
test_data.describe()
test_data = dc.main_clean(test_data, center_scale = False)
