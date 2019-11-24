import numpy as np
import pandas as pd
import zipfile
import src.features.feature_engineering as fe
import src.features.outlier_handler as oh

# Load the raw train data obtained from Kaggle
train_zip = zipfile.ZipFile('data/raw/sf-crime.zip').open('train.csv')
train_data = pd.read_csv(train_zip)


# Exploratory Data Analysis

# Handle X and Y outliers
# TODO: Move R Code here


# Feature Engineering

# PdDistrict
# One-hot encoding
train_data = fe.get_districts(train_data)
train_data = train_data.drop('PdDistrict')

# Patrol Division 
# 0 for Metro and 1 for GoldenGate
train_data['Patrol_Division'] = fe.get_patrol_div(train_data)

# Intersection
# Group addresses by type 
# 0 for not an intersection and 1 for an intersection
train_data['Intersection'] = fe.get_intersection(train_data)
train_data = train_data.drop('Address')

# Day of Week 
# Cyclic representation
day_of_week_tuple = fe.get_day_of_week(train_data)
train_data['DayOfWeek_X'] = day_of_week_tuple[0]
train_data['DayOfWeek_Y'] = day_of_week_tuple[1]
train_data = train_data.drop('DayOfWeek')

# Day of Year
# Cyclic representation
day_of_year_tuple = fe.get_day_of_year(train_data)
train_data['DayOfYear_X'] = day_of_year_tuple[0]
train_data['DayOfYear_Y'] = day_of_year_tuple[1]

# Hour 
# Cylic representation
hour_tuple = fe.get_hour(train_data)
train_data['Hour_X'] = hour_tuple[0]
train_data['Hour_Y'] = hour_tuple[1]

# Periods
# 00:00-05:59, 06:00-17:59, 18:00-23:59
period_tuple = fe.get_periods(train_data) 
train_data['00:00-05:59'] = period_tuple[0]
train_data['06:00-17:59'] = period_tuple[1]
train_data['18:00-23:59'] = period_tuple[2]

# Years
# 2003-2005, 2006-2009, 2010-2012, 2013-2016
years_tuple = fe.get_year_groups(train_data)
train_data['2003-2005'] = years_tuple[0]
train_data['2006-2009'] = years_tuple[1]
train_data['2010-2012'] = years_tuple[2]
train_data['2013-2016'] = years_tuple[3]
train_data = train_data.drop('Dates')

# Polar Coordinates
polar_tuple = fe.get_polar_coord(train_data)
train_data['Polar_Rho'] = polar_tuple[0]
train_data['Polar_Phi'] = polar_tuple[1]

# Rotated Coordinates
# Rotate coordinates by 30 degrees
r30_tuple = fe.rotate_coord(train_data, degrees = 30)
train_data['X_R30'] = r30_tuple[0]
train_data['Y_R30'] = r30_tuple[1]
# Rotate coordinates by 60 degrees
r60_tuple = fe.rotate_coord(train_data, degrees = 60)
train_data['X_R60'] = r60_tuple[0]
train_data['Y_R60'] = r60_tuple[1]

# PCA Decomposition of Coordinates
pc_tuple = fe.pca_decomp_coord(train_data, n_components = 2)
train_data['XY_PCA1'] = pc_tuple[0]
train_data['XY_PCA2'] = pc_tuple[1]

# Nearest Police Station Distance
train_data['Nearest_Station'] = fe.get_nearest_station_distance(train_data)

# Nearest Police Station Bearing
train_data['Nearest_Station_Bearing'] = fe.get_nearest_station_bearing(train_data)

# Remove unused columns from dataframe
train_data = train_data.drop(columns = ['Descript', 'Resolution'])

# Center and scale data for linear models
train_data = fe.center_and_scale(train_data)


# Model Fitting and Prediction

# Prepare data for training
X_train = train_data.drop('Category', axis = 1)
y_train = train_data['Category'].astype('category')

# Train the best model
# TODO:
