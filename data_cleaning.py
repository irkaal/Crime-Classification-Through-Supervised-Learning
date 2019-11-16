import os
import sys
import time
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.preprocessing import StandardScaler
from zipfile import ZipFile
from sklearn.decomposition import PCA
# Global variables
i = 0
max_i = 0
tic = time.time()


def main_clean(dataset, center_scale = False):
    start(total = int(center_scale) + 13) # Start progress tracker
    # Check if dataset has all the necessary columns
    colList = ['PdDistrict', 'Address', 'Dates', 'DayOfWeek', 'Descript', 'Resolution']
    if set(colList).issubset(dataset.columns):
        # Encode PdDistrict (One-Hot encoding)
        update_progress('Encoding PdDistrict')
        dataset = dataset.join(pd.get_dummies(dataset['PdDistrict'], prefix = 'PdDistrict'))
        
        # Encode Patrol Division (Label encoding - 0 for Metro and 1 for GoldenGate)        
        update_progress('Encoding PatrolDivision')
        dataset = encode_patrol_div(dataset)
        
        # Encode Address by type (Label encoding - 0 for not an intersection and 1 for an intersection)
        update_progress('Encoding Address')
        dataset['Intersection'] = dataset['Address'].map(lambda a: int('/' in a))
        
        # Retrieve DatetimeIndex
        update_progress('Retrieving DatetimeIndex')
        date_indices = pd.DatetimeIndex(dataset['Dates'])
        
        # Encode DayOfWeek
        update_progress('Encoding DayOfWeek')
        dataset = encode_cyclic(dataset, 'DayOfWeek', date_indices.dayofweek, 7) 
        
        # Encode DayofYear
        update_progress('Encoding DayOfYear')
        dataset = encode_cyclic(dataset, 'DayOfYear', date_indices.dayofyear, 365)
        
        # Encode DayofMonth
        update_progress('Encoding DayOfMonth')
        dataset = encode_cyclic(dataset, 'DayOfMonth', date_indices.day, 31)
        
        # Encode Year
        update_progress('Encoding Year')
        dataset = dataset.join(pd.get_dummies(date_indices.year, prefix = 'Year'))
       
        # Encode Hour + Minute / 60
        update_progress('Encoding Hour')
        dataset = encode_cyclic(dataset, 'Hour', date_indices.hour + date_indices.minute / 60, 24)
        
        # Encode time into 6-Hour Period
        dataset = encode_period(dataset, date_indices.hour)
        
        # Encode haversine distance to nearest police station 
        update_progress('Encoding NearestStation_Hav') 
        dataset = encode_nearest_hav(dataset)
        
        # TODO: Add comments for these additional features
        dataset = encode_nearest_man(dataset)
        dataset = polar_coord(dataset)
        dataset = rotate_coord(dataset)
        dataset = rotate_pca(dataset)
        dataset = encode_nearest_bearing_hav(dataset)
        dataset = encode_nearest_bearing_man(dataset)

        # Drop unused columns
        update_progress('Dropping unused columns') 
        dataset = dataset.drop(columns = colList)

        # Center and scale
        if (center_scale):
            update_progress('Centering and scaling features')
            dataset = center_scale_features(dataset)
    end() # End progress tracker
    return dataset


def encode_patrol_div(dataset):
    division = {'BAYVIEW': 0, 'MISSION': 0, 'PARK': 0, 'RICHMOND': 0, 'TARAVAL': 0, 
                'CENTRAL': 1, 'INGLESIDE': 1, 'NORTHERN': 1, 'SOUTHERN': 1, 'TENDERLOIN': 1} 
    dataset['PatrolDivision'] = dataset['PdDistrict'].map(division) 
    return dataset

def encode_period(dataset, hour):
    period = np.floor(hour / 6) % 6
    dataset['00:00-05:59'] = (period == 0).astype(int)
    dataset['06:00-11:59'] = (period == 1).astype(int)
    dataset['12:00-17:59'] = (period == 2).astype(int)
    dataset['18:00-23:59'] = (period == 3).astype(int)
    return dataset

def rotate_coord(dataset):
    # Source - https://en.wikipedia.org/wiki/Rotation_of_axes
    # x' = x * cosθ + y * sinθ
    # y' = y * cosθ - x * sinθ
    x, y = dataset['X'], dataset['Y']
    dataset['R30_X'] = (x * np.cos(30)) + (y * np.sin(30))
    dataset['R30_Y'] = (y * np.cos(30)) - (x * np.sin(30))
    dataset['R45_X'] = (x * np.cos(45)) + (y * np.sin(45))
    dataset['R45_Y'] = (y * np.cos(45)) - (x * np.sin(45))
    dataset['R60_X'] = (x * np.cos(60)) + (y * np.sin(60))
    dataset['R60_Y'] = (y * np.cos(60)) - (x * np.sin(60))
    return dataset

def rotate_pca(dataset):
    xy = dataset[['X','Y']]
    coordinates = xy.values
    pca_trans = PCA().fit(coordinates).transform(xy)
    dataset['PCA_X'] = pca_trans[:,0]
    dataset['PCA_Y'] = pca_trans[:,1]
    return dataset

def polar_coord(dataset):
    x, y = dataset['X'], dataset['Y']
    dataset['Polar_Rho'] = np.sqrt(x**2 + y**2)
    dataset['Polar_Phi'] = np.arctan(y / x)
    return dataset

def encode_nearest_hav(dataset):
    x, y = dataset['X'], dataset['Y']
    # Add distance to the closest police station
    dataset['NearestStation_Hav'] = np.array([
        haversine(x, y, -122.409960, 37.798736),  # Central Station 
        haversine(x, y, -122.446261, 37.724694),  # Ingleside Station
        haversine(x, y, -122.432516, 37.780226),  # Northern Station
        haversine(x, y, -122.389411, 37.772382),  # Southern Station
        haversine(x, y, -122.412924, 37.783783),  # Tenderloin Station
        haversine(x, y, -122.397771, 37.729825),  # Bayview Station
        haversine(x, y, -122.421951, 37.763013),  # Mission Station
        haversine(x, y, -122.455391, 37.767835),  # Park Station
        haversine(x, y, -122.464462, 37.780016),  # Richmond Station 
        haversine(x, y, -122.481516, 37.743755)   # Taraval Station
        ]).min(axis = 0) # Take min index wise
    return dataset

def encode_nearest_man(dataset):
    x, y = dataset['X'], dataset['Y']
    # Add distance to the closest police station
    dataset['NearestStation_Man'] = np.array([
        manhattan(x, y, -122.409960, 37.798736),  # Central Station 
        manhattan(x, y, -122.446261, 37.724694),  # Ingleside Station
        manhattan(x, y, -122.432516, 37.780226),  # Northern Station
        manhattan(x, y, -122.389411, 37.772382),  # Southern Station
        manhattan(x, y, -122.412924, 37.783783),  # Tenderloin Station
        manhattan(x, y, -122.397771, 37.729825),  # Bayview Station
        manhattan(x, y, -122.421951, 37.763013),  # Mission Station
        manhattan(x, y, -122.455391, 37.767835),  # Park Station
        manhattan(x, y, -122.464462, 37.780016),  # Richmond Station 
        manhattan(x, y, -122.481516, 37.743755)   # Taraval Station
        ]).min(axis = 0) # Take min index wise
    return dataset

def encode_nearest_bearing_hav(dataset):
    # TODO: Put this together with distance function to avoid repetition
    x, y = dataset['X'], dataset['Y']
    x_station = np.array([-122.409960, -122.446261, -122.432516, -122.389411, -122.412924, -122.397771, -122.421951, -122.455391, -122.464462, -122.481516])
    y_station = np.array([37.798736, 37.724694, 37.780226, 37.772382, 37.783783, 37.729825, 37.763013, 37.767835, 37.780016, 37.743755])
    min_index = np.array([
        haversine(x, y, -122.409960, 37.798736),  # Central Station 
        haversine(x, y, -122.446261, 37.724694),  # Ingleside Station
        haversine(x, y, -122.432516, 37.780226),  # Northern Station
        haversine(x, y, -122.389411, 37.772382),  # Southern Station
        haversine(x, y, -122.412924, 37.783783),  # Tenderloin Station
        haversine(x, y, -122.397771, 37.729825),  # Bayview Station
        haversine(x, y, -122.421951, 37.763013),  # Mission Station
        haversine(x, y, -122.455391, 37.767835),  # Park Station
        haversine(x, y, -122.464462, 37.780016),  # Richmond Station 
        haversine(x, y, -122.481516, 37.743755)   # Taraval Station
        ]).argmin(axis = 0) # Take argmin index wise
    dataset['NearestStationBearing_Hav'] = bearing(x, y, x_station[min_index], y_station[min_index])
    return dataset

def encode_nearest_bearing_man(dataset):
    # TODO: Put this together with distance function to avoid repetition
    x, y = dataset['X'], dataset['Y']
    x_station = np.array([-122.409960, -122.446261, -122.432516, -122.389411, -122.412924, -122.397771, -122.421951, -122.455391, -122.464462, -122.481516])
    y_station = np.array([37.798736, 37.724694, 37.780226, 37.772382, 37.783783, 37.729825, 37.763013, 37.767835, 37.780016, 37.743755])
    min_index = np.array([
        manhattan(x, y, -122.409960, 37.798736),  # Central Station 
        manhattan(x, y, -122.446261, 37.724694),  # Ingleside Station
        manhattan(x, y, -122.432516, 37.780226),  # Northern Station
        manhattan(x, y, -122.389411, 37.772382),  # Southern Station
        manhattan(x, y, -122.412924, 37.783783),  # Tenderloin Station
        manhattan(x, y, -122.397771, 37.729825),  # Bayview Station
        manhattan(x, y, -122.421951, 37.763013),  # Mission Station
        manhattan(x, y, -122.455391, 37.767835),  # Park Station
        manhattan(x, y, -122.464462, 37.780016),  # Richmond Station 
        manhattan(x, y, -122.481516, 37.743755)   # Taraval Station
        ]).argmin(axis = 0) # Take argmin index wise
    dataset['NearestStationBearing_Man'] = bearing(x, y, x_station[min_index], y_station[min_index])
    return dataset

def center_scale_features(dataset):
    y = dataset['Category']
    x = dataset.drop(columns = ['Category'])
    std_scale = StandardScaler().fit(x)
    x_scaled = std_scale.transform(x)
    dataset = pd.DataFrame(x_scaled, columns = x.columns)
    dataset['Category'] = y
    return dataset

# Helper functions

# Cyclic encoding helper
def encode_cyclic(dataset, colName, val, maxVal):
    dataset[colName + '_X'] = np.cos(2 * np.pi * val / maxVal)
    dataset[colName + '_Y'] = np.sin(2 * np.pi * val / maxVal)
    return dataset

# Haversine distance - Great-circle distance in kilometres
def haversine(x1, y1, x2, y2):
    lon1, lat1, lon2, lat2 = map(np.radians, [x1, y1, x2, y2])
    h = np.sin((lat2 - lat1) / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2)**2
    return 2 * 6378.137 * np.arcsin(np.sqrt(h))

# Manhattan distance L1-norm
def manhattan(x1, y1, x2, y2):
  lat_dist = haversine(x1, y1, x1, y2)
  lon_dist = haversine(x1, y1, x2, y1)
  return lat_dist + lon_dist

# Bearing degree
def bearing(x1, y1, x2, y2):
  x1, y1, x2, y2 = map(np.radians, [x1, y1, x2, y2]) 
  y = np.sin(x2 - x1) * np.cos(y2)
  x = np.cos(y1) * np.sin(y2) - np.sin(y1) * np.cos(y2) * np.cos(x2 - x1)
  return np.degrees(np.arctan2(y, x))

# Progress tracker helpers

def update_progress(task = ''):
    global i, max_i
    progress = '=' * round(48 / max_i * i)
    percent = round(100 / max_i * i, 3)
    space = ' ' * (30 - len(task))
    i += 1 # increment task count
    sys.stdout.write('\r[%-48s] %d%% (%s)%s' % (progress, percent, task, space))
    sys.stdout.flush()

def start(total):
    global max_i
    max_i = total
    print('Python> Cleaning...', flush = True)
    update_progress()
    # return time.time()

def end():
    global i, tic
    i = max_i
    update_progress('Done') 
    print(f'\nElapsed time: {round(time.time() - tic, 3)} second(s)', flush = True)


if __name__== "__main__":
    try:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        zipFile = ZipFile('data/sf-crime.zip')
        dataset = pd.read_csv(zipFile.open('train.csv'))
        main_clean(dataset, center_scale = True)
    except Exception as e:
        print(e)
        print("Can't find csv!")
