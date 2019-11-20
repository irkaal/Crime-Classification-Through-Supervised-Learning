import sys
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def main_clean(dataset, center_scale = False):
    # Encode PdDistrict (One-Hot encoding)
    dataset = dataset.join(pd.get_dummies(dataset['PdDistrict'], prefix = 'PdDistrict'))
    
    # Encode Patrol Division (Label encoding - 0 for Metro and 1 for GoldenGate)        
    division = {'BAYVIEW': 0, 'MISSION': 0, 'PARK': 0, 'RICHMOND': 0, 'TARAVAL': 0, 
                'CENTRAL': 1, 'INGLESIDE': 1, 'NORTHERN': 1, 'SOUTHERN': 1, 'TENDERLOIN': 1} 
    dataset['PatrolDivision'] = dataset['PdDistrict'].map(division)
    
    # Encode Address by type (Label encoding - 0 for not an intersection and 1 for an intersection)
    dataset['Intersection'] = dataset['Address'].map(lambda a: int('/' in a))
    
    # Date and time features
    date_indices = pd.DatetimeIndex(dataset['Dates'])
    # Encode Hour + Minute / 60
    date_indices_hour = date_indices.hour
    dataset = encode_cyclic(dataset, 'Hour', date_indices_hour + date_indices.minute / 60, 24)
    # Encode time into 3 Periods 
    period = np.floor(date_indices_hour / 6) % 6
    dataset['00:00-05:59'] = (period == 0).astype(int)
    dataset['06:00-17:59'] = np.logical_or(period == 1, period == 2).astype(int)
    dataset['18:00-23:59'] = (period == 3).astype(int)
    # Encode DayOfWeek
    dataset = encode_cyclic(dataset, 'DayOfWeek', date_indices.dayofweek, 7) 
    # Encode DayofYear
    dataset = encode_cyclic(dataset, 'DayOfYear', date_indices.dayofyear, 365)
    # Encode Year
    date_indices_year = date_indices.year
    dataset['2003-2005'] = (date_indices_year <= 2005).astype(int)
    dataset['2006-2009'] = np.logical_and(date_indices_year >= 2006, date_indices_year <= 2009).astype(int)
    dataset['2010-2012'] = np.logical_and(date_indices_year >= 2010, date_indices_year <= 2012).astype(int)
    dataset['2013-2016'] = (date_indices_year >= 2013).astype(int)
    
    # Geospatial features
    x, y = dataset['X'], dataset['Y']
    xy = dataset[['X','Y']]
    # Polar Coordinates
    dataset['Polar_Rho'] = np.sqrt(x**2 + y**2)
    dataset['Polar_Phi'] = np.arctan(y / x)
    # Rotation by 30 and 60 degrees - Source: https://en.wikipedia.org/wiki/Rotation_of_axes
    dataset['R30_X'] = (x * np.cos(30)) + (y * np.sin(30))
    dataset['R30_Y'] = (y * np.cos(30)) - (x * np.sin(30))
    dataset['R60_X'] = (x * np.cos(60)) + (y * np.sin(60))
    dataset['R60_Y'] = (y * np.cos(60)) - (x * np.sin(60))
    # Rotation by PCA
    coordinates = xy.values
    pca_trans = PCA().fit(coordinates).transform(xy)
    dataset['PCA_X'] = pca_trans[:,0]
    dataset['PCA_Y'] = pca_trans[:,1]

    # Distance to 10 police stations
    x_station = np.array([-122.409960, -122.446261, -122.432516, -122.389411, -122.412924, -122.397771, -122.421951, -122.455391, -122.464462, -122.481516])
    y_station = np.array([37.798736, 37.724694, 37.780226, 37.772382, 37.783783, 37.729825, 37.763013, 37.767835, 37.780016, 37.743755])
    station_distances = np.array([
        haversine(x, y, x_station[0], y_station[0]),  # Central Station 
        haversine(x, y, x_station[1], y_station[1]),  # Ingleside Station
        haversine(x, y, x_station[2], y_station[2]),  # Northern Station
        haversine(x, y, x_station[3], y_station[3]),  # Southern Station
        haversine(x, y, x_station[4], y_station[4]),  # Tenderloin Station
        haversine(x, y, x_station[5], y_station[5]),  # Bayview Station
        haversine(x, y, x_station[6], y_station[6]),  # Mission Station
        haversine(x, y, x_station[7], y_station[7]),  # Park Station
        haversine(x, y, x_station[8], y_station[8]),  # Richmond Station 
        haversine(x, y, x_station[9], y_station[9])   # Taraval Station
    ])
    # Haversine distance to nearest police station
    dataset['NearestStation'] = station_distances.min(axis = 0) # Take min index wise
    # Bearing degrees to nearest police station
    min_index = station_distances.argmin(axis = 0) # Take argmin index wise
    dataset['NearestStationBearing'] = bearing(x, y, x_station[min_index], y_station[min_index])

    # Drop unused columns
    dataset = dataset.drop(columns = ['PdDistrict', 'Address', 'Dates', 'DayOfWeek', 'Descript', 'Resolution'])

    # Center and scale
    if (center_scale):
        label = dataset['Category']
        data = dataset.drop(columns = ['Category'])
        data_scaled = StandardScaler().fit(data).transform(data)
        dataset = pd.DataFrame(data_scaled, columns = data.columns)
        dataset['Category'] = label
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

# Bearing degree
def bearing(x1, y1, x2, y2):
  x1, y1, x2, y2 = map(np.radians, [x1, y1, x2, y2]) 
  y = np.sin(x2 - x1) * np.cos(y2)
  x = np.cos(y1) * np.sin(y2) - np.sin(y1) * np.cos(y2) * np.cos(x2 - x1)
  return np.degrees(np.arctan2(y, x))

# FOR DEBUGGING:
# import os
# from zipfile import ZipFile
# os.chdir(os.path.dirname(os.path.abspath(__file__)))
# zipFile = ZipFile('data/sf-crime-raw.zip')
# dataset = pd.read_csv(zipFile.open('train.csv'))
# main_clean(dataset, center_scale = True)
