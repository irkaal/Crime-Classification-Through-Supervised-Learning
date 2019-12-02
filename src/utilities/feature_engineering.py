import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def encode_district(dataset):
    district_df = pd.get_dummies(dataset['PdDistrict'], prefix = 'PdDistrict')
    return dataset.join(district_df)

def insert_patrol_division(dataset):
    # Source: https://en.wikipedia.org/wiki/San_Francisco_Police_Department#Stations
    district_to_division = {
        'BAYVIEW': 0, 'MISSION': 0, 'PARK': 0, 'RICHMOND': 0, 'TARAVAL': 0, 
        'CENTRAL': 1, 'INGLESIDE': 1, 'NORTHERN': 1, 'SOUTHERN': 1, 'TENDERLOIN': 1
    } 
    dataset['Patrol_Division'] = dataset['PdDistrict'].map(district_to_division)
    return dataset

def insert_intersection(dataset):
    dataset['Intersection'] = dataset['Address'].map(lambda a: int('/' in a))
    return dataset

def encode_day_of_week(dataset):
    dayofweek = pd.DatetimeIndex(dataset['Dates']).dayofweek
    day_of_week = encode_cyclic(dataset, 'DayOfWeek', dayofweek, 7)
    dataset['DayOfWeek_X'] = day_of_week[0]
    dataset['DayOfWeek_Y'] = day_of_week[1]
    return dataset

def encode_day_of_year(dataset):
    dayofyear = pd.DatetimeIndex(dataset['Dates']).dayofyear
    day_of_year = encode_cyclic(dataset, 'DayOfYear', dayofyear, 365)
    dataset['DayOfYear_X'] = day_of_year[0]
    dataset['DayOfYear_Y'] = day_of_year[1]
    return dataset

def encode_hour(dataset):
    datetimeIndex = pd.DatetimeIndex(dataset['Dates'])
    hour = encode_cyclic(dataset, 'Hour', datetimeIndex.hour + datetimeIndex.minute / 60, 24)
    dataset['Hour_X'] = hour[0]
    dataset['Hour_Y'] = hour[1]
    return dataset

def insert_periods(dataset):
    hour = pd.DatetimeIndex(dataset['Dates']).hour
    period = np.floor(hour / 6) % 6
    dataset['00:00-05:59'] = (period == 0).astype(int)
    dataset['06:00-17:59'] = np.logical_or(period == 1, period == 2).astype(int)
    dataset['18:00-23:59'] = (period == 3).astype(int)
    return dataset

def insert_year_groups(dataset):
    date_indices_year = pd.DatetimeIndex(dataset['Dates']).year
    dataset['2003-2005'] = (date_indices_year <= 2005).astype(int)
    dataset['2006-2009'] = np.logical_and(date_indices_year >= 2006, date_indices_year <= 2009).astype(int)
    dataset['2010-2012'] = np.logical_and(date_indices_year >= 2010, date_indices_year <= 2012).astype(int)
    dataset['2013-2016'] = (date_indices_year >= 2013).astype(int)
    return dataset

def insert_polar(dataset):
    x, y = dataset['X'], dataset['Y']
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan(y / x)
    dataset['Polar_Rho'] = rho
    dataset['Polar_Phi'] = phi
    return dataset

def insert_rotation(dataset, degrees):
    x, y = dataset['X'], dataset['Y']
    # Source: https://en.wikipedia.org/wiki/Rotation_of_axes
    rotated_x = (x * np.cos(degrees)) + (y * np.sin(degrees))
    rotated_y = (y * np.cos(degrees)) - (x * np.sin(degrees))
    dataset['X_R' + str(degrees)] = rotated_x
    dataset['Y_R' + str(degrees)] = rotated_y
    return dataset

def insert_pca_rotation(dataset):
    pca_trans = PCA().fit_transform(dataset[['X','Y']])
    pca_df = pd.DataFrame(data = pca_trans, columns = ['XY_PCA1', 'XY_PCA2'])
    return dataset.join(pca_df)

def insert_nearest_station(dataset):
    x, y = dataset['X'], dataset['Y']
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
    dataset['Nearest_Station'] = station_distances.min(axis = 0) # Take min index wise
    return dataset 

def insert_nearest_station_bearing(dataset):
    x, y = dataset['X'], dataset['Y']
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
    min_index = station_distances.argmin(axis = 0) # Take argmin index wise
    dataset['Nearest_Station_Bearing'] = bearing(x, y, x_station[min_index], y_station[min_index])
    return dataset

# Helper functions

# Encoding to preserve cyclical properties
def encode_cyclic(dataset, colName, val, maxVal):
    x_com = np.cos(2 * np.pi * val / maxVal)
    y_com = np.sin(2 * np.pi * val / maxVal)
    return x_com, y_com

# Calculate haversine distance between two points in kilometres
def haversine(x1, y1, x2, y2):
    # Source: https://en.wikipedia.org/wiki/Haversine_formula
    lon1, lat1, lon2, lat2 = map(np.radians, [x1, y1, x2, y2])
    h = np.sin((lat2 - lat1) / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2)**2
    distance = 2 * 6378.137 * np.arcsin(np.sqrt(h))
    return distance

# Calculate bearing degree between two points
def bearing(x1, y1, x2, y2):
    # Source: https://www.movable-type.co.uk/scripts/latlong.html
    x1, y1, x2, y2 = map(np.radians, [x1, y1, x2, y2]) 
    y = np.sin(x2 - x1) * np.cos(y2)
    x = np.cos(y1) * np.sin(y2) - np.sin(y1) * np.cos(y2) * np.cos(x2 - x1)
    degrees = np.degrees(np.arctan2(y, x))
    return degrees 
