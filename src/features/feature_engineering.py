import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def get_districts(dataset):
    cols = dataset.join(pd.get_dummies(dataset['PdDistrict'], prefix = 'PdDistrict'))

    return cols

def get_patrol_div(dataset):
    # Source: https://en.wikipedia.org/wiki/San_Francisco_Police_Department#Stations
    district_to_division = {
        'BAYVIEW': 0, 'MISSION': 0, 'PARK': 0, 'RICHMOND': 0, 'TARAVAL': 0, 
        'CENTRAL': 1, 'INGLESIDE': 1, 'NORTHERN': 1, 'SOUTHERN': 1, 'TENDERLOIN': 1
    } 
    division = dataset['PdDistrict'].map(district_to_division)

    return division

def get_intersection(dataset):
    intersection = dataset['Address'].map(lambda a: int('/' in a))

    return intersection

def get_day_of_week(dataset):
    dayofweek = pd.DatetimeIndex(dataset['Dates']).dayofweek
    day_of_week = encode_cyclic(dataset, 'DayOfWeek', dayofweek, 7)
    
    return day_of_week

def get_day_of_year(dataset):
    dayofyear = pd.DatetimeIndex(dataset['Dates']).dayofyear
    day_of_year = encode_cyclic(dataset, 'DayOfYear', dayofyear, 365)

    return day_of_year

def get_hour(dataset):
    datetimeIndex = pd.DatetimeIndex(dataset['Dates'])
    hour = encode_cyclic(dataset, 'Hour', datetimeIndex.hour + datetimeIndex.minute / 60, 24)

    return hour

def get_periods(dataset):
    hour = pd.DatetimeIndex(dataset['Dates']).hour
    period = np.floor(hour / 6) % 6
    hour_0000_0559 = (period == 0).astype(int)
    hour_0600_1759 = np.logical_or(period == 1, period == 2).astype(int)
    hour_1800_2359 = (period == 3).astype(int)

    return hour_0000_0559, hour_0600_1759, hour_1800_2359

def get_year_groups(dataset):
    date_indices_year = pd.DatetimeIndex(dataset['Dates']).year
    year_2003_2005 = (date_indices_year <= 2005).astype(int)
    year_2006_2009 = np.logical_and(date_indices_year >= 2006, date_indices_year <= 2009).astype(int)
    year_2010_2012 = np.logical_and(date_indices_year >= 2010, date_indices_year <= 2012).astype(int)
    year_2013_2016 = (date_indices_year >= 2013).astype(int)

    return year_2003_2005, year_2006_2009, year_2010_2012, year_2013_2016

def get_polar_coord(dataset):
    x, y = dataset['X'], dataset['Y']
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan(y / x)

    return rho, phi

def rotate_coord(dataset, degrees):
    x, y = dataset['X'], dataset['Y']
    # Source: https://en.wikipedia.org/wiki/Rotation_of_axes
    rotated_x = (x * np.cos(degrees)) + (y * np.sin(degrees))
    rotated_y = (y * np.cos(degrees)) - (x * np.sin(degrees))
    
    return rotated_x, rotated_y

def pca_decomp_coord(dataset, n_components):
    pca = PCA(n_components = n_components)
    pca_trans = pca.fit_transform(dataset[['X','Y']])
    
    return pca_trans

def get_nearest_station_distance(dataset):
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
    d = station_distances.min(axis = 0) # Take min index wise

    return d 

def get_nearest_station_bearing(dataset):
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
    b = bearing(x, y, x_station[min_index], y_station[min_index])

    return b

def center_and_scale(dataset):
    label = dataset['Category']
    data = dataset.drop(columns = ['Category'])
    
    data_scaled = StandardScaler().fit_transform(data)
    dataset = pd.DataFrame(data_scaled, columns = data.columns)
    dataset['Category'] = label
    
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
