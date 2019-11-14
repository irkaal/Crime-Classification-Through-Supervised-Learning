import os
import sys
import time
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.preprocessing import StandardScaler
from zipfile import ZipFile
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
        
        # TODO: Encode Event (Label encoding - 0 for not an event day and 1 for an event day)
        update_progress('Encoding Event')
        dataset = encode_event(dataset, date_indices.date)
        
        # TODO: Encode Geospatial
        update_progress('Encoding Geospatial') 
        dataset = encode_geospatial(dataset)
        
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

def encode_event(dataset, date_indices_date):
    holidays = pd.Series(USFederalHolidayCalendar().holidays(start = '2003-01-01', end = '2015-05-13'))
    # This will not include original time info which makes comparison possible
    datetime = pd.to_datetime(date_indices_date)
    dataset['Event'] = pd.Series(datetime).isin(holidays).astype(int)
    return dataset

def encode_geospatial(dataset):
    x, y = dataset['X'], dataset['Y']
    # Add distance to the closest police station
    dataset['NearestStation'] = np.array([
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
    global tic
    update_progress('Done') 
    print(f'\nElapsed time: {round(time.time() - tic, 3)} second(s)', flush = True)


# if __name__== "__main__":
#     try:
#         os.chdir(os.path.dirname(os.path.abspath(__file__)))
#         zipFile = ZipFile('data/sf-crime.zip')
#         dataset = pd.read_csv(zipFile.open('train.csv'))
#         main_clean(dataset, center_scale = True)
#     except Exception as e:
#         print(e)
#         print("Can't find csv!")
