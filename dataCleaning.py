import os
import sys
import time
from zipfile import ZipFile
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.preprocessing import StandardScaler

# Global variables
i = 0
max_i = 0
tic = time.time()

def mainClean(dataset, centerScale = False):
    start(total = int(centerScale) + 13) # Start progress tracker
    # Check if dataset has all the necessary columns
    if set(['PdDistrict', 'Address', 'Dates', 'DayOfWeek', 'Descript', 'Resolution']).issubset(dataset.columns):
        # Encode PdDistrict (One-Hot encoding)
        updateProgress('Encoding PdDistrict')
        dataset = dataset.join(pd.get_dummies(dataset['PdDistrict'], prefix = 'PdDistrict'))
        
        # Encode Patrol Division (Label encoding - 0 for Metro and 1 for GoldenGate)        
        updateProgress('Encoding PatrolDivision')
        dataset = encodePatrolDiv(dataset)
        
        # Encode Address by type (Label encoding - 0 for not an intersection and 1 for an intersection)
        updateProgress('Encoding Address')
        dataset['Intersection'] = dataset['Address'].map(lambda a: int('/' in a))
        
        # Retrieve DatetimeIndex
        updateProgress('Retrieving DatetimeIndex')
        dateIndices = pd.DatetimeIndex(dataset['Dates'])
        dateIndicesHour = dateIndices.hour
        dateIndicesDay = dateIndices.day
        
        # Encode DayOfWeek
        updateProgress('Encoding DayOfWeek')
        dataset = encodeCyclic(dataset, 'DayOfWeek', dateIndices.dayofweek, 7) 
        
        # Encode DayofYear
        updateProgress('Encoding DayOfYear')
        dataset = encodeCyclic(dataset, 'DayOfYear', dateIndices.dayofyear, 365)
        
        # Encode DayofMonth
        updateProgress('Encoding DayOfMonth')
        dataset = encodeCyclic(dataset, 'DayOfMonth', dateIndicesDay, 31)
        
        # Encode Year
        updateProgress('Encoding Year')
        dataset = dataset.join(pd.get_dummies(dateIndices.year, prefix = 'Year'))
       
        # Encode Hour + Minute / 60
        updateProgress('Encoding Hour')
        dataset = encodeCyclic(dataset, 'Hour', dateIndicesHour + dateIndices.minute / 60, 24)
        
        # TODO: Encode Event (Label encoding - 0 for not an event day and 1 for an event day)
        updateProgress('Encoding Event')
        dataset = encodeEvent(dataset, pd.to_datetime(dateIndices))
        
        # TODO: Encode Geospatial
        updateProgress('Encoding Geospatial') 
        dataset = encodeGeospatial(dataset)
        
        # Drop unused columns
        updateProgress('Dropping unused columns') 
        dataset = dataset.drop(columns = ['PdDistrict', 'Address', 'Dates', 'DayOfWeek', 'Descript', 'Resolution'])

        # Center and scale
        if (centerScale):
            updateProgress('Centering and scaling features')
            dataset = centerScaleFeatures(dataset)
    end() # End progress tracker
    return dataset

def encodePatrolDiv(dataset):
    division = {'BAYVIEW': 0, 'MISSION': 0, 'PARK': 0, 'RICHMOND': 0, 'TARAVAL': 0, 
                'CENTRAL': 1, 'INGLESIDE': 1, 'NORTHERN': 1, 'SOUTHERN': 1, 'TENDERLOIN': 1} 
    dataset['Patrol_Div'] = dataset['PdDistrict'].map(division) 
    return dataset

def encodeEvent(dataset, datetime):
    # From Kaggle - 1/1/2003 to 5/13/2015
    holidays = USFederalHolidayCalendar().holidays(start = '2003-01-01', end = '2015-05-13')
    dataset['Event'] = datetime.isin(holidays).astype(int)
    return dataset

def encodeGeospatial(dataset):
    x, y = dataset['X'], dataset['Y']
    # Add distance to the closest police station.
    dataset['Station_Dist'] = np.array([
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

def centerScaleFeatures(dataset):
    y = dataset['Category']
    x = dataset.drop(columns = ['Category'])
    stdScale = StandardScaler().fit(x)
    xScaled = stdScale.transform(x)
    dataset = pd.DataFrame(xScaled, columns = x.columns)
    dataset['Category'] = y
    return dataset

# Helper functions

# Cyclic encoding helper
def encodeCyclic(dataset, colName, val, maxVal):
    dataset[colName + '_X'] = np.cos(2 * np.pi * val / maxVal)
    dataset[colName + '_Y'] = np.sin(2 * np.pi * val / maxVal)
    return dataset

# Haversine distance - Great-circle distance in kilometres
def haversine(x1, y1, x2, y2):
    lon1, lat1, lon2, lat2 = map(np.radians, [x1, y1, x2, y2])
    h = np.sin((lat2 - lat1) / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2)**2
    return 2 * 6378.137 * np.arcsin(np.sqrt(h))

# Progress tracker helpers

def updateProgress(task):
    global i, max_i
    progress = '=' * round(48 / max_i * i)
    space = ' ' * (30 - len(task))
    percent = round(100 / max_i * i, 3)
    i += 1 # increment task count
    sys.stdout.write('\r[%-48s] %d%% (%s)%s' % (progress, percent, task, space))
    sys.stdout.flush()

def start(total):
    global max_i
    max_i = total
    print('Python> Cleaning...', flush = True)
    updateProgress('')
    # return time.time()

def end():
    global tic
    updateProgress('Done') 
    print(f'\nElapsed time: {round(time.time() - tic, 3)} second(s)', flush = True)

# if __name__== "__main__":
#     try:
#         os.chdir(os.path.dirname(os.path.abspath(__file__)))
#         zip = ZipFile('data/sf-crime.zip')
#         train_data = pd.read_csv(zip.open('train.csv'))
#         mainClean(train_data, centerScale = True)
#     except Exception as e:
#         print(e)
#         print("Can't find csv!")
