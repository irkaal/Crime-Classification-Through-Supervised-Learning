import os
from zipfile import ZipFile
import sys
import time
import pandas as pd
import numpy as np

def mainClean(dataset):
    tic = start()
    if set(['PdDistrict', 'Address', 'Dates', 'DayOfWeek', 'Descript', 'Resolution']).issubset(dataset.columns):
        # Encode PdDistrict (One-Hot encoding)
        updateProgress(1, 'Encoding PdDistrict')
        dataset = dataset.join(pd.get_dummies(dataset['PdDistrict'], prefix = 'PdDistrict'))
        
        # Encode Patrol Division (Label encoding - 0 for Metro and 1 for GoldenGate)        
        updateProgress(2, 'Encoding PatrolDivision')
        dataset = encodePatrolDiv(dataset)
        
        # Encode Address by type (Label encoding - 0 for not an intersection and 1 for an intersection)
        updateProgress(3, 'Encoding Address')
        dataset['Intersection'] = dataset['Address'].map(lambda a: int('/' in a))
        
        # Retrieve DatetimeIndex
        updateProgress(4, 'Retrieving DatetimeIndex')
        dateIndices = pd.DatetimeIndex(dataset['Dates'])
        dateIndicesHour = dateIndices.hour
        dateIndicesDay = dateIndices.day
        dateIndicesMonth = dateIndices.month
        
        # Encode DayOfWeek
        updateProgress(5, 'Encoding DayOfWeek')
        dataset = encodeCyclic(dataset, 'DayOfWeek', dateIndices.dayofweek, 7) 
        
        # Encode DayofYear
        updateProgress(6, 'Encoding DayOfYear')
        dataset = encodeCyclic(dataset, 'DayOfYear', dateIndices.dayofyear, 365)
        
        # Encode DayofMonth
        updateProgress(7, 'Encoding DayOfMonth')
        dataset = encodeCyclic(dataset, 'DayOfMonth', dateIndicesDay, 31)
        
        # Encode Year
        updateProgress(8, 'Encoding Year')
        dataset = dataset.join(pd.get_dummies(dateIndices.year, prefix = 'Year'))
       
        # Encode Hour + Minute / 60
        updateProgress(9, 'Encoding Hour')
        dataset = encodeCyclic(dataset, 'Hour', dateIndicesHour + dateIndices.minute / 60, 24)
        
        # TODO: Encode Event (Label encoding - 0 for not an event day and 1 for an event day)
        updateProgress(10, 'Encoding Event')
        dataset = encodeEvent(dataset, dateIndicesDay, dateIndicesMonth)
        
        # TODO: Encode Geospatial
        updateProgress(11, 'Encoding Geospatial') 
        dataset = encodeGeospatial(dataset)
        
        # Drop unused columns
        updateProgress(12, 'Dropping unused columns') 
        dataset = dataset.drop(columns = ['PdDistrict', 'Address', 'Dates', 'DayOfWeek', 'Descript', 'Resolution'])
    end(tic)
    return dataset

def encodePatrolDiv(dataset):
    division = {'BAYVIEW': 0, 'MISSION': 0, 'PARK': 0, 'RICHMOND': 0, 'TARAVAL': 0, 
                'CENTRAL': 1, 'INGLESIDE': 1, 'NORTHERN': 1, 'SOUTHERN': 1, 'TENDERLOIN': 1} 
    dataset['Patrol_Div'] = dataset['PdDistrict'].map(division) 
    return dataset

def encodeEvent(dataset, day, month):
    # TODO:
    dataset['Event'] = (day == 1).astype(int)
    # dataset['NewYearsDay'] = np.logical_and(day == 1, month == 1).astype(int)
    # dataset['MartinLutherDay'] = np.logical_and(day == 21, month == 1).astype(int)
    # dataset['PresidentsDay'] = np.logical_and(day == 18, month == 2).astype(int)
    # dataset['MemorialDay'] = np.logical_and(day == 27, month == 5).astype(int)
    # dataset['IndependenceDay'] = np.logical_and(day == 4, month == 7).astype(int)
    # dataset['LaborDay'] = np.logical_and(day == 2, month == 9).astype(int)
    # dataset['IndigenousDay'] = np.logical_and(day == 14, month == 10).astype(int)
    # dataset['VeteransDay'] = np.logical_and(day == 11, month == 11).astype(int)
    # dataset['ThanksgivingDay'] = np.logical_and(day == 28, month == 11).astype(int)
    # dataset['ChristmasDay'] = np.logical_and(day == 25, month == 12).astype(int)
    return dataset

# Encode Geospatial
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
        ]).min(axis = 0)
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

# Progress tracker
def updateProgress(i, task):
    progress = '=' * round(48 / 13 * i)
    space = ' ' * (30 - len(task))
    percent = round(100 / 13 * i, 3)
    sys.stdout.write('\r[%-48s] %d%% (%s)%s' % (progress, percent, task, space))
    sys.stdout.flush()

def start():
    print('Python> Cleaning...', flush = True)
    updateProgress(0, '')
    return time.time()

def end(tic):
    updateProgress(13, 'Done')
    print(f'\nElapsed time: {round(time.time() - tic, 3)} second(s)', flush = True)

# if __name__== "__main__":
#     try:
#         os.chdir(os.path.dirname(os.path.abspath(__file__)))
#         zip = ZipFile('data/sf-crime.zip')
#         train_data = pd.read_csv(zip.open('train.csv'))
#         mainClean(train_data)
#     except Exception as e:
#         print(e)
#         print("Can't find csv!")
