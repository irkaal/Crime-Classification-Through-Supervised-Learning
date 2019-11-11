import os
from zipfile import ZipFile
import sys
import time
import pandas as pd
import numpy as np

def mainClean(dataset):
    tic = start()
    if set(['PdDistrict', 'Address', 'Dates', 'DayOfWeek', 'Descript', 'Resolution']).issubset(dataset.columns):
        # One-hot encoding of column PdDistrict
        updateProgress(1, 'Encoding PdDistrict')
        dataset = dataset.join(pd.get_dummies(dataset['PdDistrict'], prefix = 'PdDistrict'))
        # Encode Patrol Division
        updateProgress(2, 'Encoding PatrolDivision')
        dataset = encodePatrolDiv(dataset)
        # Encode Address by type (Is it an intersection?)
        updateProgress(3, 'Encoding Address')
        dataset['Intersection'] = dataset['Address'].map(lambda a: int('/' in a))
        # Retrieve datetime indices
        updateProgress(4, 'Retrieving Date indices')
        dateIndices = pd.DatetimeIndex(dataset['Dates'])
        dateIndicesHour = dateIndices.hour
        dateIndicesDay = dateIndices.day
        dateIndicesMonth = dateIndices.month
        # Encode time into 6-Hour Period
        updateProgress(5, 'Encoding 6-Hour Period')
        dataset = encodePeriod(dataset, dateIndicesHour)
        # Encode month into 4 Seasons
        updateProgress(6, 'Encoding Season')
        dataset = encodeSeason(dataset, dateIndicesMonth)
        # One-hot encoding of column DayOfWeek
        updateProgress(7, 'Encoding DayOfWeek')
        dataset = encodeCyclic(dataset, 'DayOfWeek', dateIndices.dayofweek, 7) 
        # Cyclic encoding for Day of Year
        updateProgress(8, 'Encoding DayOfYear')
        dataset = encodeCyclic(dataset, 'DayOfYear', dateIndices.dayofyear, 365)
        # Cyclic encoding for Day of Month
        updateProgress(9, 'Encoding DayOfMonth')
        dataset = encodeCyclic(dataset, 'DayOfMonth', dateIndicesDay, 31)
        # Cyclic encoding for Month
        updateProgress(10, 'Encoding Month')
        dataset = encodeCyclic(dataset, 'Month', dateIndicesMonth, 12)
        # One-hot encoding for Year
        updateProgress(11, 'Encoding Year')
        dataset = dataset.join(pd.get_dummies(dateIndices.year, prefix = 'Year'))
        # Cyclic encoding for Hour + Minute / 60
        updateProgress(12, 'Encoding Hour')
        dataset = encodeCyclic(dataset, 'Hour', dateIndicesHour + dateIndices.minute / 60, 24)
        # Encode holidays
        updateProgress(13, 'Encoding Holidays')
        dataset = encodeHolidays(dataset, dateIndicesDay, dateIndicesMonth)
        # Encode geospatial
        updateProgress(14, 'Encoding geospatial') 
        dataset = encodeGeospatial(dataset)
        # Drop unused columns
        updateProgress(15, 'Dropping unused columns') 
        dataset = dataset.drop(columns = ['PdDistrict', 'Address', 'Dates', 'DayOfWeek', 'Descript', 'Resolution'])
    end(tic)
    return dataset

# Encode PdDistrict to 2 Patrol Divisions
def encodePatrolDiv(dataset):
    division = {'CENTRAL': 0, 'INGLESIDE': 0, 'NORTHERN': 0, 'SOUTHERN': 0, 'TENDERLOIN': 0,
                'BAYVIEW': 1, 'MISSION': 1, 'PARK': 1, 'RICHMOND': 1, 'TARAVAL': 1}
    districtToDiv = dataset['PdDistrict'].map(division)
    dataset['MetroDiv'] = districtToDiv ^ 1
    dataset['GoldenGateDiv'] = districtToDiv
    return dataset

# One-hot encoding of column Date by 6 hour periods
def encodePeriod(dataset, hour):
    period = np.floor(hour / 6) % 6
    dataset['00:00-05:59'] = (period == 0).astype(int)
    dataset['06:00-11:59'] = (period == 1).astype(int)
    dataset['12:00-17:59'] = (period == 2).astype(int)
    dataset['18:00-23:59'] = (period == 3).astype(int)
    return dataset

# One-hot encoding of column Date by four seasons
def encodeSeason(dataset, month):
    season = np.floor(month / 12 * 4) % 4
    dataset['Winter'] = (season == 0).astype(int) # Winter (December, January, February)
    dataset['Spring'] = (season == 1).astype(int) # Spring (March, April, May)
    dataset['Summer'] = (season == 2).astype(int) # Summer (June, July, August)
    dataset['Autumn'] = (season == 3).astype(int) # Autumn (September, October, November)
    return dataset

# Cyclic encoding helper
def encodeCyclic(dataset, colName, val, maxVal):
    dataset[colName + '_X'] = np.cos(2 * np.pi * val / maxVal)
    dataset[colName + '_Y'] = np.sin(2 * np.pi * val / maxVal)
    return dataset

def encodeHolidays(dataset, day, month):
    # New Years Day - January 1
    dataset['NewYearsDay'] = np.logical_and(day == 1, month == 1).astype(int)
    # Dr. Martin Luther King, Jr. Day -  January 21, 2019
    dataset['MartinLutherDay'] = np.logical_and(day == 21, month == 1).astype(int)
    # President's Day - February 18, 2019
    dataset['PresidentsDay'] = np.logical_and(day == 18, month == 2).astype(int)
    # Memorial Day -  May 27, 2019
    dataset['MemorialDay'] = np.logical_and(day == 27, month == 5).astype(int)
    # Independence Day -  July 4
    dataset['IndependenceDay'] = np.logical_and(day == 4, month == 7).astype(int)
    # Labor Day -  September 2, 2019
    dataset['LaborDay'] = np.logical_and(day == 2, month == 9).astype(int)
    # Indigenous Peoples Day - October 14, 2019
    dataset['IndigenousDay'] = np.logical_and(day == 14, month == 10).astype(int)
    # Veterans Day - November 11
    dataset['VeteransDay'] = np.logical_and(day == 11, month == 11).astype(int)
    # Thanksgiving Day - November 28, 2019
    dataset['ThanksgivingDay'] = np.logical_and(day == 28, month == 11).astype(int)
    # Christmas Day - December 25
    dataset['ChristmasDay'] = np.logical_and(day == 25, month == 12).astype(int)
    return dataset

# Encode Geospatial
def encodeGeospatial(dataset):
    # Add distances between center of the city + all police stations and crime locations.
    x, y = dataset['X'], dataset['Y'] 
    # Center of city (https://www.citylab.com/design/2016/06/exact-center-of-san-francisco/486341/)
    dataset['CenterDist'] = haversine(x, y, -122.442500, 37.754540)
    # Coordinates obtained from google maps
    dataset['CentralDist'] = haversine(x, y, -122.409960, 37.798736) 
    dataset['IngleDist'] = haversine(x, y, -122.446261, 37.724694) 
    dataset['NorthDist'] = haversine(x, y, -122.432516, 37.780226)
    dataset['SouthDist'] = haversine(x, y, -122.389411, 37.772382)
    dataset['TenderDist'] = haversine(x, y, -122.412924, 37.783783)
    dataset['BayDist'] = haversine(x, y, -122.397771, 37.729825)
    dataset['MissionDist'] = haversine(x, y, -122.421951, 37.763013)
    dataset['ParkDist'] = haversine(x, y, -122.455391, 37.767835)
    dataset['RichDist'] = haversine(x, y, -122.464462, 37.780016)
    dataset['TaravalDist'] = haversine(x, y, -122.481516, 37.743755)
    return dataset


# Helper functions

# Haversine distance - Great-circle distance in kilometres
def haversine(x1, y1, x2, y2):
    lon1, lat1, lon2, lat2 = map(np.radians, (x1, y1, x2, y2))
    h = np.sin((lat2 - lat1) / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2)**2
    r = 6378.137 # Earth's radius
    d = 2 * r * np.arcsin(np.sqrt(h))
    return d

# Progress tracker
def updateProgress(i, task):
    sys.stdout.write('\r[%-48s] %d%% (%s)%s' % ('===' * i, 100 / 16 * i, task, ' ' * (23 - len(task))))
    sys.stdout.flush()

def start():
    print('Python> Cleaning...', flush = True)
    updateProgress(0, '')
    return time.time()

def end(tic):
    toc = time.time()
    updateProgress(16, 'Done')
    print(f'\nElapsed time: {round(time.time() - tic, 3)} second(s)', flush = True)
    return toc - tic

# if __name__== "__main__":
#     try:
#         os.chdir(os.path.dirname(os.path.abspath(__file__)))
#         zip = ZipFile('data/sf-crime.zip')
#         train_data = pd.read_csv(zip.open('train.csv'))
#         train_data = mainClean(train_data)
#     except Exception as e:
#         print(e)
#         print("Can't find csv!")
