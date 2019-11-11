import pandas as pd
import numpy as np
import time
import sys
import os

def mainClean(dataset):
    tic = start()
    if set(['PdDistrict', 'Address', 'Dates', 'DayOfWeek', 'Descript', 'Resolution']).issubset(dataset.columns):
        # One-hot encoding of column PdDistrict
        dataset = dataset.join(pd.get_dummies(dataset['PdDistrict'], prefix = 'PdDistrict'))
        updateProgress(1, 'Encoded PdDistrict')
        # Encode Address by type (Is it an intersection?)
        dataset['Intersection'] = dataset['Address'].map(lambda a: int('/' in a))
        updateProgress(2, 'Encoded Address')
        # Retrieve datetime indices
        dateIndices = pd.DatetimeIndex(dataset['Dates'])
        dateIndicesHour = dateIndices.hour
        dateIndicesDay = dateIndices.day
        dateIndicesMonth = dateIndices.month
        updateProgress(3, 'Retrieved Date Indices')
        # Encode time into 6-Hour Period
        dataset = encodePeriod(dataset, dateIndicesHour)
        updateProgress(4, 'Encoded 6-Hour Period')
        # Encode month into 4 Seasons
        dataset = encodeSeason(dataset, dateIndicesMonth)
        updateProgress(5, 'Encoded Season')
        # One-hot encoding of column DayOfWeek
        dataset = encodeCyclic(dataset, 'DayOfWeek', dateIndices.dayofweek, 7) 
        updateProgress(6, 'Encoded DayOfWeek')
        # Cyclic encoding for Day of Year
        dataset = encodeCyclic(dataset, 'DayOfYear', dateIndices.dayofyear, 365)
        updateProgress(7, 'Encoded PdDistrict')
        # Cyclic encoding for Day of Month
        dataset = encodeCyclic(dataset, 'DayOfMonth', dateIndicesDay, 31)
        updateProgress(8, 'Encoded PdDistrict')
        # Cyclic encoding for Month
        dataset = encodeCyclic(dataset, 'Month', dateIndicesMonth, 12)
        updateProgress(9, 'Encoded PdDistrict')
        # One-hot encoding for Year
        dataset = dataset.join(pd.get_dummies(dateIndices.year, prefix = 'Year'))
        updateProgress(10, 'Encoded PdDistrict')
        # Cyclic encoding for Hour + Minute / 60
        dataset = encodeCyclic(dataset, 'Hour', dateIndicesHour + dateIndices.minute / 60, 24)
        updateProgress(11, 'Encoded PdDistrict')
        # Encode holidays
        dataset = encodeHolidays(dataset, dateIndicesDay, dateIndicesMonth)
        updateProgress(12, 'Encoded Holidays')
        # Drop unused columns
        dataset = dataset.drop(columns = ['PdDistrict', 'Address', 'Dates', 'DayOfWeek', 'Descript', 'Resolution'])
        updateProgress(12, 'Dropped unused columns') 
    end(tic)
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
    # New Years Day - January 1, 2019
    dataset['NewYearsDay'] = np.logical_and(day == 1, month == 1).astype(int)
    # Dr. Martin Luther King, Jr. Day -  January 21, 2019
    dataset['MartinLutherDay'] = np.logical_and(day == 21, month == 1).astype(int)
    # President's Day - February 18, 2019
    dataset['PresidentsDay'] = np.logical_and(day == 18, month == 2).astype(int)
    # Memorial Day -  May 27, 2019
    dataset['MemorialDay'] = np.logical_and(day == 27, month == 5).astype(int)
    # Independence Day -  July 4, 2019
    dataset['IndependenceDay'] = np.logical_and(day == 4, month == 7).astype(int)
    # Labor Day -  September 2, 2019
    dataset['LaborDay'] = np.logical_and(day == 2, month == 9).astype(int)
    # Indigenous Peoples Day - October 14, 2019
    dataset['IndigenousDay'] = np.logical_and(day == 14, month == 10).astype(int)
    # Veterans Day - November 11, 2019
    dataset['VeteransDay'] = np.logical_and(day == 11, month == 11).astype(int)
    # Thanksgiving Day - November 28, 2019
    dataset['ThanksgivingDay'] = np.logical_and(day == 28, month == 11).astype(int)
    # Christmas Day - December 25, 2019
    dataset['ChristmasDay'] = np.logical_and(day == 25, month == 12).astype(int)
    return dataset

# Progress helpers
def updateProgress(i, task):
    sys.stdout.write('\r[%-52s] %d%% (%s)%s' % ('====' * i, 100 / 13 * i, task, ' ' * (36 - len(task))))
    sys.stdout.flush()

def start():
    print('Cleaning...', flush = True)
    updateProgress(0, '')
    return time.time()

def end(tic):
    toc = time.time()
    updateProgress(13, 'Done')
    print(f'\nElapsed time: {time.time() - tic} seconds', flush = True)
    return toc - tic

# if __name__== "__main__":
#     try:
#         os.chdir(os.path.dirname(os.path.abspath(__file__)))
#         trainDataset = pd.read_csv('./data/train.csv')
#         trainDataset = mainClean(trainDataset)
#     except Exception as e:
#         print(e)
#         print("Can't find csv!")
