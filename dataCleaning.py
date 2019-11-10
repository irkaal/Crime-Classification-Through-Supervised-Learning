import pandas as pd
import numpy as np
import datetime
import time
import sys


def mainClean(dataset):
    tic = time.time()
    print('Cleaning...', flush = True)

    df = pd.DataFrame(dataset)
    df = dropDescRes(df) 
    df = separateDistrict(df) # data clean district
    df = separateDayOfWeek(df) # data clean days of week
    df = encodeAddress(df)
    try:
        dates = getDateObjects(df)
        df = df.drop('Dates', axis = 1)
    except:
        updateProgress(13, 'Done')
        print(f'\nElapsed time: {time.time() - tic} seconds', flush = True)
        return df
    df = separateTimeByFourPeriods(df, dates) # data clean by time of the day
    df = separateTimeBySeasons(df, dates) # data clean by seasons
    df = encodeDayOfYear(df, dates)
    df = encodeDayOfMonth(df, dates)
    df = encodeMonth(df, dates)
    df = encodeYear(df, dates)
    df = encodeHour(df, dates)
    # TODO: keep holiday (1 or 0) (parse all holidays)
    # TODO: do research on events

    updateProgress(13, 'Done')
    print(f'\nElapsed time: {time.time() - tic} seconds', flush = True)
    return df


def updateProgress(i, task):
    sys.stdout.write('\r[%-52s] %d%% (%s)%s' % ('====' * i, 100 / 13 * i, task, ' ' * (36 - len(task))))
    sys.stdout.flush()


def dropDescRes(dataset):
    updateProgress(1, 'Drop Descript and Resolution columns')
    if 'Descript' in dataset:
        dataset = dataset.drop('Descript', axis = 1)
    if 'Resolution' in dataset:
        dataset = dataset.drop('Resolution', axis = 1)
    return dataset


# get one hot encoding of column PdDistrict
def separateDistrict(dataset):
    updateProgress(2, 'Encoding PdDistrict')
    if 'PdDistrict' in dataset:
        one_hot_day = pd.get_dummies(dataset['PdDistrict'])
        dataset = dataset.drop('PdDistrict', axis = 1) # Drop column PdDistrict as it is now encoded
        dataset = dataset.join(one_hot_day) # Join the encoded df
    return dataset


# Cyclic encoding helper
def encodeCyclic(dataset, colName, maxVal):
    if colName in dataset:
        dataset[colName + '_X'] = np.cos(2 * np.pi * dataset[colName] / maxVal)
        dataset[colName + '_Y'] = np.sin(2 * np.pi * dataset[colName] / maxVal)
        dataset = dataset.drop(colName, axis = 1)
    return dataset


# get one hot encoding of column DayOfWeek
def separateDayOfWeek(dataset):
    updateProgress(3, 'Encoding DayOfWeek')
    if 'DayOfWeek' in dataset:
        days = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        dataset['DayOfWeek'] = [days[d] for d in dataset['DayOfWeek']]
        dataset = encodeCyclic(dataset, 'DayOfWeek', 7)
    return dataset


# Binary encoding for Address Type
def encodeAddress(dataset):
    updateProgress(4, 'Encoding Address')
    if 'Address' in dataset:
        dataset['Intersection'] = [int('/' in a) for a in dataset['Address']]
        dataset = dataset.drop('Address', axis = 1)
    return dataset


def getDateObjects(dataset):
    updateProgress(5, 'Retrieving Dates')
    try:
        dateObjects = [datetime.datetime(*map(int, [d[:4], d[5:7], d[8:10], d[11:13], d[14:16]])) for d in dataset['Dates']]
    except:
        raise
    return dateObjects


# get one hot encoding of column Date by 6 hour periods
def separateTimeByFourPeriods(dataset, dateObjects):
    updateProgress(6, 'Encoding 6-Hour Period')
    n = len(dataset)
    zeros = [0] * n
    eighteenPeriod = zeros
    twelvePeriod = zeros
    sixPeriod = zeros
    zeroPeriod = zeros
    for i in range(0, n): # for each element in feature
        dateHour = dateObjects[i].hour
        if dateHour >= 18: # if time is above 18:00
            eighteenPeriod[i] = 1
        elif dateHour >= 12: # else if time is above 12:00
            twelvePeriod[i] = 1
        elif dateHour >= 6: # else if time is above 6:00
            sixPeriod[i] = 1
        elif dateHour >= 0: # else if time is above 00:00
            zeroPeriod[i] = 1 
    dataset.insert(2, "18:00-23:59", eighteenPeriod, True) 
    dataset.insert(2, "12:00-17:59", twelvePeriod, True) 
    dataset.insert(2, "6:00-11:59", sixPeriod, True)
    dataset.insert(2, "00:00-5:59", zeroPeriod, True)
    return dataset
  

# get one hot encoding of column Date by four seasons
def separateTimeBySeasons(dataset, dateObjects):
    updateProgress(7, 'Encoding Season')
    n = len(dataset)
    zeros = [0] * n
    spring = zeros # spring (March, April, May)
    summer = zeros # summer (June, July, August)
    autumn = zeros # autumn (September, October, November)
    winter = zeros # winter (December, January, February)
    for i in range(0, n): # for each element in feature
        dateMonth = dateObjects[i].month
        if dateMonth >= 3 and dateMonth <= 5: # spring
            spring[i] = 1
        elif dateMonth >= 6 and dateMonth <= 8: # summer
            summer[i] = 1
        elif dateMonth >= 9 and dateMonth <= 11: # autumn
            autumn[i] = 1
        elif dateMonth == 12 or dateMonth <= 2: # winter
            winter[i] = 1
    dataset.insert(2, "Spring", spring, True)
    dataset.insert(2, "Summer", summer, True) 
    dataset.insert(2, "Autumn", autumn, True)
    dataset.insert(2, "Winter", winter, True)
    return dataset


# Cyclic encoding for Day of Year
def encodeDayOfYear(dataset, dateObjects):
    updateProgress(8, 'Encoding DayOfYear')
    dataset['DayOfYear'] = [d.toordinal() - datetime.date(d.year, 1, 1).toordinal() + 1 for d in dateObjects]
    dataset = encodeCyclic(dataset, 'DayOfYear', 365)
    return dataset


# Cyclic encoding for Day of Month
def encodeDayOfMonth(dataset, dateObjects):
    updateProgress(9, 'Encoding DayOfMonth')
    dataset['DayOfMonth'] = [d.day for d in dateObjects]
    dataset = encodeCyclic(dataset, 'DayOfMonth', 31)
    return dataset


# Cyclic encoding for Month
def encodeMonth(dataset, dateObjects):
    updateProgress(10, 'Encoding Month')
    dataset['Month'] = [d.month for d in dateObjects]
    dataset = encodeCyclic(dataset, 'Month', 12)
    return dataset


# One-hot encoding for Year
def encodeYear(dataset, dateObjects):
    updateProgress(11, 'Encoding Year')
    dataset['Year'] = [d.year for d in dateObjects]
    one_hot_year = pd.get_dummies(dataset['Year'])
    dataset = dataset.drop('Year', axis = 1) # Drop column Year as it is now encoded
    dataset = dataset.join(one_hot_year)
    return dataset


# Cyclic encoding for Hour + Minute / 60
def encodeHour(dataset, dateObjects):
    updateProgress(12, 'Encoding Hour')
    dataset['Hour'] = [d.hour + d.minute / 60 for d in dateObjects]
    dataset = encodeCyclic(dataset, 'Hour', 24)
    return dataset


# if __name__== "__main__":
#     # import CSV:
#     try:
#         trainDataset = pd.read_csv('./data/train.csv')
#         mainClean(trainDataset) # specify a dataframe
#     except Exception as e:
#         print(e)
#         print("Can't find csv!")
