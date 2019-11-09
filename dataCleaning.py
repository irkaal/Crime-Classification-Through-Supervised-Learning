import pandas as pd
import numpy as np
import datetime
import calendar


def mainClean(dataset):
    # getting size of dataset: eg, (878049, 9)
    # print(dataset.shape)

    df = pd.DataFrame(dataset)
    df = separateDistrict(df) # data clean district
    try:
        do = getDateObjects(df)
    except:
        print('Unable to retrieve Date Objects')
        return df
    df = separateDayOfWeek(df) # data clean days of week
    df = separateTimeByFourPeriods(df, do) # data clean by time of the day
    df = separateTimeBySeasons(df, do) # data clean by seasons
    df = encodeDayOfYear(df, do)
    df = encodeDayOfMonth(df, do)
    df = encodeMonth(df, do)
    df = encodeYear(df, do)
    df = encodeHour(df, do) 
    # TODO: keep holiday (1 or 0) (parse all holidays)
    # TODO: do research on events

    # uncomment if you want to save to csv:
    # df.to_csv('./cleanedDataset.csv')

    print(df) # or df.head()
    return df


def getDateObjects(dataset):
    print('Retrieve Date Objects')
    try:
        dateObjects = [datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in dataset['Dates']]       
    except:
        try:
            dateObjects = [datetime.datetime.strptime(d, "%Y-%m-%d %H:%M") for d in dataset['Dates']]  # if CSV doesn't have seconds
        except: 
            raise
    return dateObjects


# Cyclic encoding helper function
def encodeCyclic(dataset, colName, maxVal):
    dataset[colName + '_X'] = np.cos(2 * np.pi * dataset[colName] / maxVal)
    dataset[colName + '_Y'] = np.sin(2 * np.pi * dataset[colName] / maxVal)
    dataset = dataset.drop(colName, axis = 1)
    return dataset


# get one hot encoding of column DayOfWeek
def separateDayOfWeek(dataset):
    print("Clean DayOfWeek")
    days = dict(zip(calendar.day_name, range(7))); 
    dataset['DayOfWeek'] = [days[d] for d in dataset['DayOfWeek']]
    dataset = encodeCyclic(dataset, 'DayOfWeek', 7)
    return dataset


# get one hot encoding of column PdDistrict
def separateDistrict(dataset):
    print("Clean PdDistrict")
    one_hot_day = pd.get_dummies(dataset['PdDistrict'])
    dataset = dataset.drop('PdDistrict', axis = 1) # Drop column PdDistrict as it is now encoded
    dataset = dataset.join(one_hot_day) # Join the encoded df
    return dataset


# get one hot encoding of column Date by 6 hour periods
def separateTimeByFourPeriods(dataset, dateObjects):
    print("Clean Hours")
    eighteenPeriod = [0] * len(dataset)
    twelvePeriod = [0] * len(dataset)
    sixPeriod = [0] * len(dataset)
    zeroPeriod = [0] * len(dataset)
    for i in range(0,(len(dataset))): # for each element in feature
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
    print("Clean Seasons")
    spring = [0] * len(dataset) # spring (March, April, May)
    summer = [0] * len(dataset) # summer (June, July, August)
    autumn = [0] * len(dataset) # autumn (September, October, November)
    winter = [0] * len(dataset) # winter (December, January, February)
    for i in range(0,(len(dataset))): # for each element in feature
        dateMonth = dateObjects[i].month
        if dateMonth >= 3 and dateMonth <= 5: # spring
            spring[i] = 1
        elif dateMonth >= 6 and dateMonth <= 8: # summer
            summer[i] = 1
        elif dateMonth >= 9 and dateMonth <= 11: # autumn
            autumn[i] = 1
        elif dateMonth == 12 or dateMonth <= 2: # winter
            winter[i] = 1
    dataset.insert(2, "spring", spring, True)
    dataset.insert(2, "summer", summer, True) 
    dataset.insert(2, "autumn", autumn, True)
    dataset.insert(2, "winter", winter, True)
    return dataset


# Cyclic encoding for Day of Year
def encodeDayOfYear(dataset, dateObjects):
    print('Encode DayOfYear')
    dataset['DayOfYear'] = [d.toordinal() - datetime.date(d.year, 1, 1).toordinal() + 1 for d in dateObjects]
    dataset = encodeCyclic(dataset, 'DayOfYear', 365)
    return dataset


# Cyclic encoding for Day of Month
def encodeDayOfMonth(dataset, dateObjects):
    print('Encode DayOfMonth')
    dataset['DayOfMonth'] = [d.day for d in dateObjects]
    dataset = encodeCyclic(dataset, 'DayOfMonth', 31)
    return dataset


# Cyclic encoding for Month
def encodeMonth(dataset, dateObjects):
    print('Encode Month')
    dataset['Month'] = [d.month for d in dateObjects]
    dataset = encodeCyclic(dataset, 'Month', 12)
    return dataset


# One-hot encoding for Year
def encodeYear(dataset, dateObjects):
    print('Encode Year')
    dataset['Year'] = [d.year for d in dateObjects]
    one_hot_year = pd.get_dummies(dataset['Year'])
    dataset = dataset.drop('Year', axis = 1) # Drop column Year as it is now encoded
    dataset = dataset.join(one_hot_year)
    return dataset


# Cyclic encoding for Hour + Minute / 60
def encodeHour(dataset, dateObjects):
    print('Encode Hour')
    dataset['Hour'] = [d.hour + d.minute / 60 for d in dateObjects]
    dataset = encodeCyclic(dataset, 'Hour', 24)
    dataset = dataset.drop('Dates', axis = 1) # drop column Dates as it is now encoded
    return(dataset)


# if __name__== "__main__":
    # import CSV:
    # try:
        # trainDataset = pd.read_csv("./data/sf-crime/train.csv")
        # mainClean(trainDataset) # specify a dataframe
        # dataset = pd.read_csv("./Desktop/San-Francisco-Crime-Classification/data/train.csv")
    # except Exception as e:
        # print(e)
        # print("Can't find csv!")