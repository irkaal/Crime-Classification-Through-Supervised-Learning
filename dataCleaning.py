import pandas as pd
import numpy as np
import datetime

# TODO: take in a dataframe argument
def mainClean(dataset):

    # getting size of dataset: eg, (878049, 9)
    # print(dataset.shape)

    df = pd.DataFrame(dataset)
    df = separateDistrict(df) # data clean district
    df = separateDayOfWeek(df) # data clean days of week
    df = separateTimeByFourPeriods(df) # data clean by time of the day
    df = separateTimeBySeasons(df) # data clean by seasons
    # TODO: keep Day of Year (cont.), Day of Month (cont.), Month, Year
    # TODO: keep hours 0 - 23 (cont.), minute 0 - 59 (cont.)
    # TODO: keep holiday (1 or 0) (parse all holidays)
    # TODO: do research on events

    # uncomment if you want to save to csv:
    # df.to_csv('./cleanedDataset.csv')

    print(df) # or df.head()
    return df


# get one hot encoding of column DayOfWeek
def separateDayOfWeek(dataset):
    print("Clean DayOfWeek")
    one_hot_day = pd.get_dummies(dataset['DayOfWeek'])
    dataset = dataset.drop('DayOfWeek', axis = 1) # Drop column DayOfWeek as it is now encoded
    dataset = dataset.join(one_hot_day) # Join the encoded df
    return dataset


# get one hot encoding of column PdDistrict
def separateDistrict(dataset):
    print("Clean PdDistrict")
    one_hot_day = pd.get_dummies(dataset['PdDistrict'])
    dataset = dataset.drop('PdDistrict', axis = 1) # Drop column DayOfWeek as it is now encoded
    dataset = dataset.join(one_hot_day) # Join the encoded df
    return dataset


# get one hot encoding of column Date by 6 hour periods
def separateTimeByFourPeriods(dataset):
    print("Clean Hours")
    eighteenPeriod = [0] * len(dataset)
    twelvePeriod = [0] * len(dataset)
    sixPeriod = [0] * len(dataset)
    zeroPeriod = [0] * len(dataset)
    for i in range(0,(len(dataset))): # for each element in feature
        try:
            dateObject = datetime.datetime.strptime(str(dataset['Dates'][i]), "%Y-%m-%d %H:%M:%S")
        except:
            try:
                dateObject = datetime.datetime.strptime(str(dataset['Dates'][i]), "%Y-%m-%d %H:%M") # if CSV doesn't have seconds
            except:
                print("Failed to encode hours")
                return dataset
        dateHour = dateObject.hour
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
    # dataset = dataset.drop('Dates',axis = 1) # Drop column DayOfWeek as it is now encoded
    return dataset
  

# get one hot encoding of column Date by four seasons
def separateTimeBySeasons(dataset):
    print("Clean Seasons")
    spring = [0] * len(dataset) # spring (March, April, May)
    summer = [0] * len(dataset) # summer (June, July, August)
    autumn = [0] * len(dataset) # autumn (September, October, November)
    winter = [0] * len(dataset) # winter (December, January, February)
    for i in range(0,(len(dataset))): # for each element in feature
        try:
            dateObject = datetime.datetime.strptime(str(dataset['Dates'][i]), "%Y-%m-%d %H:%M:%S")
        except:
            try:
                dateObject = datetime.datetime.strptime(str(dataset['Dates'][i]), "%Y-%m-%d %H:%M") # if CSV doesn't have seconds
            except:
                print("Failed to encode seasons")
                return dataset
        dateMonth = dateObject.month
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
    dataset = dataset.drop('Dates',axis = 1) # drop column DayOfWeek as it is now encoded
    return dataset

  
if __name__== "__main__":
    # import CSV:
    try:
        trainDataset = pd.read_csv("./data/sf-crime/train.csv")
        mainClean(trainDataset) # specify a dataframe
    except Exception as e:
        # print(e)
        print("Can't find csv!")