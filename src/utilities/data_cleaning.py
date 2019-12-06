import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import src.utilities.feature_engineering as fe


def main_clean(dataset, pca_fit = None):
    # PdDistrict
    # One-hot encoding
    dataset = fe.encode_district(dataset)

    # Patrol Division 
    # 0 for Metro and 1 for GoldenGate
    dataset = fe.insert_patrol_division(dataset)

    # Intersection
    # Group addresses by type 
    # 0 for not an intersection and 1 for an intersection
    dataset = fe.insert_intersection(dataset)

    # Day of Week 
    # Cyclic representation
    dataset = fe.encode_day_of_week(dataset)

    # Day of Year
    # Cyclic representation
    dataset = fe.encode_day_of_year(dataset)

    # Hour 
    # Cyclic representation
    dataset = fe.encode_hour(dataset)

    # Periods
    # 00:00-05:59, 06:00-17:59, 18:00-23:59
    dataset = fe.insert_periods(dataset) 

    # Years
    # 2003-2005, 2006-2009, 2010-2012, 2013-2016
    dataset = fe.insert_year_groups(dataset)

    # Polar Coordinates
    dataset = fe.insert_polar(dataset)

    # Rotation of Coordinates
    # Rotation by 30 degrees
    dataset = fe.insert_rotation(dataset, degrees = 30)
    # Rotation by 60 degrees
    dataset = fe.insert_rotation(dataset, degrees = 60)
    # Rotation by PCA
    pca_tuple = fe.insert_pca_rotation(dataset, pca_fit)
    dataset = pca_tuple[0]
    pca_fit = pca_tuple[1]

    # Nearest Police Station Distance
    dataset = fe.insert_nearest_station(dataset)

    # Nearest Police Station Bearing
    dataset = fe.insert_nearest_station_bearing(dataset)

    # Remove unused columns from dataframe
    dataset = dataset.drop(columns = ['PdDistrict', 'Address', 'DayOfWeek', 'Dates'])

    return dataset, pca_fit
