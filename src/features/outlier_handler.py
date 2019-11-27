import numpy as np
import pandas as pd


def handle_outliers(dataset):
    # Define outlier boundaries. These are just estimates from google map.
    min_X, max_X, min_Y, max_Y = -122.515465, -122.356443, 37.707462, 37.834977
    filter_X = np.logical_or(dataset['X'] < min_X, dataset['X'] > max_X)
    filter_Y = np.logical_or(dataset['Y'] < min_Y, dataset['Y'] > max_Y)
    outlier_filter = np.logical_or(filter_X, filter_Y)
    outlier = dataset[outlier_filter]

    # 1st Pass (Use matching address coordinates)
    match_filter = np.logical_and(dataset['Address'].isin(outlier['Address']), ~outlier_filter)
    match = dataset[match_filter]
    # Sort the data.table by Address to allow factor indexing trick to work
    match = match.sort_values(by=['Address']).reset_index()
    # Replace the incorrect coordinates using factor index
    replace_filter = np.logical_and(dataset['Address'].isin(match['Address']), outlier_filter)
    replace_index = pd.factorize(dataset[replace_filter]['Address'], sort = True)[0]
    dataset['X'].loc[replace_filter] = match['X'].loc[replace_index].values
    dataset['Y'].loc[replace_filter] = match['Y'].loc[replace_index].values

    # Update outliers
    filter_X = np.logical_or(dataset['X'] < min_X, dataset['X'] > max_X)
    filter_Y = np.logical_or(dataset['Y'] < min_Y, dataset['Y'] > max_Y)
    outlier_filter = np.logical_or(filter_X, filter_Y)
    outlier = dataset[outlier_filter]

    # 2nd Pass (Estimate the remaining outlier coordinates)
    # Get mean coordinates by PdDistrict
    non_outlier = dataset[~outlier_filter]
    avg_XY = non_outlier.groupby('PdDistrict')[['X', 'Y']].mean()
    # Replace the incorrect coordinates
    replace_index = pd.factorize(outlier['PdDistrict'], sort = True)[0]
    dataset['X'].loc[outlier_filter] = avg_XY['X'].iloc[replace_index].values
    dataset['Y'].loc[outlier_filter] = avg_XY['Y'].iloc[replace_index].values
    
    return dataset
