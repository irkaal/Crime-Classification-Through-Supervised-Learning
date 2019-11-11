from zipfile import ZipFile
import os
import pandas as pd
import numpy as np
import dataCleaning as dc
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import lightgbm as lgb

if __name__== "__main__":
    # Use current script dir as working dir
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Load Data
    # zip = ZipFile('data/sf-crime.zip')
    # crimeData = pd.read_csv(zip.open('train.csv'))
    crimeData = pd.read_csv('data/clean.csv')

    # Clean Data
    # source = ro.r['source']
    # source('pre_process.R')
    # preProcessR = ro.globalenv['preProcessR']    
    # with localconverter(ro.default_converter + pandas2ri.converter):
    #     rCrimeData = preProcessR(crimeData)
    # TODO: Convert rCrimeData back to Pandas df

    # LightGBM
    label = crimeData['Category']
    data = crimeData.drop('Category', axis = 1)
    train_data = lgb.Dataset(data, label = label)
    train_data.save_binary('train.bin')
    param = {'objective': 'softmax', 'num_classes': 39, 'metric': 'multi_logloss'}
    bst = lgb.train(param, train_data)
    # Save model
    bst.save_model('model.txt')
    json_model = bst.dump_model()
    # Load saved model
    bst = lgb.Booster(model_file = 'model.txt')
    
    ypred = bst.predict(data)
