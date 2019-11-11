import os
from zipfile import ZipFile
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
    zip = ZipFile('data/sf-crime.zip')
    train_data = pd.read_csv(zip.open('train.csv'))

    # Clean Data
    source = ro.r['source']
    source('pre_process.R')
    preProcessPy = ro.globalenv['preProcessPy']    
    with localconverter(ro.default_converter + pandas2ri.converter):
        preProcessPy(train_data)
    train_data = pd.read_csv('data/train_rclean.csv')
    os.remove('data/train_rclean.csv')
    train_data = dc.mainClean(train_data)

    print(train_data)
    print(train_data.info())

    # LightGBM
    # label = train_data['Category']
    # data = train_data.drop('Category', axis = 1)
    # train_data = lgb.Dataset(data, label = label)
    # train_data.save_binary('train.bin')
    # param = {'objective': 'softmax', 'num_classes': 39, 'metric': 'multi_logloss'}
    # bst = lgb.train(param, train_data)
    # # Save model
    # bst.save_model('model.txt')
    # json_model = bst.dump_model()
    # # Load saved model
    # bst = lgb.Booster(model_file = 'model.txt')
    
    # ypred = bst.predict(data)
