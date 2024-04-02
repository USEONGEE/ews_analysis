import numpy as np
import pandas as pd
import random
import tensorflow as tf
import os
import math

## pycaret
import pycaret.regression as reg
import pycaret.classification as cls
## FinBERT
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline


## LSTM
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import keras_tuner
from keras import layers
from . import preprocess as pp
## Regression
def auto_ml_reg(df, target, bank_name):
    df = df[df['bank.code'] == bank_name]
    df = df.drop(columns=['bank.code'])
    model = reg.setup(df, target = target, session_id= 42)
    best = reg.compare_models()
    
    linear_reg = reg.create_model('lr', cross_validation=True)
    ridge_reg = reg.create_model('ridge', cross_validation=True)
    xgb = reg.create_model('xgboost', cross_validation = True)
    lgb = reg.create_model('lightgbm', cross_validation = True)
    
    linear_port = 9000
    XGBoost_port = 9010
    lightGBM_port =9020
    GBM_port =9030
    ridge_port = 9040
    
    ## dashboard of linear
    reg.dashboard(linear_reg, run_kwargs={'port':linear_port})#, 'host':'0.0.0.0'})
    linear_address = pp.make_localhost(linear_port)
    
    ## dashboard of ridge
    reg.dashboard(ridge_reg, run_kwargs={'port':ridge_port})#, 'host':'0.0.0.0'})
    ridge_address = pp.make_localhost(ridge_port)
    
    ## dashboard of xgb
    reg.dashboard(xgb, run_kwargs={'port':XGBoost_port})#, 'host':'0.0.0.0'})
    xgb_address = pp.make_localhost(XGBoost_port)
    
    ## dashboard of lightgbm
    reg.dashboard(lgb, run_kwargs={'port':lightGBM_port})#, 'host':'0.0.0.0'})
    lightgbbm_address = pp.make_localhost(lightGBM_port)

    address_dic = {
        {
            "name": "lr",
            "url": linear_address
        },
        {
            "name": "ridge",
            "url": ridge_address
        },
        {
            "name": "xgboost",
            "url": xgb_address
        },
        {
            "name": "lightgbm",
            "url": lightgbbm_address
        }

    }
    return address_dic

def auto_ml_all(df, target):
    # df = df.drop(columns=['bank.code'])
    
    model = reg.setup(df, target = target, session_id= 42)
    best = reg.compare_models()
    
    linear_reg = reg.create_model('lr', cross_validation=True)
    ridge_reg = reg.create_model('ridge', cross_validation=True)
    xgb = reg.create_model('xgboost', cross_validation = True)
    lgb = reg.create_model('lightgbm', cross_validation = True)
    
    linear_port = 9050
    XGBoost_port = 9060
    lightGBM_port =9070
    GBM_port =9080
    ridge_port = 9090
    
    ## dashboard of linear
    reg.dashboard(linear_reg, run_kwargs={'port':linear_port})#, 'host':'0.0.0.0'})
    linear_address = pp.make_localhost(linear_port)
    
    ## dashboard of ridge
    reg.dashboard(ridge_reg, run_kwargs={'port':ridge_port})#, 'host':'0.0.0.0'})
    ridge_address = pp.make_localhost(ridge_port)
    
    ## dashboard of xgb
    reg.dashboard(xgb, run_kwargs={'port':XGBoost_port})#, 'host':'0.0.0.0'})
    xgb_address = pp.make_localhost(XGBoost_port)
    
    ## dashboard of lightgbm
    reg.dashboard(lgb, run_kwargs={'port':lightGBM_port})#, 'host':'0.0.0.0'})
    lightgbbm_address = pp.make_localhost(lightGBM_port)
    

    address_dic = {
     {
          "name" : "lr",
          "url" : linear_address
     },
     {
          "name" : "ridge",
          "url" : ridge_address
     },
     {
          "name" : "xgboost",
          "url" : xgb_address
     },
     {
          "name" : "lightgbm",
          "url" : lightgbbm_address
     },
}
    return address_dic

# def lstm_model(df,target,feature):  # 미완성
#     ## data prep
#     x_train, x_val, x_test, y_train, y_val, y_test = pp.lstm_train_test(df,target,feature)
#
#     ## lstm model
#     model = Sequential()
#     model.add(LSTM(50, activation='relu', input_shape=(1,time_stemp)))
#     model.add(Dense(1))
#     model.compile(optimizer='adam', loss='mse')
#     model.fit(x_train, y_train,
#               validation_data=(x_val, y_val),
#               epochs=50,
#               batch_size=1)
#
#     ## hp tuning
#
#     return None

## classification 추가 예정
def auto_ml_class(df, target, bank_name):
    df = df[df['bank.code']==bank_name]
    df = df.drop(columns = ['bank.code'])
    df[target]= pd.qcut(df[target], q = 4, labels = [f'range_{i}' for i in range(0,4)])
    model = cls.setup(df, target = target, train_size = 0.8)
   
    fold_value = math.floor((df[target].value_counts().min())/2)

    best = cls.compare_models(fold = fold_value)
    
#     gbc = create_model('gbc',cross_validation=True, fold = fold_value)
    xgb = cls.create_model('xgboost',cross_validation=True, fold = fold_value)
    lightgbm = cls.create_model('lightgbm',cross_validation=True, fold = fold_value)
    lr = cls.create_model('lr',cross_validation=True, fold = fold_value)
    
#     gbc_port = 9000
    xgb_port = 9100
    lightgbm_port =9110
    logistic_port = 9120
    
    ## dashboard of gbc 지원 안 함
#     dashboard(gbc, run_kwargs={'port':gbc_port})#, 'host':'0.0.0.0'})
#     gbc_address = make_localhost(gbc_port)
    
    ## dashboard of xgboost
    cls.dashboard(xgb, run_kwargs={'port':xgb_port})#, 'host':'0.0.0.0'})
    xgboost_address = pp.make_localhost(xgb_port)
    
    ## dashboard of lightgbm
    cls.dashboard(lightgbm, run_kwargs={'port':lightgbm_port})#, 'host':'0.0.0.0'})
    lightGBM_address = pp.make_localhost(lightgbm_port)
    
    ## dashboard of logitic_reg
    cls.dashboard(lr, run_kwargs={'port':logistic_port})#, 'host':'0.0.0.0'})
    logistic_address = pp.make_localhost(logistic_port)
    
    address_dic = {
        {
            "name": "xgb",
            "url": xgboost_address
        },
        {
            "name": "lightGBM",
            "url": lightGBM_address
        },
        {
            "name": "lr",
            "url": logistic_address
        },

    }
    return address_dic
