import pandas as pd
from pipeline.extract import df
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def dateTime(_reuslt):
    return _reuslt['Date'].apply(lambda x: pd.to_datetime(x))

def mmScaler(_result):
    scaler = MinMaxScaler()
    scaler.fit(_result['Price'].values.reshape(-1,1))
    df_scaled = scaler.transform(_result['Price'].values.reshape(-1,1))
    _result['Price_mm'] = df_scaled
    return _result 

def train_test_Split(_result):
    data = _result[['Price_mm']]
    x = data.values.tolist() 
    y = data.values.tolist()  
  
    sep = round(len(data)*0.8)
    train, test = x[0:sep], x[sep:len(data)]
    return train, test

def train_test_Dummy(train, test):
    train_dummy = np.append(train, np.repeat(train[-1], 4))
    test_dummy = np.append(test, np.repeat(test[-1], 4))
    return train_dummy, test_dummy


def convert_to_matrix(_result):
    data = _result[['Price_mm']]

    mtr_x, mtr_y = [], []
    for i in range(len(data) - 5):
        d = i + 5  
        mtr_x.append(data[i:d])
        mtr_y.append(data[d])

    return np.array(mtr_x), np.array(mtr_y)


def total_matrix(train_dummy, test_dummy):
    train_x, train_y = convert_to_matrix(train_dummy, 5) 
    test_x, test_y = convert_to_matrix(test_dummy, 5)
    return train_x, train_y, test_x, test_y


def final_Reshape(train_x, train_y):
    train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
    test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))
    return train_x, train_y