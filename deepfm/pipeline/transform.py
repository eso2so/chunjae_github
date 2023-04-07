import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from deepctr.models import DeepFM


def df_notnull_rename(_result):
    df = _result[_result['rating'].notnull()]
    df_new = df.rename(columns={'display(in inch)': 'display', 'price(in Rs.)':'price'}, inplace=True)

def df_isnull_rename(_result):
    df = _result[_result['rating'].isnull()]
    df_new = df.rename(columns={'display(in inch)': 'display', 'price(in Rs.)':'price'}, inplace=True)
    return df_new


def train_data(df_new):
    data = df_model[['name', 'processor', 'ram', 'os', 'storage', 
                     'display', 'price', 'no_of_ratings', 'no_of_reviews']]
    return data

def prepro_data(data):
    data['Manufacturer'] = data['name'].apply(lambda x: x.split(' ')[0])
    data['Model'] = data['name'].apply(lambda x: ' '.join(x.split(' ')[1:]))
    data['Ram_GB'] = data['ram'].apply(lambda x: x.split('GB')[0])
    data['Ram_type'] = data['ram'].apply(lambda x: x.split('GB')[1])
    return data

def sparse_lbe(data):
    sparse_feature = ['Manufacturer', 'Model', 'processor', 'Ram_GB', 'Ram_type', 'os', 'storage']

    for feat in sparse_feature:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    
    return data

def dense_mmscaler(data):
    dense_feature = ['display','price', 'no_of_ratings', 'no_of_reviews']
    data[dense_feature].fillna(0, inplace=True)
 
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_feature] = mms.fit_transform(data[dense_feature])
    return data

def fixlen_data(data):
    sparse_feature = ['Manufacturer', 'Model', 'processor', 'Ram_GB', 'Ram_type', 'os', 'storage']
    dense_feature = ['display','price', 'no_of_ratings', 'no_of_reviews']

    fixlen_feature_columns = [SparseFeat(feat, data[feat].max() + 1, embedding_dim=4) for feat in sparse_feature] \
                         + [DenseFeat(feat, 1, ) for feat in dense_feature]
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    return dnn_feature_columns, linear_feature_columns, feature_names

def split_data(dnn_feature_columns, linear_feature_columns, feature_names, df_new):
    train, test = train_test_split(data, test_size=0.2, random_state=2020)
    train_y, test_y = train_test_split(df_new['rating'], test_size=0.2, random_state=2020)

    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}
    return train, test, train_y, test_y, train_model_input, test_model_input

def model_fit_data(linear_feature_columns, dnn_feature_columns, train_model_input, train_y):
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='regression')
    model.compile("adam", loss="mse")
    history = model.fit(train_model_input, train_y,
                    batch_size=256, epochs=10, verbose=2, validation_split=0.2)
    
    plt.plot(history.history['loss'], label = 'Train Loss', color='blue')
    plt.plot(history.history['val_loss'], label = 'Validation Loss', color='red')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('Model Train-Validation Loss')
    plt.show()
    return history, plt.show()

def predict_data(test_model_input):
    predict_y = model.predict(test_model_input, batch_size=256)
    return predict_y


    

