import pandas as pd
from pipeline import extract, transform, load, model
from settings import DB_SETTINGS
from db.connector import DBConnector


def etl():
    print('start_etl')

    result = extract.df(
        db_connector=DBConnector(**DB_SETTINGS['dir_path'])
    )

    result = transform.dateTime(result.columns[0])
    result = transform.mmScaler(result)
    train = transform.train_test_Split(result)[0]
    test = transform.train_test_split(result)[1]
    train_dummy = transform.train_test_Dummy(train)
    test_dummy = transform.train_test_Dummy(test)
    train_x = transform.total_matrix(train_dummy)[0]
    train_y = transform.total_matrix(train_dummy)[1]
    test_x = transform.total_matrix(test_dummy)[0]
    test_y = transform.total_matrix(test_dummy)[1]
    train_x = transform.final_Reshape(train_x)
    train_y = transform.final_Reshape(train_y)
    history = model.RNN_Model(train_x, train_y)
    pred_y = model.plt_Predict(test_x, test_y)


    print('end_etl')
        
    