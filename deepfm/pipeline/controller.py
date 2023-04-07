from pipeline import extract, transform
from settings import DB_SETTINGS
from db.connector import DBConnector


def etl():
    result = extract.df(
        db_connector=DBConnector(**DB_SETTINGS['dir_path'])
    )

    df_new = transform.df_notnull_rename(result)
    data = transform.train_data(df_new)
    data = transform.prepro_data(data)
    data = transform.sparse_lbe(data)
    data = transform.dense_mmscaler(data)
    dnn_feature_columns = transform.fixlen_data(data)[0]
    linear_feature_columns = transform.fixlen_data(data)[1]
    feature_names = transform.fixlen_data(data)[2]
    train = transform.split_data(dnn_feature_columns, linear_feature_columns, feature_names, df_new)[0]
    test = transform.split_data(dnn_feature_columns, linear_feature_columns, feature_names, df_new)[1]
    train_y = transform.split_data(dnn_feature_columns, linear_feature_columns, feature_names, df_new)[2]
    test_y = transform.split_data(dnn_feature_columns, linear_feature_columns, feature_names, df_new)[3]
    train_model_input = transform.split_data(dnn_feature_columns, linear_feature_columns, feature_names, df_new)[4]
    test_model_input = transform.split_data(dnn_feature_columns, linear_feature_columns, feature_names, df_new)[5]
    history = transform.model_fit_data(linear_feature_columns, dnn_feature_columns, train_model_input, train_y)
    predict_y = transform.predict_data(test_model_input)

    df_new = transform.df_isnull_rename(result)
    data = transform.train_data(df_new)
    data = transform.prepro_data(data)
    data = transform.sparse_lbe(data)
    data = transform.dense_mmscaler(data)
    dnn_feature_columns = transform.fixlen_data(data)[0]
    linear_feature_columns = transform.fixlen_data(data)[1]
    feature_names = transform.fixlen_data(data)[2]
    train = transform.split_data(dnn_feature_columns, linear_feature_columns, feature_names, df_new)[0]
    test = transform.split_data(dnn_feature_columns, linear_feature_columns, feature_names, df_new)[1]
    train_y = transform.split_data(dnn_feature_columns, linear_feature_columns, feature_names, df_new)[2]
    test_y = transform.split_data(dnn_feature_columns, linear_feature_columns, feature_names, df_new)[3]
    train_model_input = transform.split_data(dnn_feature_columns, linear_feature_columns, feature_names, df_new)[4]
    test_model_input = transform.split_data(dnn_feature_columns, linear_feature_columns, feature_names, df_new)[5]
    history = transform.model_fit_data(linear_feature_columns, dnn_feature_columns, train_model_input, train_y)
    predict_y = transform.predict_data(test_model_input)
