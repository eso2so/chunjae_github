import pandas as pd

def df(db_connector):
    result = pd.read_csv(db_connector)
    return result