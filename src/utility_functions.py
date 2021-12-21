import pandas as pd
from typing import List
import numpy as np
import math


def get_num_cat_features(*, df: pd.DataFrame) -> List:

    category_columns = df.select_dtypes(include=["object"]).columns.tolist()
    integer_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    return category_columns, integer_columns

def is_float(value):
    """ Returns True is string is a float. """
    try:
        val = float(value)
        return not math.isnan(val)
    except ValueError:
        return False

def is_integer(value):
    """ Returns True is string is an integer. """
    try:
        val = int(value)
        return not math.isnan(val)
    except ValueError:
        return False

def switch_from_cat_to_num(df, numerical_columns, category_columns):
    """ Switches feature from category columns to numerical columns if first the two strings of the column contain an integer or a float. """
    switching_columns = []
    for col in category_columns:
        if is_float(df[col][0]) or is_integer(df[col][0]) or is_float(df[col][1]) or is_integer(df[col][1]):
            switching_columns.append(col)
    numerical_columns.extend(switching_columns)
    for item in switching_columns:
        category_columns.remove(item)
    return category_columns, numerical_columns, switching_columns

def drop_id(df):
    """ Drops 'id' feature useless for the prediction """
    if 'id' in df.columns:
        df = df.drop(columns=["id"])
    return df


# if __name__ == '__main__':
#     df = pd.read_csv("../data/kidney_disease.csv", sep=",")
#     df = drop_id(df)
#     category_columns, numerical_columns = get_num_cat_features(df=df.loc[:, df.columns != 'classification'])
#     category_columns, numerical_columns, switching_columns = switch_from_cat_to_num(df, numerical_columns, category_columns)
#     for col in switching_columns :
#         df[col] = pd.to_numeric(df[col],errors='coerce')
#     print(df.head())