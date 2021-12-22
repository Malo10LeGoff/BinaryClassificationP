import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import List
import numpy as np

from utility_functions import drop_id, get_num_cat_features, switch_from_cat_to_num


def fill_na_values(
        *, df: pd.DataFrame, category_columns: List, numerical_columns: List
) -> pd.DataFrame:
    """
    @Writer : Malo Le Goff
    """

    for column in df:
        if df[column].isnull().any():
            if column in category_columns:
                df[column] = df[column].fillna(df[column].mode()[0])
            else:
                df[column] = df[column].fillna(df[column].mean(skipna=True))
    return df

def data_cleaning(df):
    """
    @Writer : Loïc Turounet
    """
    df = drop_id(df)
    category_columns, numerical_columns = get_num_cat_features(df=df.loc[:, df.columns != 'classification'])
    category_columns, numerical_columns, switching_columns = switch_from_cat_to_num(df, numerical_columns, category_columns)
    for col in switching_columns :
        df[col] = pd.to_numeric(df[col],errors='coerce')
    df = fill_na_values(
        df=df, category_columns=category_columns, numerical_columns=numerical_columns
    )
    return df, category_columns, numerical_columns

def normalize_dataset(*, data):
    """
    Center and normalize the dataset.
    :param data: the data to center/normalize
    :param numerical_columns: List of columns in the dataset that hold numerical values.
    :return: the scaled data.
    
    @Writer : Malo Le Goff
    """
    sc = StandardScaler()

    data_scaled = sc.fit_transform(data)

    return data_scaled



def preprocess(*, data: pd.DataFrame, numerical_columns: List):
    """
    Preprocessing pipeline for our project.
    :param data: the data to process.
    :param numerical_columns: List of the data's column names which contain numerical values.
    :param n_components: number of components to keep after feature reduction
    :return: processed data

    @Writer : Loïc Turounet
    """
    data_num = data[numerical_columns].values
    data_cat = data.loc[:,~data.columns.isin(numerical_columns)].values

    # Normalize data
    data_scaled = normalize_dataset(data=data_num)

    data_complete = np.hstack((data_scaled, data_cat))

    return data_complete
