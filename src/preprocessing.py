import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import List


def fill_na_values(
        *, df: pd.DataFrame, category_columns: List, numerical_columns: List
) -> pd.DataFrame:
    for column in df:
        if df[column].isnull().any():
            if column in category_columns:
                df[column] = df[column].fillna(df[column].mode()[0])
            else:
                df[column] = df[column].fillna(df[column].mean(skipna=True))
    return df


def normalize_dataset(*, data):
    """
    Center and normalize the dataset.
    :param data: the data to center/normalize
    :param numerical_columns: List of columns in the dataset that hold numerical values.
    :return: the scaled data.
    """
    sc = StandardScaler()

    data_scaled = sc.fit_transform(data)

    return data_scaled


def feature_selection(*, data):
    """
    Reduce the dimensionality of the data through Principal Component Analysis/
    :param data: data to reduce
    :return: the reduced data
    """
    pca = PCA(n_components=10, random_state=42)

    data_reduced = pca.fit_transform(data)
    return data_reduced


def preprocess(*, data: pd.DataFrame, numerical_columns: List):
    """
    Preprocessing pipeline for our project.
    :param data: the data to process.
    :param numerical_columns: List of the data's column names which contain numerical values.
    :return: processed data

    #TODO :(@minh tri)  Technically there could be inconsistencies in the preprocessing, because the feature selection and normalization are fitted to 2 different sets of data.
    """
    data = data[numerical_columns].values
    # Reduce dimensionality (PCA)
    data_reduced = feature_selection(data=data)

    # Normalize data
    data_reduced_scaled = normalize_dataset(data=data_reduced)

    return data_reduced_scaled
