import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import List
import numpy as np


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


def feature_selection(*, data, n_components):
    """
    Reduce the dimensionality of the data through Principal Component Analysis/
    :param data: data to reduce
    :param n_components: number of components to keep
    :return: the reduced data
    """
    pca = PCA(n_components=n_components, random_state=42)

    data_reduced = pca.fit_transform(data)
    return data_reduced


def preprocess(
    *, data: pd.DataFrame, numerical_columns: List, category_columns: List, n_components
):
    """
    Preprocessing pipeline for our project.
    :param data: the data to process.
    :param numerical_columns: List of the data's column names which contain numerical values.
    :param n_components: number of components to keep after feature reduction
    :return: processed data
    """
    data = data[numerical_columns].values

    # Reduce dimensionality (PCA)
    data_reduced = feature_selection(data=data, n_components=n_components)

    # Normalize data
    data_reduced_scaled = normalize_dataset(data=data_reduced)

    return data_reduced_scaled
