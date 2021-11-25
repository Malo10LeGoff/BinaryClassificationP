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


def normalize_dataset(*, X_train, X_test, numerical_columns):
    sc = StandardScaler()

    X_train_scaled = sc.fit_transform(X_train[numerical_columns].values)
    X_test_scaled = sc.fit_transform(X_test[numerical_columns].values)

    return X_train_scaled, X_test_scaled


def feature_selection(*, X_train, X_test):

    pca = PCA(n_components=10)

    X_train_scaled_reduced = pca.fit_transform(X_train)
    X_test_scaled_reduced = pca.fit_transform(X_test)
    return X_train_scaled_reduced, X_test_scaled_reduced
