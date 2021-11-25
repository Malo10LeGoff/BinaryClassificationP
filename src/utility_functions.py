import pandas as pd
from typing import List


def get_num_cat_features(*, df: pd.DataFrame) -> List:

    cateogry_columns = df.select_dtypes(include=["object"]).columns.tolist()
    integer_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    return cateogry_columns, integer_columns
