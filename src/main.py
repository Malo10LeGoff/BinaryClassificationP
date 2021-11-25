from numpy.random.mtrand import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from evaluation import evaluate_model
from preprocessing import normalize_dataset
from utility_functions import get_num_cat_features
from preprocessing import fill_na_values, normalize_dataset, feature_selection


### Import the dataset
df = pd.read_csv("../data/kidney_disease.csv", sep=",")


### Clean the missing values
category_columns, numerical_columns = get_num_cat_features(df=df)

print(type(category_columns))

df = fill_na_values(
    df=df, category_columns=category_columns, numerical_columns=numerical_columns
)

y = df["classification"]
X = df.drop(columns=["classification"])


### Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)


### Normalize the dataset
X_train_scaled, X_test_scaled = normalize_dataset(
    X_train=X_train, X_test=X_test, numerical_columns=numerical_columns
)


### Feature selection (PCA). PCA must be performed before normalization
X_train_scaled_reduced, X_test_scaled_reduced = feature_selection(
    X_train=X_train_scaled, X_test=X_test_scaled
)


### Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled_reduced, y_train)


### Predict and Evaluate
y_pred = model.predict(X_test_scaled_reduced)
evaluate_model(y_pred=y_pred, y_test=y_test)
