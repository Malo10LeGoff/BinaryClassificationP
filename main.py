from numpy.random.mtrand import random
import pandas as pd
import numpy as np
from scipy.sparse.construct import rand
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.decomposition import PCA

df = pd.read_csv("data/kidney_disease.csv", sep=",")

### Clean the data and preprocess it (normalize, ...)
cateogry_columns = df.select_dtypes(include=["object"]).columns.tolist()
integer_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

for column in df:
    if df[column].isnull().any():
        if column in cateogry_columns:
            df[column] = df[column].fillna(df[column].mode()[0])
        else:
            df[column] = df[column].fillna(df[column].mean(skipna=True))

df.to_csv("results/modified_df.csv")

y = df["classification"]
X = df.drop(columns=["classification"])


### Split the dataset

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

### Normalize integer functions
sc = StandardScaler()

X_train_scaled = sc.fit_transform(X_train[integer_columns].values)
X_test_scaled = sc.fit_transform(X_test[integer_columns].values)

### Feature selection

pca = PCA(n_components=10)

X_train_scaled_reduced = pca.fit_transform(X_train_scaled)
X_test_scaled_reduced = pca.fit_transform(X_test_scaled)


### Training

model = RandomForestClassifier()
model.fit(X_train_scaled_reduced, y_train)

### Predict

y_pred = model.predict(X_test_scaled_reduced)

conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)


print(f"confusion matrix : ")
print(conf_matrix)
print(f"Accuracy : {accuracy_score(y_test, y_pred)}")
