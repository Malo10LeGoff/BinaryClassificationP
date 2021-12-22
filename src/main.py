from numpy.random.mtrand import random
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from evaluation import evaluate_model, conf_matrix
from preprocessing import normalize_dataset, preprocess
from preprocessing import data_cleaning

if __name__ == '__main__':
    
    """
    @Writer : Malo Le Goff
    """
    
    ### Import the dataset
    df = pd.read_csv("../data/data_banknote_authentication.csv", sep=",")

    df, category_columns, numerical_columns = data_cleaning(df)

    x = pd.get_dummies(df.drop(columns=["classification"]))
    y = df["classification"].apply(lambda x: 1 if x == "ckd" or x == 1 else 0)
    ### Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.33, random_state=42
    )

    # Preprocess the split
    x_train_preprocessed = preprocess(data=x_train, numerical_columns=numerical_columns)
    x_test_preprocessed = preprocess(data=x_test, numerical_columns=numerical_columns)
    print(x_train_preprocessed)

    ### Training
    model = RandomForestClassifier(random_state=42, max_depth=10, n_estimators=100)
    model.fit(x_train_preprocessed, y_train)

    ### Predict and Evaluate
    result = evaluate_model(model=model, x_test=x_test_preprocessed, y_true=y_test, metric=accuracy_score)
    confusion_matrix = conf_matrix(model=model, x_test=x_test_preprocessed, y_true=y_test)
    print(f"confusion matrix : ")
    print(confusion_matrix)
    print(f"Accuracy : {result}")
