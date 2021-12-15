from numpy.random.mtrand import random
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from evaluation import evaluate_model, conf_matrix
from preprocessing import normalize_dataset, preprocess
from utility_functions import get_num_cat_features
from preprocessing import fill_na_values, normalize_dataset
import sys

if __name__ == "__main__":
    ### Import the dataset

    if str(sys.argv[1]) == "kidney_disease":
        df = pd.read_csv("../data/kidney_disease.csv", sep=",")
    if str(sys.argv[1]) == "data_banknote":
        df = pd.read_csv("../data/data_banknote_authentication.csv", sep=",")
    else:
        raise (
            "Check the name of the dataset you want to run in your command line arguments !"
        )

    n_components = 4

    ### Clean the missing values
    category_columns, numerical_columns = get_num_cat_features(
        df=df.loc[:, df.columns != "classification"]
    )

    df = fill_na_values(
        df=df, category_columns=category_columns, numerical_columns=numerical_columns
    )

    y = df["classification"].apply(lambda x: 1 if x == "ckd" or x == 1 else 0)

    x = pd.get_dummies(df.drop(columns=["classification"]))

    ### Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.33, random_state=42
    )

    # Preprocess the split
    x_train_preprocessed = preprocess(
        data=x_train,
        numerical_columns=numerical_columns,
        category_columns=category_columns,
        n_components=n_components,
    )
    x_test_preprocessed = preprocess(
        data=x_test,
        numerical_columns=numerical_columns,
        category_columns=category_columns,
        n_components=n_components,
    )

    ### Training
    model = RandomForestClassifier(random_state=42)
    model.fit(x_train_preprocessed, y_train)

    ### Predict and Evaluate
    accu = evaluate_model(
        model=model, x_test=x_test_preprocessed, y_true=y_test, metric=accuracy_score
    )
    f1score = evaluate_model(
        model=model, x_test=x_test_preprocessed, y_true=y_test, metric=f1_score
    )
    confusion_matrix = conf_matrix(
        model=model, x_test=x_test_preprocessed, y_true=y_test
    )
    print("\n")
    print(f"confusion matrix : ")
    print(confusion_matrix)
    print(f"Accuracy : {accu}")
    print(f"F1-score : {f1score}")
