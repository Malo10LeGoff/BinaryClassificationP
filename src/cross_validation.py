from typing import List

import pandas as pd
import math

from evaluation import evaluate_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from src.preprocessing import preprocess, fill_na_values
from src.utility_functions import get_num_cat_features
from preprocessing import data_cleaning




def cross_val_model(ModelClass,
                    dataset,
                    labels,
                    numerical_columns: List,
                    parameter_name: str,
                    parameter_range: List,
                    n_trials=10,
                    test_size=0.33,
                    metric=accuracy_score,
                    **kwargs):
    """
    Cross validate a model agaisnt a given metric, returning the results

    :param ModelClass: the model class to test
    :param dataset: The Dataset you want to test the model on
    :param labels: The true labels from the dataset
    :param n_components: The number of dimensions to reduce the dataset to
    :param numerical_columns: Name of the dataset's columns that contain numerical data
    :param parameter_name: The name of the parameter to test
    :param parameter_range: The parameters to evaluate the model on
    :param n_trials: The number of times you want to train the  model on the data per parameter value
    :param test_size: the proportion of the dataset that should be used for the test set
    :param metric: The metric to use to evaluate the model
    :param kwargs: keyword arguments to pass to the model

    :return:
        parameter_range:
            The range of parameters the model was tested on and
        results :
            A len(parameter_range) x n_trials list containing the results of each trial run for each parameter value
            
     @Writer : Minh Tri Truong
    """
    results = []
    for parameter in parameter_range:
        kwargs[parameter_name] = parameter
        parameter_results = []
        for i in range(n_trials):
            print(f"""starting trial {i}/{n_trials} :
    {parameter_name} : value = {parameter}""")
            # Create a new model to test
            model = ModelClass(**kwargs)

            # Split the dataset at random
            x_train, x_test, y_train, y_test = train_test_split(
                dataset, labels, test_size=test_size
            )

            # preprocess the data
            x_train_preprocessed = preprocess(data=x_train, numerical_columns=numerical_columns)
            x_test_preprocessed = preprocess(data=x_test, numerical_columns=numerical_columns)

            # Fit the model to the training data
            model.fit(x_train_preprocessed, y_train)

            # evaluate the model
            result = evaluate_model(model=model, x_test=x_test_preprocessed, y_true=y_test, metric=metric)
            parameter_results.append(result)

        # Add the run results to the results
        results.append(parameter_results)
    return parameter_range, results


def _calculate_trial_length(trial):
    length = 0
    for model in trial:
        # Add the number of parameter ranges to test
        length += len(trial[model]["parameters"])
    return length


def plot_trial(trial):
    """
    Plot the trial
    :param trial: (dict) dict coming from cross_val_model function
            { str : {
                        "parameters" : [{
                            "name": str,
                            "range":str,
                            "results" : list},
                            ...],
                        "metric" : function
    :return: None
    """
    trial_length = _calculate_trial_length(trial)
    fig, ax = plt.subplots(ncols=2, nrows=math.ceil(trial_length/2))
    # Quick and dirty bugfix
    if trial_length == 1:
        ax = [ax]
    count = 0
    lin = 0
    col = 0
    for model in trial:
        for parameter_setup in trial[model]["parameters"]:
            ax[lin][col].set_title(model,fontsize=5)
            ax[lin][col].set_xlabel(parameter_setup["name"],fontsize=5)
            ax[lin][col].set_ylabel(trial[model]["metric"].__name__,fontsize=5)
            ax[lin][col].boxplot(parameter_setup["results"], labels=parameter_setup["range"])
            ax[lin][col].set_ylim([0.8, 1])
            ax[lin][col].tick_params(axis='both', which='major', labelsize=5)
            count += 1
            lin+=1
            if count==math.ceil(trial_length/2):
                lin = 0
                col += 1
    return fig, ax


if __name__ == '__main__':
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    import pickle
    from datetime import datetime
    import numpy as np

    trial = {
        "RandomForestClassifier": {
            "model_class": RandomForestClassifier,
            "parameters": [{
                "name": "max_depth",
                "range": list(range(1, 15)),
                "results": None
            },
            {
                "name": "n_estimators",
                "range": [10,50,100,200,400],
                "results": None
            },
            ],
            "metric": accuracy_score,
        },
        "LogisticRegression": {
            "model_class": LogisticRegression,
            "parameters": [{
                "name": "C",
                "range": list(np.logspace(-3, 3, 7)),
                "results": None
            },
            ],
            "metric": accuracy_score,
        },
        "SVC": {
            "model_class": SVC,
            "parameters": [{
                "name": "C",
                "range": [0.1, 1, 10, 100, 1000],
                "results": None
            },
            {
                "name": "kernel",
                "range": ['rbf','linear','poly','sigmoid'],
                "results": None
            },
            {
                "name": "gamma",
                "range": [1, 0.1, 0.01, 0.001, 0.0001],
                "results": None
            },
            ],
            "metric": accuracy_score,
        },
        "KNN": {
            "model_class": KNeighborsClassifier,
            "parameters": [{
                "name": "n_neighbors",
                "range": [3, 6, 10, 15, 20, 30],
                "results": None
            },
            {
                "name": "metric",
                "range": ['euclidean', 'minkowski'],
                "results": None
            },
            {
                "name": "weights",
                "range": ['uniform','distance'],
                "results": None
            },
            ],
            "metric": accuracy_score,
        }
    }

    # Import the dataset
    df = pd.read_csv("../data/kidney_disease.csv", sep=",")

    df, category_columns, numerical_columns = data_cleaning(df)

    x = pd.get_dummies(df.drop(columns=["classification"]))
    y = df["classification"].apply(lambda x: 1 if x == "ckd" or x == 1 else 0)
    ### Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.33, random_state=42
    )
    for model in trial:
        for parameter_setup in trial[model]["parameters"]:
            model_parameter_range, model_results = cross_val_model(ModelClass=trial[model]["model_class"],
                                                                   dataset=x,
                                                                   labels=y,
                                                                   numerical_columns=numerical_columns,
                                                                   parameter_name=parameter_setup["name"],
                                                                   parameter_range=parameter_setup["range"],
                                                                   metric=trial[model]["metric"]
                                                                   )
            # Register the results in a dict
            parameter_setup["results"] = model_results



    # Save results and parameter ranges
    date = datetime.now()
    with open(
            f"../results/cross_val_result-{date.year}-{date.month}-{date.day}_{date.hour}-{date.minute}.pkl",
            "wb+") as trial_file:
        pickle.dump(trial, trial_file)

    # Plots the trial
    fig, ax = plot_trial(trial)
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.8)
    plt.show()

    with np.printoptions(precision=2, floatmode="fixed"):
        for model, model_param in trial.items():
            print(f"{model:<15}")
            for parameter_dict in model_param['parameters']:
                print(f"{parameter_dict['name']:>15}")
                for i in range(len(parameter_dict['range'])):
                    print(f"{parameter_dict['range'][i]:>20}: median accuracy = {np.median(parameter_dict['results'][i])}")

