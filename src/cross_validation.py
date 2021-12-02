from typing import List

import pandas as pd

from evaluation import evaluate_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from src.preprocessing import preprocess, fill_na_values
from src.utility_functions import get_num_cat_features


def cross_val_model(ModelClass,
                    dataset,
                    labels,
                    n_components,
                    numerical_columns: List,
                    parameter_name: str,
                    parameter_range: List,
                    n_trials=30,
                    test_size=0.33,
                    metric=accuracy_score,
                    **kwargs):
    """
    Cross validate a model agaisnt a given metric, returning the results

    :param parameter_range:
    :param ModelClass: the model class to test
    :param dataset: The Dataset you want to test the model on
    :param labels: The true labels from the dataset
    :param n_trials: The number of times you want to train the  model on the data per parameter value
    :param kwargs: keyword arguments to pass to the model

    :return:
        parameter_range:
            The range of parameters the model was tested on and
        results :
            A len(parameter_range) x n_trials list containing the results of each trial run for each parameter value
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
            x_train_preprocessed = preprocess(data=x_train, numerical_columns=numerical_columns,
                                              n_components=n_components)
            x_test_preprocessed = preprocess(data=x_test, numerical_columns=numerical_columns,
                                             n_components=n_components)

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
    :return: None
    """
    trial_length = _calculate_trial_length(trial)
    fig, ax = plt.subplots(ncols=1, nrows=trial_length)
    # Quick and dirty bugfix
    if trial_length == 1:
        ax = [ax]
    count = 0
    for model in trial:
        for parameter_setup in trial[model]["parameters"]:
            ax[count].set_title(model)
            ax[count].set_xlabel(parameter_setup["name"])
            ax[count].boxplot(parameter_setup["results"], labels=parameter_setup["range"])
            ax[count].set_ylim([0.8, 1])
            count += 1
    return fig, ax


if __name__ == '__main__':
    from sklearn.ensemble import RandomForestClassifier
    import pickle
    from datetime import datetime

    trial = {
        "RandomForestClassifier": {
            "model_class": RandomForestClassifier,
            "parameters": [{
                "name": "max_depth",
                "range": list(range(1, 5)),
                "results": None
            },
            ],
        }
    }

    # Import the dataset
    df = pd.read_csv("../data/data_banknote_authentication.csv", sep=",")
    n_components = 4

    # Clean the missing values
    category_columns, numerical_columns = get_num_cat_features(df=df.loc[:, df.columns != 'classification'])

    df = fill_na_values(
        df=df, category_columns=category_columns, numerical_columns=numerical_columns
    )
    y = df["classification"]
    x = df.drop(columns=["classification"])
    for model in trial:
        for parameter_setup in trial[model]["parameters"]:
            model_parameter_range, model_results = cross_val_model(ModelClass=trial[model]["model_class"],
                                                                   dataset=x,
                                                                   labels=y,
                                                                   n_components=n_components,
                                                                   numerical_columns=numerical_columns,
                                                                   parameter_name=parameter_setup["name"],
                                                                   parameter_range=parameter_setup["range"],
                                                                   )
            # Register the results in a dict
            parameter_setup["results"] = model_results

    print(trial)

    # Save results and parameter ranges
    date = datetime.now()
    with open(
            f"../results/cross_val_result-{date.year}-{date.month}-{date.day}_{date.hour}-{date.minute}.pkl",
            "wb+") as trial_file:
        pickle.dump(trial, trial_file)

    fig, ax = plot_trial(trial)
    plt.show()
