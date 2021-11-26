from typing import List
from evaluation import evaluate_model
from sklearn.model_selection import train_test_split

from src.preprocessing import preprocess


def cross_val_model(ModelClass,
                    dataset,
                    labels,
                    numerical_columns: List,
                    parameter_name: str,
                    parameter_range: List,
                    n_trials=30,
                    test_size=0.33,
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
            result = evaluate_model(model=model, x_test=x_test_preprocessed, y_true=y_test)
            parameter_results.append(result)

        # Add the run results to the results
        results.append(parameter_results)
    return parameter_range, results
