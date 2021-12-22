from sklearn.metrics import accuracy_score, confusion_matrix


def evaluate_model(model, x_test, y_true, metric=accuracy_score):
    """
    Evaluate a model's classification against a given metric.
    By default evaluates with by accuracy score.
    :param y_pred: The output from the model's predictions.
    :param y_true: Ground truth
    :param metric: The metric to evaluate against. Takes y_pred and y_true and outputs a score
    :return: the result of the evaluation
    
    @Writer : Stanislas de Charentenay
    """
    # get the model's predictions
    y_pred = model.predict(x_test)

    return metric(y_true, y_pred)


def conf_matrix(model, x_test, y_true):
    """
    Calculate the model's confusion matrix
    
    @Writer : Stanislas de Charentenay
    """
    y_pred = model.predict(x_test)
    return confusion_matrix(y_true, y_pred)
