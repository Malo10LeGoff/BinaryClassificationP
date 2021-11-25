from sklearn.metrics import confusion_matrix, accuracy_score


def evaluate_model(*, y_pred, y_test):

    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

    print(f"confusion matrix : ")
    print(conf_matrix)
    print(f"Accuracy : {accuracy_score(y_test, y_pred)}")
