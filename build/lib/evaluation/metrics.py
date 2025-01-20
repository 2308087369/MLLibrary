from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def calculate_regression_metrics(y_true, y_pred):
    """
    Calculate regression metrics.
    
    Args:
        y_true (array-like): True values.
        y_pred (array-like): Predicted values.
    
    Returns:
        dict: A dictionary with MAE, MSE, and R².
    """
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }

def calculate_classification_metrics(y_true, y_pred):
    """
    Calculate classification metrics.
    
    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
    
    Returns:
        dict: A dictionary with accuracy, precision, recall, and F1-score.
    """
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted"),
        "Recall": recall_score(y_true, y_pred, average="weighted"),
        "F1-Score": f1_score(y_true, y_pred, average="weighted"),
    }

def calculate_confusion_matrix(y_true, y_pred):
    """
    Compute the confusion matrix.
    
    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
    
    Returns:
        array: The confusion matrix.
    """
    return confusion_matrix(y_true, y_pred)
