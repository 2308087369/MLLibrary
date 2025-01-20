import matplotlib.pyplot as plt
import seaborn as sns
from .metrics import calculate_confusion_matrix

def plot_training_curve(history, metrics=["loss"]):
    """
    Plot training and validation curves.

    Args:
        history (dict): Training history with keys for metrics (e.g., 'loss', 'accuracy').
        metrics (list): Metrics to plot (default is ['loss']).
    """
    for metric in metrics:
        plt.figure(figsize=(8, 6))
        plt.plot(history[f"train_{metric}"], label=f"Train {metric}")
        if f"val_{metric}" in history:
            plt.plot(history[f"val_{metric}"], label=f"Validation {metric}")
        plt.title(f"Training and Validation {metric.capitalize()}")
        plt.xlabel("Epoch")
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid()
        plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Plot a confusion matrix.
    
    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        class_names (list): Class names for the matrix (optional).
    """
    cm = calculate_confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def plot_predictions(y_true, y_pred):
    """
    Plot true vs. predicted values.

    Args:
        y_true (array-like): True values.
        y_pred (array-like): Predicted values.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors="k")
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], "r--")
    plt.title("True vs Predicted Values")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.grid()
    plt.show()
