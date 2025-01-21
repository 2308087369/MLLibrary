import numpy as np
from evaluation.metrics import calculate_regression_metrics, calculate_classification_metrics
from evaluation.visualization import plot_training_curve, plot_confusion_matrix, plot_predictions

def test_classification_metrics():
    # Simulated data
    y_true = np.random.randint(0, 2, size=100)
    y_pred = np.random.randint(0, 2, size=100)

    metrics = calculate_classification_metrics(y_true, y_pred)
    assert "accuracy" in metrics  # Example assertion
    assert 0 <= metrics["accuracy"] <= 1

def test_confusion_matrix():
    y_true = np.random.randint(0, 2, size=100)
    y_pred = np.random.randint(0, 2, size=100)

    # This is for manual inspection; plot functions typically don't return values
    plot_confusion_matrix(y_true, y_pred, class_names=["Class 0", "Class 1"])

def test_regression_metrics():
    y_true_reg = np.random.uniform(0, 1, size=100)
    y_pred_reg = y_true_reg + np.random.normal(0, 0.1, size=100)

    metrics = calculate_regression_metrics(y_true_reg, y_pred_reg)
    assert "mse" in metrics
    assert metrics["mse"] >= 0

def test_training_curve():
    history = {
        "train_loss": np.random.uniform(0.5, 0.1, size=10),
        "val_loss": np.random.uniform(0.6, 0.2, size=10),
    }
    plot_training_curve(history, metrics=["loss"])
