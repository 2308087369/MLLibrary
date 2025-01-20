import numpy as np
from evaluation.metrics import calculate_regression_metrics, calculate_classification_metrics, calculate_confusion_matrix
from evaluation.visualization import plot_training_curve, plot_confusion_matrix, plot_predictions

# Simulated data
y_true = np.random.randint(0, 2, size=100)  # For classification
y_pred = np.random.randint(0, 2, size=100)  # For classification

# Classification metrics
print("Classification Metrics:")
print(calculate_classification_metrics(y_true, y_pred))

# Confusion matrix
print("Confusion Matrix:")
plot_confusion_matrix(y_true, y_pred, class_names=["Class 0", "Class 1"])

# Regression example
y_true_reg = np.random.uniform(0, 1, size=100)
y_pred_reg = y_true_reg + np.random.normal(0, 0.1, size=100)

# Regression metrics
print("Regression Metrics:")
print(calculate_regression_metrics(y_true_reg, y_pred_reg))

# True vs Predicted plot
plot_predictions(y_true_reg, y_pred_reg)

# Training curve example
history = {
    "train_loss": np.random.uniform(0.5, 0.1, size=10),
    "val_loss": np.random.uniform(0.6, 0.2, size=10),
}
plot_training_curve(history, metrics=["loss"])
