import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class EnsembleModel:
    def __init__(self, method="weighted_average", weights=None, meta_model=None):
        """
        Initialize the ensemble model.

        Args:
            method (str): The ensemble method, one of ["weighted_average", "stacking"].
            weights (list or None): Weights for weighted averaging. If None, equal weights are used.
            meta_model (object or None): Meta model for stacking. Defaults to LinearRegression.
        """
        self.method = method
        self.weights = weights
        self.meta_model = meta_model or LinearRegression()
        self.fitted_meta_model = None

    def fit(self, base_model_predictions, y_train):
        """
        Fit the ensemble model (for stacking).

        Args:
            base_model_predictions (list of np.ndarray): Predictions from base models for training data.
            y_train (np.ndarray): Ground truth labels for training data.
        """
        if self.method == "stacking":
            X_stack = np.column_stack(base_model_predictions)
            self.fitted_meta_model = self.meta_model.fit(X_stack, y_train)

    def predict(self, base_model_predictions):
        """
        Predict using the ensemble model.

        Args:
            base_model_predictions (list of np.ndarray): Predictions from base models for test data.

        Returns:
            np.ndarray: Ensemble predictions.
        """
        if self.method == "weighted_average":
            if self.weights is None:
                self.weights = [1 / len(base_model_predictions)] * len(base_model_predictions)
            return np.average(np.column_stack(base_model_predictions), axis=1, weights=self.weights)

        elif self.method == "stacking":
            if self.fitted_meta_model is None:
                raise ValueError("Meta model is not fitted. Call `fit` first.")
            X_stack = np.column_stack(base_model_predictions)
            return self.fitted_meta_model.predict(X_stack)

        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")


def train_and_evaluate_ensemble(base_models, train_data, val_data, test_data, ensemble_method="weighted_average", weights=None, meta_model=None):
    """
    Train and evaluate an ensemble model.

    Args:
        base_models (list): List of trained base models (from algorithms.py).
        train_data (tuple): Tuple of (X_train, y_train).
        val_data (tuple): Tuple of (X_val, y_val).
        test_data (tuple): Tuple of (X_test, y_test).
        ensemble_method (str): Ensemble method, one of ["weighted_average", "stacking"].
        weights (list or None): Weights for weighted averaging.
        meta_model (object or None): Meta model for stacking.

    Returns:
        dict: Results including ensemble predictions and test MSE.
    """
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    # 获取各基础模型的预测结果
    train_predictions = [model.predict(X_train) for model in base_models]
    val_predictions = [model.predict(X_val) for model in base_models]
    test_predictions = [model.predict(X_test) for model in base_models]

    # 初始化集成模型
    ensemble = EnsembleModel(method=ensemble_method, weights=weights, meta_model=meta_model)

    # 训练集成模型（仅对堆叠）
    if ensemble_method == "stacking":
        ensemble.fit(train_predictions, y_train)

    # 生成集成预测
    val_ensemble_pred = ensemble.predict(val_predictions)
    test_ensemble_pred = ensemble.predict(test_predictions)

    # 计算性能指标
    val_mse = mean_squared_error(y_val, val_ensemble_pred)
    test_mse = mean_squared_error(y_test, test_ensemble_pred)

    return {
        "ensemble_model": ensemble,
        "val_mse": val_mse,
        "test_mse": test_mse,
        "test_predictions": test_ensemble_pred,
    }
