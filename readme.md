# **MLLibrary**

A custom machine learning library providing tools for data preprocessing, feature extraction, model selection, training, evaluation, and more. Designed to streamline machine learning workflows and offer flexibility for various tasks.

---

## **Features**
- **Data Processing**: 
  - Load and preprocess datasets with missing value handling, outlier detection, normalization, and feature extraction.
- **Model Selection**:
  - Supports common machine learning models (e.g., Random Forest, XGBoost, and PyTorch-based RNN models like LSTM, GRU, and Transformers).
  - Hyperparameter tuning with grid search and Bayesian optimization.
  - Ensemble learning techniques for improved performance.
- **Evaluation and Visualization**:
  - Built-in metrics for performance evaluation.
  - Easy-to-use visualization tools for training curves, predictions, and more.

---

## **Installation**

1. Clone this repository:
   ```bash
   git clone https://github.com/your_username/MLLibrary.git
   cd MLLibrary
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

## **Usage**
- **Example1: Data Preprocessing**
    ```python
    from MLLibrary.data_processing.preprocessing import preprocess_data
    import pandas as pd

    # Load dataset
    data = pd.read_csv("your_dataset.csv")

    # Preprocess the data
    processed_data = preprocess_data(
        data,
        target_column="target",
        handle_missing="mean",
        outliers=True,
        feature="mi",
        normalize=True,
        split_ratio=(0.7, 0.15, 0.15)
    )

    # Access processed datasets
    X_train, y_train = processed_data["train_data"], processed_data["train_labels"]
    X_val, y_val = processed_data["val_data"], processed_data["val_labels"]
    X_test, y_test = processed_data["test_data"], processed_data["test_labels"]
- **Example 2: Model Training and Hyperparameter Tuning**
    ```python
    from MLLibrary.model_selection.hyperparameter_tuning import grid_search
    from sklearn.ensemble import RandomForestRegressor

    # Define a model
    rf_model = RandomForestRegressor(random_state=42)

    # Define hyperparameter grid
    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [5, 10, 15],
        "min_samples_split": [2, 5, 10],
    }

    # Perform grid search
    best_params = grid_search(rf_model, param_grid, X_train, y_train, cv=3)
    print("Best Hyperparameters:", best_params)
- **Example 3: Ensemble Learning**
    ```python
    from MLLibrary.model_selection.ensembling import StackingEnsemble

    # Define base models
    base_models = {
        "RandomForest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42)
    }

    # Define meta-model
    meta_model = RandomForestRegressor(random_state=42)

    # Create and train stacking ensemble
    stacking_ensemble = StackingEnsemble(base_models, meta_model)
    stacking_ensemble.fit(X_train, y_train)

    # Make predictions
    predictions = stacking_ensemble.predict(X_test)

## **Project Structure**
    ```plainttext
    MLLibrary/
    ├── __init__.py
    ├── data_processing/
    │   ├── __init__.py
    │   ├── data_loader.py       # Data loading functions
    │   ├── preprocessing.py     # Data cleaning and preprocessing
    │   ├── feature_extraction.py # Feature selection and extraction
    ├── model_selection/
    │   ├── __init__.py
    │   ├── algorithms.py        # Supported ML algorithms (e.g., Random Forest, RNN)
    │   ├── hyperparameter_tuning.py # Hyperparameter search
    │   ├── ensembling.py        # Ensemble methods
    ├── evaluation/
    │   ├── __init__.py
    │   ├── metrics.py           # Performance metrics
    │   ├── visualization.py     # Visualization tools
    ├── utils/
    │   ├── __init__.py
    │   ├── file_operations.py   # Save and load models/results
    │   ├── logger.py            # Logging utilities
    │   ├── general_utils.py     # General utilities (e.g., timing, random seed)
    ├── examples/
    │   ├── example_script.py    # Usage examples
    └── requirements.txt         # Project dependencies
## **Dependencies**
For a complete list of dependencies, see requirements.txt.

## **License**
This project is licensed under the MIT License - see the LICENSE.md file for details.
