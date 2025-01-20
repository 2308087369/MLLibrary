# data_processing/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from .feature_extraction import correlation_analysis, mutual_info_analysis

def preprocess_data(data, target_column=None, normalize=False, handle_missing="none", outliers=False, feature="none", feature_threshold=0.2, split_ratio=(0.7, 0.15, 0.15)):
    """Preprocess data by handling missing values, outliers, feature selection, and normalization.

    Args:
        data (pd.DataFrame): The input data.
        target_column (str): The target column name.
        normalize (bool): Whether to normalize the data.
        handle_missing (str): How to handle missing values ('none', 'mean', 'random', 'linear').
        outliers (bool): Whether to handle outliers using the 3-sigma rule.
        feature (str): Feature selection method ('none', 'ca' for correlation analysis, 'mi' for mutual information analysis).
        feature_threshold (float): Threshold for feature selection.
        split_ratio (tuple): Ratios for splitting data. Default is (0.7, 0.15, 0.15).

    Returns:
        dict: A dictionary containing processed data splits and scalers.
    """
    def handle_missing_values(df, strategy):
        """Handle missing values based on the selected strategy."""
        if strategy == "mean":
            return df.fillna(df.mean())
        elif strategy == "random":
            for col in df.columns:
                if df[col].isnull().any():
                    lower = df[col].fillna(method="ffill")
                    upper = df[col].fillna(method="bfill")
                    df[col] = df[col].fillna(lambda x: np.random.uniform(lower, upper))
        elif strategy == "linear":
            return df.interpolate(method="linear")
        return df

    def handle_outliers(df):
        """Detect and handle outliers using the 3-sigma rule."""
        for col in df.select_dtypes(include=[np.number]).columns:
            mean = df[col].mean()
            std = df[col].std()
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            if outlier_mask.any():
                print(f"Outliers detected in column '{col}': {outlier_mask.sum()} values")
                df.loc[outlier_mask, col] = np.nan
        return df.interpolate(method="linear")

    splits = {}
    scaler = None

    # Print table header
    print("Data Columns:", data.columns.tolist())

    # Extract time features if time-like column exists
    for col in data.columns:
        if pd.api.types.is_datetime64_any_dtype(data[col]) or "date" in col.lower() or "time" in col.lower():
            if not pd.api.types.is_datetime64_any_dtype(data[col]):
                data[col] = pd.to_datetime(data[col], errors='coerce')
            data["Year"] = data[col].dt.year
            data["Month"] = data[col].dt.month
            data["Day"] = data[col].dt.day
            data["Hour"] = data[col].dt.hour
            data["Minute"] = data[col].dt.minute
            data["Second"] = data[col].dt.second
            print(f"Extracted time features from column '{col}'")
    if target_column:
    # Drop datetime columns
        datetime_columns = data.select_dtypes(include=['datetime64[ns]']).columns
        if not datetime_columns.empty:
            print(f"Dropping datetime columns: {datetime_columns.tolist()}")
            data = data.drop(columns=datetime_columns)

        X = data.drop(columns=[target_column])
        y = data[target_column]
    else:
        X, y = data, None
        
    # Encode non-numeric variables
    for col in data.select_dtypes(include=['object', 'category']).columns:
        print(f"Encoding non-numeric column: {col}")
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col].astype(str))

    # Handle missing values
    if data.isnull().any().any():
        print("Missing values detected:")
        print(data.isnull().sum()[data.isnull().sum() > 0])
        data = handle_missing_values(data, handle_missing)

    # Handle outliers
    if outliers:
        data = handle_outliers(data)

    # Perform feature selection if enabled
    if feature in ["ca", "mi"]:
        if not target_column:
            raise ValueError("Target column is required for feature selection.")
        X = data.drop(columns=[target_column])
        y = data[target_column]

        if feature == "ca":
            X = correlation_analysis(X.join(y), threshold=feature_threshold)
        elif feature == "mi":
            X = mutual_info_analysis(X, y, threshold=feature_threshold)

        data = pd.concat([X, y], axis=1)

    # Split data
    train_size, val_size, test_size = split_ratio
    if val_size > 0:
        train_data, temp_data, train_labels, temp_labels = train_test_split(
            X, y, test_size=(1 - train_size), shuffle=False
        )
        val_data, test_data, val_labels, test_labels = train_test_split(
            temp_data, temp_labels, test_size=(test_size / (test_size + val_size)), shuffle=False
        )
    else:
        train_data, test_data, train_labels, test_labels = train_test_split(
            X, y, test_size=(1 - train_size), shuffle=False
        )
        val_data, val_labels = None, None

    # Normalize data
    if normalize:
        scaler = StandardScaler()
        train_data = pd.DataFrame(scaler.fit_transform(train_data), columns=X.columns)
        if val_data is not None:
            val_data = pd.DataFrame(scaler.transform(val_data), columns=X.columns)
        test_data = pd.DataFrame(scaler.transform(test_data), columns=X.columns)

    splits = {
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
        "train_labels": train_labels,
        "val_labels": val_labels,
        "test_labels": test_labels,
        "scaler": scaler,
    }

    return splits
