# data_processing/data_loader.py
import pandas as pd

def load_csv(filepath, label_column='labels', n_preview=None):
    """Load a CSV file into a pandas DataFrame, identifying features and labels."""
    data = pd.read_csv(filepath)
    if label_column in data.columns:
        features = data.drop(columns=[label_column])
        labels = data[label_column]
    else:
        features, labels = data, None
    if n_preview:
        print(data.head(n_preview))
    return features, labels

def load_excel(filepath, label_column='labels', n_preview=None):
    """Load an Excel file into a pandas DataFrame, identifying features and labels."""
    data = pd.read_excel(filepath)
    if label_column in data.columns:
        features = data.drop(columns=[label_column])
        labels = data[label_column]
    else:
        features, labels = data, None
    if n_preview:
        print(data.head(n_preview))
    return features, labels

def load_json(filepath, label_column='labels', n_preview=None):
    """Load a JSON file into a pandas DataFrame, identifying features and labels."""
    data = pd.read_json(filepath)
    if label_column in data.columns:
        features = data.drop(columns=[label_column])
        labels = data[label_column]
    else:
        features, labels = data, None
    if n_preview:
        print(data.head(n_preview))
    return features, labels