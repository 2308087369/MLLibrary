# data_processing/feature_extraction.py
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import pandas as pd
import numpy as np

def mutual_info_analysis(data, target, threshold=0.2):
    """
    Perform Mutual Information analysis and filter features based on threshold.

    Args:
        data (pd.DataFrame): The input data.
        target (pd.Series): The target column.
        threshold (float): Threshold for dropping low-importance features. Default is 0.2.

    Returns:
        pd.DataFrame: Filtered data after Mutual Information analysis.
    """
    # Automatically select the appropriate mutual information function
    if target.dtype.kind in 'if':  # If target is numeric (int or float), use regression
        mi_scores = mutual_info_regression(data, target)
    else:  # Otherwise, assume classification
        mi_scores = mutual_info_classif(data, target)
    
    mi_df = pd.DataFrame({"Feature": data.columns, "MI_Score": mi_scores})
    print("Mutual Information Analysis Results:")
    print(mi_df)

    # Filter features based on threshold
    filtered_features = mi_df[mi_df["MI_Score"] >= threshold]["Feature"].tolist()
    print(f"Selected Features based on MI Score (Threshold: {threshold}): {filtered_features}")
    return data[filtered_features]

def correlation_analysis(data, threshold=0.2):
    """Perform correlation matrix analysis and filter features based on threshold.

    Args:
        data (pd.DataFrame): The input data.
        threshold (float): Threshold for dropping low-correlation features. Default is 0.2.

    Returns:
        pd.DataFrame: Filtered data after correlation analysis.
    """
    corr_matrix = data.corr()
    print("Correlation Matrix:")
    print(corr_matrix)
    important_features = corr_matrix.columns[(corr_matrix.abs().mean() >= threshold)]
    filtered_data = data[important_features]
    print(f"Filtered data with features having average correlation >= {threshold}:")
    print(filtered_data.head())
    return filtered_data

def select_features(data, target, k=10):
    """Select top-k features based on univariate statistical tests."""
    from sklearn.feature_selection import SelectKBest, f_classif
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(data, target)
    return X_new, selector.scores_

def perform_pca(data, n_components=2):
    """Perform Principal Component Analysis (PCA) on the dataset."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    return reduced_data, pca.explained_variance_ratio_