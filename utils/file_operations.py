import os
import joblib
import json

def save_model(model, filepath):
    """
    Save a model to the specified filepath.

    Args:
        model: Trained model object to save.
        filepath (str): Path to save the model.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """
    Load a model from the specified filepath.

    Args:
        filepath (str): Path to load the model from.

    Returns:
        Loaded model object.
    """
    if os.path.exists(filepath):
        return joblib.load(filepath)
    else:
        raise FileNotFoundError(f"No model found at {filepath}")

def save_results(results, filepath):
    """
    Save results (e.g., metrics, hyperparameters) to a JSON file.

    Args:
        results (dict): Results to save.
        filepath (str): Path to save the results.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filepath}")
