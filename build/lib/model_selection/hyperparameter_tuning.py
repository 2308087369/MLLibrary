import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score
from hyperopt import fmin, tpe, Trials
from hyperopt import hp

def tune_hyperparameters(model, param_space, X_train, y_train, method="grid", cv=3, max_evals=50, scoring="neg_mean_squared_error"):
    """
    Unified hyperparameter tuning function.

    Args:
        model: The machine learning model to tune.
        param_space (dict): Hyperparameter space for tuning.
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        method (str): Tuning method - 'grid' for grid search or 'bayes' for Bayesian optimization.
        cv (int): Number of cross-validation folds.
        max_evals (int): Maximum evaluations for Bayesian optimization.
        scoring (str): Scoring method.

    Returns:
        dict: Results containing best parameters, best score, and search/trials object.
    """
    if method == "grid":
        # Perform Grid Search
        print("Starting Grid Search...")
        grid_search = GridSearchCV(
            model,
            param_space,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            verbose=1,
        )
        grid_search.fit(X_train, y_train)
        results = {
            "best_params": grid_search.best_params_,
            "best_score": -grid_search.best_score_,  # Convert negative MSE to positive
            "search": grid_search,
        }
    elif method == "bayes":
        # Perform Bayesian Optimization
        print("Starting Bayesian Optimization...")
        
        def objective(params):
            model.set_params(**params)
            scores = cross_val_score(model, X_train, y_train, scoring=scoring, cv=cv, n_jobs=-1)
            mean_score = -scores.mean()  # Convert to positive MSE
            return mean_score

        trials = Trials()
        best = fmin(
            fn=objective,
            space=param_space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
        )
        results = {
            "best_params": best,
            "trials": trials,
        }
    else:
        raise ValueError("Invalid tuning method. Use 'grid' or 'bayes'.")

    return results

def visualize_results(results, method):
    """
    Visualize tuning results in a clear tabular format.

    Args:
        results (dict): Tuning results returned by `tune_hyperparameters`.
        method (str): The tuning method used ('grid' or 'bayes').

    Returns:
        None
    """
    if method == "grid":
        # Extract and visualize Grid Search results
        cv_results = results["search"].cv_results_
        df = pd.DataFrame(cv_results)
        display_cols = ["mean_test_score", "std_test_score", "params"]
        print("\nGrid Search Results:")
        print(df[display_cols].sort_values(by="mean_test_score", ascending=False))
    elif method == "bayes":
        # Extract and visualize Bayesian Optimization results
        trials = results["trials"]
        df = pd.DataFrame(trials.results)
        df["params"] = [str(t["misc"]["vals"]) for t in trials.trials]
        print("\nBayesian Optimization Results:")
        print(df.sort_values(by="loss", ascending=True))
    else:
        raise ValueError("Invalid tuning method. Use 'grid' or 'bayes'.")
