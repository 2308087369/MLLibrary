from hyperopt import hp, fmin, tpe, Trials
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import pandas as pd
from data_processing.preprocessing import preprocess_data

# 数据加载
file_path = "test_solar_data.xlsx"
test_data = pd.read_excel(file_path, engine="openpyxl")

# 数据预处理（省略了 preprocess_data 函数的实现）
processed_data = preprocess_data(
    test_data,
    target_column="Active_Power",
    handle_missing="mean",
    outliers=True,
    feature="mi",
    normalize=True,
    split_ratio=(0.7, 0.15, 0.15)
)

X_train, y_train = processed_data["train_data"].to_numpy(), processed_data["train_labels"].to_numpy()

# 定义参数空间
rf_param_space = {
    "n_estimators": hp.choice("n_estimators", [50, 100, 150, 200]),
    "max_depth": hp.choice("max_depth", [5, 10, 15, 20]),
    "min_samples_split": hp.choice("min_samples_split", [2, 5, 10]),
}


# 目标函数
def rf_model_fn(params):
    print("Evaluating parameters:", params)
    try:
        model = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            random_state=42,
        )
        scores = cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv=3, n_jobs=-1)
        mean_score = -scores.mean()  # Convert to positive MSE
        print(f"Mean Score: {mean_score:.4f}")
        return mean_score
    except Exception as e:
        print("Error during model evaluation:", e)
        return float("inf")

# 贝叶斯优化
print("\n--- Bayesian Optimization (Random Forest) ---")
trials = Trials()
best_params = fmin(
    fn=rf_model_fn,
    space=rf_param_space,
    algo=tpe.suggest,
    max_evals=30,
    trials=trials,
)

print("\nBest Parameters:", best_params)
