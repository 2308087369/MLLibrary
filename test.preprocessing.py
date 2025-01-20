import pandas as pd
from data_processing.preprocessing import preprocess_data

# 加载测试数据，指定正确路径和引擎
file_path = "/home/chenbo/my_project/test_solar_data.xlsx"
test_data = pd.read_excel(file_path, engine="openpyxl")  # 指定引擎以确保读取正确

# 定义目标列
target_column = "Active_Power"  # 假设目标列是 Active_Power

# 测试 1: 基础预处理，处理缺失值和异常值，不进行特征选择
print("=== Test 1: Basic Preprocessing ===")
preprocessed_data_1 = preprocess_data(
    test_data,
    target_column=target_column,
    handle_missing="mean",  # 填充缺失值为均值
    outliers=True,  # 启用异常值处理
    feature="none",  # 不进行特征选择
    normalize=True,  # 启用归一化
    split_ratio=(0.7, 0.15, 0.15)  # 默认训练集/验证集/测试集比例
)
print("Test 1 Completed: Preprocessed data shape:")
print(f"Train Data Shape: {preprocessed_data_1['train_data'].shape}")
print(f"Validation Data Shape: {preprocessed_data_1['val_data'].shape}")
print(f"Test Data Shape: {preprocessed_data_1['test_data'].shape}")

# 测试 2: 使用相关性分析进行特征选择
print("\n=== Test 2: Feature Selection with Correlation Analysis ===")
preprocessed_data_2 = preprocess_data(
    test_data,
    target_column=target_column,
    handle_missing="mean",
    outliers=True,
    feature="ca",  # 相关性分析
    feature_threshold=0.2,  # 特征选择阈值
    normalize=True
)
print("Test 2 Completed: Selected Features:")
print(preprocessed_data_2['train_data'].columns)

# 测试 3: 使用互信息分析进行特征选择
print("\n=== Test 3: Feature Selection with Mutual Information ===")
preprocessed_data_3 = preprocess_data(
    test_data,
    target_column=target_column,
    handle_missing="mean",
    outliers=True,
    feature="mi",  # 互信息分析
    feature_threshold=0.2,  # 特征选择阈值
    normalize=True,
    split_ratio=(0.7, 0.15, 0.15)
)
print("Test 3 Completed: Selected Features:")
print(preprocessed_data_3['train_data'].columns)

