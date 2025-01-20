import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# 滑动窗口数据生成
def create_sliding_window(data, sequence_length):
    def generate_sequences(X, y):
        num_samples = X.shape[0] - sequence_length + 1
        if num_samples <= 0:
            raise ValueError("Sequence length is too large for the data size.")
        X_seq = np.array([X[i:i + sequence_length] for i in range(num_samples)])
        y_seq = np.array([y[i + sequence_length - 1] for i in range(num_samples)])
        return X_seq, y_seq

    return {
        "train": generate_sequences(data["train_data"].to_numpy(), data["train_labels"].to_numpy()),
        "val": generate_sequences(data["val_data"].to_numpy(), data["val_labels"].to_numpy()),
        "test": generate_sequences(data["test_data"].to_numpy(), data["test_labels"].to_numpy()),
    }


# PyTorch-based time series models
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the last hidden state
        return out


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  # Use the last hidden state
        return out
    
class TransformerModel(nn.Module):
    def __init__(self, input_size, num_heads, num_encoder_layers, hidden_dim, output_size, dropout=0.2):
        super(TransformerModel, self).__init__()
        self.input_proj = nn.Linear(input_size, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 5000, hidden_dim))  # Positional encoding
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        seq_length = x.size(1)
        x = self.input_proj(x) + self.positional_encoding[:, :seq_length, :]
        out = self.transformer_encoder(x)
        out = self.fc(out[:, -1, :])  # Use the last hidden state
        return out


# 模型训练、验证和测试逻辑
def train_and_evaluate_model(processed_data, sequence_length, model_type, hyperparams):
    """
    Train and evaluate a time series model.

    Args:
        processed_data (dict): Preprocessed data with train, val, and test splits.
        sequence_length (int): Length of sequences for time series models.
        model_type (str): Type of model ("LSTM", "GRU", "Transformer", "RandomForest", "XGBoost").
        hyperparams (dict): Hyperparameters for the selected model.

    Returns:
        dict: Trained model and evaluation metrics.
    """
    sliding_window_data = create_sliding_window(processed_data, sequence_length)

    # PyTorch-based models
    input_size = processed_data["train_data"].shape[1]  # Number of features
    if model_type in ["LSTM", "GRU", "Transformer"]:
        hidden_size = hyperparams.get("hidden_size", 64)
        num_layers = hyperparams.get("num_layers", 1)
        dropout = hyperparams.get("dropout", 0.2)
        learning_rate = hyperparams.get("learning_rate", 0.001)

        if model_type == "LSTM":
            model = LSTMModel(input_size, hidden_size, 1, num_layers, dropout)
        elif model_type == "GRU":
            model = GRUModel(input_size, hidden_size, 1, num_layers, dropout)
        elif model_type == "Transformer":
            model = TransformerModel(input_size, num_heads=4, num_encoder_layers=2, hidden_dim=hidden_size, output_size=1)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        train_dataset = TensorDataset(
            torch.tensor(sliding_window_data["train"][0], dtype=torch.float32),
            torch.tensor(sliding_window_data["train"][1], dtype=torch.float32),
        )
        val_dataset = TensorDataset(
            torch.tensor(sliding_window_data["val"][0], dtype=torch.float32),
            torch.tensor(sliding_window_data["val"][1], dtype=torch.float32),
        )
        test_dataset = TensorDataset(
            torch.tensor(sliding_window_data["test"][0], dtype=torch.float32),
            torch.tensor(sliding_window_data["test"][1], dtype=torch.float32),
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)

        # Training
        criterion = nn.MSELoss()
        optimizer = Adam(model.parameters(), lr=learning_rate)
        epochs = hyperparams.get("epochs", 10)
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred.squeeze(), y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    y_pred = model(X_batch)
                    loss = criterion(y_pred.squeeze(), y_batch)
                    val_loss += loss.item()
            val_loss /= len(val_loader)

            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Testing
        test_loader = DataLoader(test_dataset, batch_size=32)
        y_true, y_pred = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                predictions = model(X_batch)
                y_true.extend(y_batch.numpy())
                y_pred.extend(predictions.squeeze().numpy())
        test_mse = mean_squared_error(y_true, y_pred)

    # Machine Learning-based models
    elif model_type in ["RandomForest", "XGBoost"]:
        X_train, y_train = sliding_window_data["train"]
        X_val, y_val = sliding_window_data["val"]
        X_test, y_test = sliding_window_data["test"]

        if model_type == "RandomForest":
            model = RandomForestRegressor(**hyperparams)
        elif model_type == "XGBoost":
            model = XGBRegressor(**hyperparams)

        model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))
        test_mse = mean_squared_error(y_test, y_pred)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return {"model": model, "test_mse": test_mse}