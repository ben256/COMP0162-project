import numpy as np
import torch
import torch.nn as nn

class NullModel:
    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def predict(self, stock_data, market_data, targets=None):
        if isinstance(stock_data, torch.Tensor):
            stock_data = stock_data.cpu().numpy()

        returns_array = np.zeros_like(stock_data[:, -1, 0])

        return returns_array


class NaiveModel:
    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def predict(self, stock_data, market_data, targets=None):
        if isinstance(stock_data, torch.Tensor):
            stock_data = stock_data.cpu().numpy()

        returns_array = stock_data[:, -1, 0]
        return returns_array


class LSTMModel(nn.Module):
    def __init__(
            self,
            stock_input_dim=22,
            embed_dim=128,
            num_layers=2,
            dropout=0.1
    ):
        super().__init__()

        self.stock_lstm = nn.LSTM(
            input_size=stock_input_dim,
            hidden_size=embed_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc_out = nn.Linear(embed_dim, 1)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, stock_data, market_data=None):
        stock_out, _ = self.stock_lstm(stock_data)
        stock_features = stock_out[:, -1, :]  # Get the last time step

        output = self.fc_out(stock_features)
        return output

    def predict(self, stock_data, market_data, targets=None):
        stock_out, _ = self.stock_lstm(stock_data)
        stock_features = stock_out[:, -1, :]  # Get the last time step

        output = self.fc_out(stock_features)
        return self.prediction_head(output)