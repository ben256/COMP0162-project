import numpy as np
import torch
from statsmodels.tsa.arima.model import ARIMA


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


class ARIMAModel:
    def __init__(self, order=(1, 0, 0)):
        """
        Args:
            order (tuple): The (p,d,q) order of the ARIMA model.
        """
        self.order = order

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def predict(self, stock_data, market_data, targets=None):
        """
        Given a batch of windows, each window of shape (sequence_length, features),
        this method uses only the returns (assumed to be the second feature, index 1)
        to fit an ARIMA model and forecast the next day return.

        Args:
            stock_data (np.ndarray): Shape (batch_size, sequence_length, features).
            market_data (np.ndarray): Not used in this model.

        Returns:
            np.ndarray: Predictions with shape (batch_size, 1).
        """
        # Extract the returns time series from the second feature.
        # Resulting shape: (batch_size, sequence_length)

        if isinstance(stock_data, torch.Tensor):
            stock_data = stock_data.cpu().numpy()

        returns = stock_data[:, :, 1]
        preds = []
        for i in range(returns.shape[0]):
            ts = returns[i]  # a univariate series of length 60
            try:
                # Fit ARIMA model on the 60-day series.
                model = ARIMA(ts, order=self.order)
                # The 'fit' might issue warnings on convergence; we disable them for clarity.
                fitted = model.fit(method_kwargs={"warn_convergence": False})
                # Forecast one step ahead.
                forecast = fitted.forecast(steps=1)
                preds.append(forecast[0])
            except Exception as e:
                # In case of any errors during fitting, fall back to a prediction of 0.
                preds.append(0.0)
        return np.array(preds).reshape(-1, 1)
