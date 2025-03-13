"""
Create datasets for training and testing.

This script assumes that the data is:
- the correct length (e.g., the start and end dates are the same),
- complete and continuous,
- aligns with NYSE/Nasdaq trading days
"""

import logging

import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler

from data_processing.utils import parse_time_offset

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(filename)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)


def create_sliding_windows2(
        features: pd.DataFrame,
        window_length: int = 60,
):
    """
    Create a sliding window of data for each stock.
    """
    features_array = features.drop(columns=['date', 'symbol']).to_numpy()

    windows = []
    for index in range(0, len(features_array) - window_length):
        windows.append(features_array[index:index + window_length, :])

    return windows

def create_sliding_windows(group_data, window_length):
    # group_data: numpy array of shape [n, num_features]
    group_array = group_data.drop(columns=['date', 'symbol']).to_numpy()

    if len(group_array) < window_length:
        return np.empty((0, window_length, group_array.shape[1]))
    windows = np.lib.stride_tricks.sliding_window_view(group_array, window_length, axis=0)
    return windows.swapaxes(1, 2)


def format_data(
        stock_data_path = '../data/stock',
        market_data_path = '../data/market',
):
    """
    Normalise data and calculate return ratio.
    """

    stock_data = pd.read_csv(f'{stock_data_path}/stock_data.csv', parse_dates=['date'], date_format='%Y-%m-%d')
    market_data = pd.read_csv(f'{market_data_path}/market_data.csv', parse_dates=['date'], date_format='%Y-%m-%d')

    # Ensure data is sorted by date
    stock_data.sort_values(by='date', ascending=True)
    market_data.sort_values(by='date', ascending=True)

    # Process stock features
    processed_stock_features = []

    grouped_data = stock_data.groupby('symbol')
    for symbol, data in grouped_data:
        data['returns'] = data['close'].pct_change()
        data['high_low_spread'] = data['high'] - data['low']
        data['close_open_spread'] = data['close'] - data['open']
        data['ma10'] = data['close'].rolling(window=10).mean()
        data['momentum'] = data['close'] - data['ma10']
        data['target'] = data['returns'].shift(-1)
        data['target_sign'] = np.where(data['target'] > 0, 1, 0)

        data.drop(columns=['close', 'high', 'low', 'open', 'ma10'], inplace=True)

        data.dropna(inplace=True)
        processed_stock_features.append(data)

    processed_stock_features = pd.concat(processed_stock_features)

    numerical_cols = processed_stock_features.select_dtypes(include=['float64', 'int64']).columns.tolist()
    for col in ['target', 'target_sign']:
        if col in numerical_cols:
            numerical_cols.remove(col)

    scaler = StandardScaler()
    processed_stock_features[numerical_cols] = scaler.fit_transform(processed_stock_features[numerical_cols])
    processed_stock_features.to_csv(f'{stock_data_path}/stock_features.csv', index=False)

    # Process market features
    processed_market_features = market_data
    numerical_cols = processed_market_features.select_dtypes(include=['float64', 'int64']).columns.tolist()

    scaler = StandardScaler()
    processed_market_features[numerical_cols] = scaler.fit_transform(processed_market_features[numerical_cols])
    processed_market_features.to_csv(f'{market_data_path}/market_features.csv', index=False)


def create_datasets(
        window_length: int = 60,
        validation_length: str = '1y',
        test_length: str = '1y',
        start_date: str = '2015-01-01',
        end_date: str = '2025-01-01',
        stock_features_path: str = '../data/stock',
        market_features_path: str = '../data/market',
        dataset_path: str = '../data/datasets',
):
    stock_features = pd.read_csv(f'{stock_features_path}/stock_features.csv', parse_dates=['date'], date_format='%Y-%m-%d')
    market_features = pd.read_csv(f'{market_features_path}/market_features.csv', parse_dates=['date'], date_format='%Y-%m-%d')

    # Ensure data is sorted by date
    stock_features.sort_values(by='date', ascending=True)
    market_features.sort_values(by='date', ascending=True)

    features = stock_features.merge(market_features, on='date')
    features = features[['date', 'symbol', 'volume', 'returns', 'high_low_spread', 'close_open_spread', 'momentum', 'spx_price', 'spx_vol', 'vix', 'cor1m', 'target', 'target_sign']]

    # Split into train, validation and test sets
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    val_offset = parse_time_offset(validation_length)
    test_offset = parse_time_offset(test_length)

    test_start_date = end_date - test_offset
    val_start_date = test_start_date - val_offset

    # Split the data into train, validation, and test sets
    train_raw = features[features['date'] < val_start_date]
    validation_raw = features[(features['date'] >= val_start_date) & (features['date'] < test_start_date)]
    test_raw = features[features['date'] >= test_start_date]
    datasets_raw = [train_raw, validation_raw, test_raw]

    # Sliding window
    datasets = []
    for dataset in datasets_raw:
        windows = []
        stocks = dataset.groupby('symbol')
        for symbol, data in stocks:
            windows.append(create_sliding_windows(data, window_length))

        joint_windows = np.concatenate(windows)
        datasets.append(joint_windows)

    train = datasets[0]
    validation = datasets[1]
    test = datasets[2]

    np.save(f'{dataset_path}/train.npy', train)
    np.save(f'{dataset_path}/validation.npy', validation)
    np.save(f'{dataset_path}/test.npy', test)


if __name__ == '__main__':
    create_datasets()
