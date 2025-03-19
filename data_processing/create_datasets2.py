import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from data_processing.utils import parse_time_offset

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(filename)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)


def create_sliding_windows(group_data, window_length):
    # Exclude non-feature columns (i.e. date and symbol) when creating windows.
    group_array = group_data.drop(columns=['date', 'symbol']).to_numpy()
    if len(group_array) < window_length:
        return np.empty((0, window_length, group_array.shape[1]))
    windows = np.lib.stride_tricks.sliding_window_view(group_array, window_length, axis=0)
    return windows.swapaxes(1, 2)


def create_windows(dataset, cols_to_scale=None, window_length=60):
    windows_list = []
    for symbol, group in dataset.groupby('symbol'):
        windows = create_sliding_windows(group, window_length)
        if windows.size > 0:
            windows_list.append(windows)
    if windows_list:
        return np.concatenate(windows_list)
    else:
        # Return an empty array with the correct shape if no windows were created
        num_features = len(cols_to_scale)
        return np.empty((0, window_length, num_features))


def create_datasets(
        window_length: int = 60,
        validation_length: str = '1y',
        test_length: str = '1y',
        start_date: str = '2015-01-01',
        end_date: str = '2025-01-01',
        stock_data_path: str = '../data/stock',
        market_data_path: str = '../data/market',
        dataset_path: str = '../data/datasets',
):
    # Load raw CSVs
    stock_data = pd.read_csv(f'{stock_data_path}/stock_data.csv', parse_dates=['date'], date_format='%Y-%m-%d')
    market_data = pd.read_csv(f'{market_data_path}/market_data.csv', parse_dates=['date'], date_format='%Y-%m-%d')

    # Ensure data is sorted by date
    stock_data.sort_values(by='date', inplace=True)
    market_data.sort_values(by='date', inplace=True)

    # Process stock features
    processed_stock_features = []
    for symbol, data in stock_data.groupby('symbol'):
        data = data.copy()  # Work on a copy to avoid warnings.
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

    # Process market features
    market_data['spx_returns'] = market_data['spx_price'].pct_change()
    market_data.drop(columns=['spx_price'], inplace=True)
    market_data.dropna(inplace=True)
    processed_market_features = market_data

    # Merge the processed stock and market features
    features = processed_stock_features.merge(processed_market_features, on='date')
    features = features[['date', 'symbol', 'volume', 'returns', 'high_low_spread', 'close_open_spread',
                         'momentum', 'spx_returns', 'spx_vol', 'vix', 'cor1m', 'target', 'target_sign']]
    features.sort_values(by='date', inplace=True)

    # Convert date strings to datetime objects for splitting
    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date)

    # Determine split dates using time offsets
    val_offset = parse_time_offset(validation_length)
    test_offset = parse_time_offset(test_length)
    test_start_date = end_date_dt - test_offset
    val_start_date = test_start_date - val_offset

    # Split into train, validation, and test sets
    train = features[features['date'] < val_start_date].copy()
    validation = features[(features['date'] >= val_start_date) & (features['date'] < test_start_date)].copy()
    test = features[features['date'] >= test_start_date].copy()

    # Identify numerical columns to scale (exclude date, symbol, and targets)
    cols_to_scale = [col for col in features.columns if col not in ['date', 'symbol', 'target', 'target_sign']]

    # Fit the scaler on the training set only
    scaler = StandardScaler()
    train[cols_to_scale] = scaler.fit_transform(train[cols_to_scale])
    # Transform the validation and test sets using the same scaler
    validation[cols_to_scale] = scaler.transform(validation[cols_to_scale])
    test[cols_to_scale] = scaler.transform(test[cols_to_scale])

    # Create sliding windows for each dataset
    train_windows = create_windows(train, cols_to_scale, window_length)
    validation_windows = create_windows(validation, cols_to_scale, window_length)
    test_windows = create_windows(test, cols_to_scale, window_length)

    # Save the datasets
    np.save(f'{dataset_path}/train.npy', train_windows)
    np.save(f'{dataset_path}/validation.npy', validation_windows)
    np.save(f'{dataset_path}/test.npy', test_windows)


# if __name__ == '__main__':
#     create_datasets()
