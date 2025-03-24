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


def create_windows(dataset, window_length=60):
    windows_list = []
    for symbol, group in dataset.groupby('symbol'):
        windows = create_sliding_windows(group, window_length)
        if windows.size > 0:
            windows_list.append(windows)

    return np.concatenate(windows_list)


def create_datasets(
        window_length: int = 60,
        validation_length: str = '1y',
        test_length: str = '1y',
        start_date: str = '2015-01-01',
        end_date: str = '2025-01-01',
        stock_data_path: str = '../data/stock',
        market_data_path: str = '../data/market',
        dataset_path: str = '../data/datasets',
        save_csv: bool = True
):
    # Load raw CSVs
    stock_data = pd.read_csv(f'{stock_data_path}/stock_data.csv', parse_dates=['date'], date_format='%Y-%m-%d')
    market_data = pd.read_csv(f'{market_data_path}/market_data.csv', parse_dates=['date'], date_format='%Y-%m-%d')

    # Ensure data is sorted by date
    stock_data.sort_values(by='date', ascending=True, inplace=True)
    market_data.sort_values(by='date', ascending=True, inplace=True)

    # Set rolling window sizes
    windows = [int(window_length/12), int(window_length/6), int(window_length/3), int(window_length/2), window_length]

    # Rename columns
    market_data.rename(columns={'spx_price': 'returns', 'spx_vol': 'volume'}, inplace=True)

    # Convert market features to relative
    market_data[['returns', 'volume', 'vix', 'cor1m']] = market_data[['returns', 'volume', 'vix', 'cor1m']].pct_change()

    # Calculate features over different windows
    for window in windows:
        market_data[f'returns_ma_{window}'] = market_data['returns'].rolling(window=window, min_periods=window).mean()
        market_data[f'returns_std_{window}'] = market_data['returns'].rolling(window=window, min_periods=window).std()
        market_data[f'volume_ma_{window}'] = market_data['volume'].rolling(window=window, min_periods=window).mean()
        market_data[f'volume_std_{window}'] = market_data['volume'].rolling(window=window, min_periods=window).std()

    feature_cols = market_data.columns.to_list()
    feature_cols.remove('date')

    for col in feature_cols:
        mean = market_data[col].mean()
        std = market_data[col].std()
        market_data[f'z_{col}'] = (market_data[col] - mean) / std
        # Clip Z-scores to be within [-3, 3]
        market_data[f'z_{col}'] = market_data[f'z_{col}'].clip(lower=-3, upper=3)

    market_features = market_data[['date'] + [x for x in market_data.columns.to_list() if x.startswith('z_')]]
    # market_features.columns = pd.Index(['date']).append(market_features.columns[1:].str[2:])

    # Process stock data
    stock_features_list = []
    for symbol, data in stock_data.groupby('symbol'):
        data[['open', 'high', 'low', 'close', 'volume']] = data[['open', 'high', 'low', 'close', 'volume']].pct_change()
        data.rename({'close': 'returns'}, axis=1, inplace=True)

        for window in windows:
            data[f'returns_ma_{window}'] = data['returns'].rolling(window=window, min_periods=window).mean()
            data[f'returns_std_{window}'] = data['returns'].rolling(window=window, min_periods=window).std()
            data[f'volume_ma_{window}'] = data['volume'].rolling(window=window, min_periods=window).mean()
            data[f'volume_std_{window}'] = data['volume'].rolling(window=window, min_periods=window).std()

        data.dropna(inplace=True)

        stock_features_list.append(data)

    stock_features = pd.concat(stock_features_list)
    stock_features.dropna(inplace=True)

    feature_cols = stock_features.columns.to_list()
    feature_cols = [x for x in feature_cols if x not in ['date', 'symbol', 'open', 'high', 'low', 'close']]

    for col in feature_cols:
        mean = stock_features[col].mean()
        std = stock_features[col].std()
        stock_features[f'z_{col}'] = (stock_features[col] - mean) / std
        # Clip Z-scores to be within [-3, 3]
        stock_features[f'z_{col}'] = stock_features[f'z_{col}'].clip(lower=-3, upper=3)

    for symbol, data in stock_features.groupby('symbol'):
        data['target'] = data['z_returns'].shift(-1)
        data['target_sign'] = np.where(data['target'] > 0, 1, 0)
        stock_features.loc[data.index, 'target'] = data['target']
        stock_features.loc[data.index, 'target_sign'] = data['target_sign']

    targets = stock_features[['date', 'symbol', 'target', 'target_sign']]
    stock_features = stock_features[['date', 'symbol'] + [x for x in stock_features.columns.to_list() if x.startswith('z_')]]

    features = stock_features.merge(market_features, on='date', suffixes=('_stock', '_market'))
    features = features.merge(targets, on=['date', 'symbol'])  # Ensure targets are the last two dimensions
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

    if save_csv:
        train.to_csv(f'{dataset_path}/train.csv', index=False)
        validation.to_csv(f'{dataset_path}/validation.csv', index=False)
        test.to_csv(f'{dataset_path}/test.csv', index=False)

    # Create sliding windows for each dataset
    train_windows = create_windows(train, window_length)
    validation_windows = create_windows(validation, window_length)
    test_windows = create_windows(test, window_length)

    # Save the datasets
    np.save(f'{dataset_path}/train.npy', train_windows)
    np.save(f'{dataset_path}/validation.npy', validation_windows)
    np.save(f'{dataset_path}/test.npy', test_windows)


def create_datasets_exp(
        window_length: int = 60,
        validation_length: str = '1y',
        test_length: str = '1y',
        start_date: str = '2015-01-01',
        end_date: str = '2025-01-01',
        stock_data_path: str = '../data/stock',
        market_data_path: str = '../data/market',
        dataset_path: str = '../data/datasets',
        save_csv: bool = False
):
    import pandas as pd
    import numpy as np

    # Load raw CSVs
    stock_data = pd.read_csv(f'{stock_data_path}/stock_data.csv',
                             parse_dates=['date'], date_format='%Y-%m-%d')
    market_data = pd.read_csv(f'{market_data_path}/market_data.csv',
                              parse_dates=['date'], date_format='%Y-%m-%d')

    # Ensure data is sorted by date
    stock_data.sort_values(by='date', ascending=True, inplace=True)
    market_data.sort_values(by='date', ascending=True, inplace=True)

    # Set rolling window sizes
    windows = [int(window_length/12), int(window_length/6),
               int(window_length/3), int(window_length/2), window_length]

    # ----- Process Market Data -----
    # Rename columns for consistency
    market_data.rename(columns={'spx_price': 'returns', 'spx_vol': 'volume'}, inplace=True)
    # Convert market features to relative changes (e.g., daily returns)
    market_data[['returns', 'volume', 'vix', 'cor1m']] = market_data[['returns', 'volume', 'vix', 'cor1m']].pct_change()

    # Calculate rolling features for the market data
    for window in windows:
        market_data[f'returns_ma_{window}'] = market_data['returns'].rolling(window=window, min_periods=window).mean()
        market_data[f'returns_std_{window}'] = market_data['returns'].rolling(window=window, min_periods=window).std()
        market_data[f'volume_ma_{window}'] = market_data['volume'].rolling(window=window, min_periods=window).mean()
        market_data[f'volume_std_{window}'] = market_data['volume'].rolling(window=window, min_periods=window).std()

    # We'll keep the raw market features (other than date)
    market_feature_cols = [col for col in market_data.columns if col != 'date']
    market_features = market_data[['date'] + market_feature_cols]

    # ----- Process Stock Data -----
    stock_features_list = []
    for symbol, data in stock_data.groupby('symbol'):
        # Compute percentage changes on OHLC and volume (making the series more stationary)
        data[['open', 'high', 'low', 'close', 'volume']] = data[['open', 'high', 'low', 'close', 'volume']].pct_change()
        # Rename close to returns
        data.rename({'close': 'returns'}, axis=1, inplace=True)

        # Calculate rolling features for each stock
        for window in windows:
            data[f'returns_ma_{window}'] = data['returns'].rolling(window=window, min_periods=window).mean()
            data[f'returns_std_{window}'] = data['returns'].rolling(window=window, min_periods=window).std()
            data[f'volume_ma_{window}'] = data['volume'].rolling(window=window, min_periods=window).mean()
            data[f'volume_std_{window}'] = data['volume'].rolling(window=window, min_periods=window).std()

        data.dropna(inplace=True)
        stock_features_list.append(data)

    stock_features = pd.concat(stock_features_list)
    stock_features.dropna(inplace=True)

    # For stock features, define the columns to use (exclude raw OHLC)
    stock_feature_cols = [col for col in stock_features.columns
                          if col not in ['date', 'symbol', 'open', 'high', 'low', 'close']]

    # ----- Merge Market and Stock Features -----
    # Append suffixes to differentiate stock and market features
    features = stock_features.merge(market_features, on='date', suffixes=('_stock', '_market'))
    features.sort_values(by='date', inplace=True)

    # Compute target from stock returns (using raw returns; normalization will come later)
    features['target'] = features.groupby('symbol')['returns_stock'].shift(-1)
    features['target_sign'] = np.where(features['target'] > 0, 1, 0)
    features.dropna(inplace=True)

    # ----- Split into Train, Validation, and Test -----
    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date)
    # parse_time_offset should convert strings like '1y' into a pd.Timedelta or pd.DateOffset
    val_offset = parse_time_offset(validation_length)
    test_offset = parse_time_offset(test_length)
    test_start_date = end_date_dt - test_offset
    val_start_date = test_start_date - val_offset

    train = features[features['date'] < val_start_date].copy()
    validation = features[(features['date'] >= val_start_date) & (features['date'] < test_start_date)].copy()
    test = features[features['date'] >= test_start_date].copy()

    # ----- Normalize Features Using Training Set Statistics -----
    # Identify columns to normalize: all except date, symbol, target, target_sign.
    norm_cols = [col for col in features.columns if col not in ['date', 'symbol', 'target', 'target_sign']]

    # Compute training set mean and std for each column
    train_means = train[norm_cols].mean()
    train_stds = train[norm_cols].std()

    def normalize_df(df, cols, means, stds):
        df_norm = df.copy()
        for col in cols:
            df_norm[col] = (df_norm[col] - means[col]) / stds[col]
            df_norm[col] = df_norm[col].clip(lower=-3, upper=3)
        return df_norm

    train_norm = normalize_df(train, norm_cols, train_means, train_stds)
    validation_norm = normalize_df(validation, norm_cols, train_means, train_stds)
    test_norm = normalize_df(test, norm_cols, train_means, train_stds)

    # ----- Optionally Save CSVs -----
    if save_csv:
        train_norm.to_csv(f'{dataset_path}/train.csv', index=False)
        validation_norm.to_csv(f'{dataset_path}/validation.csv', index=False)
        test_norm.to_csv(f'{dataset_path}/test.csv', index=False)

    # ----- Create Sliding Windows -----
    train_windows = create_windows(train_norm, window_length)
    validation_windows = create_windows(validation_norm, window_length)
    test_windows = create_windows(test_norm, window_length)

    # Save the datasets as .npy files
    np.save(f'{dataset_path}/train.npy', train_windows)
    np.save(f'{dataset_path}/validation.npy', validation_windows)
    np.save(f'{dataset_path}/test.npy', test_windows)

    return train_norm, validation_norm, test_norm


# if __name__ == '__main__':
#     create_datasets()
