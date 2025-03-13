"""
Stock data processing functions.

These functions fetch the stock data from yFinance and process it for use in the dataset.
The downloaded price data will be stored in stock_data.csv, located in data/stock/raw.
If the data is already downloaded locally, then the data will be checked and if okay then .
The following data is downloaded:
- High Price
- Low Price
- Open Price
- Close Price
- Daily Volume

"""
import logging

import numpy as np
import pandas as pd
import exchange_calendars as xcals
import yfinance
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(filename)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)


def fetch_data(
        total_stock_count: int = 100,
        start_date: str = '2015-01-01',
        end_date: str = '2025-01-01',
        reference_data_path = '../data/reference',
        stock_data_path = '../data/stock/raw'
):
    stock_data = pd.read_csv(f'{stock_data_path}/stock_data.csv', parse_dates=['date'], date_format='%Y-%m-%d')
    reference_data = pd.read_csv(f'{reference_data_path}/valid_stocks.csv', parse_dates=['date_added'], date_format='%Y-%m-%d')

    nyse_calendar = xcals.get_calendar('NYSE').sessions_in_range(start_date, end_date)

    to_fetch = []
    fetched = []
    stock_count = 0

    # Check if stock data is already downloaded
    for index, row in reference_data.iterrows():
        symbol = row['symbol']
        symbol_subset = stock_data[stock_data['symbol'] == symbol]
        if not symbol_subset.empty:
            dates_symbol = pd.DatetimeIndex(symbol_subset['date'].dt.date)

            if not np.array_equal(dates_symbol, nyse_calendar):
                to_fetch.append(symbol)
                stock_data.drop(symbol_subset.index, inplace=True)
            else:
                stock_count += 1
        else:
            to_fetch.append(symbol)

    existing_stock_count = stock_count

    for symbol in tqdm(to_fetch, total=total_stock_count-existing_stock_count):

        if stock_count == total_stock_count:
            logging.info(f'Reached stock count limit of {total_stock_count}')
            break

        if '.' in symbol:
            symbol = symbol.split('.')[0]
        symbol_data = yfinance.Ticker(symbol).history(
            start=start_date,
            end=end_date,
            interval='1D'
        )
        dates_yf = symbol_data.index.tz_localize(None).normalize()

        if np.array_equal(dates_yf, nyse_calendar):
            symbol_data = symbol_data[['Open', 'High', 'Low', 'Close', 'Volume']]
            symbol_data['date'] = dates_yf
            symbol_data['symbol'] = symbol

            symbol_data.rename({
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }, axis=1, inplace=True)

            fetched.append(symbol_data)
            stock_count += 1
        else:
            logging.warning(f'Not enough data for {symbol}, skipping')

    if fetched:
        logging.info(f'Fetched data for {len(fetched)} stocks')
        logging.info(f'Saving data to CSV')
        stock_data = pd.concat([stock_data] + fetched, ignore_index=True)
        stock_data.to_csv(f'{stock_data_path}/stock_data.csv', index=False, date_format='%Y-%m-%d')
    else:
        logging.info('No new data fetched')


if __name__ == '__main__':
    fetch_data()
