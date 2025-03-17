"""
Market context data processing functions.

These functions process the files found in data/market/raw into usable data.
The processed data will be saved to market_data.csv, located in data/market.
"""

import logging

import pandas as pd
import exchange_calendars as xcals

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(filename)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)

def process_market_data(
        start_date: str = '2015-01-01',
        end_date: str = '2025-01-01',
        market_data_path = '../data/market'
):
    features = ['cor1m', 'spx', 'spx_vol', 'vix']
    market_context_list = []

    # Process SPX
    spx = pd.read_csv(f'{market_data_path}/raw/spx.csv', parse_dates=['date'], date_format='%d/%m/%Y')
    spx_vol = pd.read_csv(f'{market_data_path}/raw/spx_vol.csv', parse_dates=['date'], date_format='%d/%m/%Y')

    spx.rename({'price': 'spx_price'}, axis=1, inplace=True)
    spx_vol.rename({'volume': 'spx_vol'}, axis=1, inplace=True)

    # Process VIX
    vix = pd.read_csv(f'{market_data_path}/raw/vix.csv', parse_dates=['date'], date_format='%d/%m/%Y')
    vix.rename({'price': 'vix'}, axis=1, inplace=True)

    # Process COR1M
    cor1m = pd.read_csv(f'{market_data_path}/raw/cor1m.csv', parse_dates=['date'], date_format='%d/%m/%Y')
    cor1m.rename({'price': 'cor1m'}, axis=1, inplace=True)

    # Merge all data
    market_context_list.append(spx)
    market_context_list.append(spx_vol)
    market_context_list.append(vix)
    market_context_list.append(cor1m)

    market_context = spx.merge(spx_vol, on='date', how='left')
    market_context = market_context.merge(vix, on='date', how='left')
    market_context = market_context.merge(cor1m, on='date', how='left')

    # Drop any data points with nans
    market_context = market_context.dropna()

    market_context = market_context[(market_context['date'] >= start_date) & (market_context['date'] < end_date)]
    nyse_calendar = xcals.get_calendar('NYSE').sessions_in_range(start_date, end_date)
    market_context = market_context[market_context['date'].isin(nyse_calendar)]

    market_context.to_csv(f'{market_data_path}/market_data.csv', index=False, date_format='%Y-%m-%d')

#
# if __name__ == '__main__':
#     process_market_data()
