"""
Creates a list of possible valid stocks for the dataset.

The main selection criteria is that the stock has been present on the S&P since before 2015.
The finalised list is saved to a CSV file: valid_stocks.csv.
This script is designed to be run independently and prior to any processing/training.
"""

import logging
import datetime

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(filename)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)


def lower_columns(df):
    lower_cols = df.columns.str.lower()
    df.columns = lower_cols
    return df

def main():
    sp500 = pd.read_csv('/Users/bennaylor/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Documents/Computational Finance/COMP0162/Coursework/COMP0162-project/data/reference/sp500.csv', parse_dates=['Date added'], date_format='%d/%m/%Y')
    nasdaq = pd.read_csv('/Users/bennaylor/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Documents/Computational Finance/COMP0162/Coursework/COMP0162-project/data/reference/nasdaq.csv')
    nyse = pd.read_csv('/Users/bennaylor/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Documents/Computational Finance/COMP0162/Coursework/COMP0162-project/data/reference/nyse.csv')

    # Rename and lower column titles
    sp500 = lower_columns(sp500)[['symbol', 'security', 'gics sector', 'gics sub-industry', 'date added', 'cik']].rename({
        'gics sector': 'sector',
        'gics sub-industry': 'sub_industry',
        'date added': 'date_added'
    }, axis=1)
    nasdaq = lower_columns(nasdaq)[['symbol', 'market cap']].rename({'market cap': 'market_cap'}, axis=1)
    nyse = lower_columns(nyse)[['symbol', 'market cap']].rename({'market cap': 'market_cap'}, axis=1)

    # Ensure only . used in tickers
    nasdaq['symbol'] = nasdaq['symbol'].str.replace('^', '.').str.replace('/', '.')
    nyse['symbol'] = nyse['symbol'].str.replace('^', '.').str.replace('/', '.')

    # Merge to get market cap
    sp500 = sp500.merge(nasdaq, on='symbol', how='left')
    sp500 = sp500.merge(nyse, on='symbol', how='left')
    sp500['market_cap'] = sp500['market_cap_x'].fillna(sp500['market_cap_y'])
    sp500.drop(['market_cap_x', 'market_cap_y'], axis=1, inplace=True)  # Drop any stocks with missing market caps

    # remove stocks not present added before 2010
    sp500 = sp500[sp500['date_added'] < pd.to_datetime(datetime.date(2015, 1, 1))]
    sp500.sort_values('market_cap', ascending=False, inplace=True)  # Sort by market cap
    sp500.drop_duplicates(subset='cik', keep='first', inplace=True)  # Drop duplicates of different share class

    sp500.to_csv('/Users/bennaylor/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Documents/Computational Finance/COMP0162/Coursework/COMP0162-project/data/reference/valid_stocks.csv', index=False, date_format='%Y-%m-%d')


if __name__ == '__main__':
    main()
