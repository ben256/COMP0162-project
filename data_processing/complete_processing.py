import logging

from data_processing.create_datasets import format_data, create_datasets
from data_processing.process_market_data import process_market_data
from data_processing.process_stock_data import fetch_data

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(filename)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)


def main():
    # process_market_data()  # Already on github
    fetch_data()  # Fetch stock data to create data/stock/stock_data.csv
    format_data()   # Create market_features.csv and stock_features.csv
    create_datasets()  # Create train, validation and test datasets


if __name__ == '__main__':
    main()