import logging

from data_processing.create_datasets import create_datasets
from data_processing.process_stock_data import fetch_stock_data

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(filename)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)


def main():
    # process_market_data()  # Already on github
    fetch_stock_data()  # Fetch stock data to create data/stock/stock_data.csv
    create_datasets()  # Create train, validation and test datasets


if __name__ == '__main__':
    main()