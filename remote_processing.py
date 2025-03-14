import logging

from data_processing.create_datasets import format_data, create_datasets
from data_processing.process_market_data import process_market_data
from data_processing.process_stock_data import fetch_data

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(filename)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)


def main():
    # process_market_data()  # Already on github
    fetch_data(
        reference_data_path = './data/reference',
        stock_data_path = './data/stock'
    )  # Fetch stock data to create data/stock/stock_data.csv
    format_data(
        stock_data_path = './data/stock',
        market_data_path = './data/market'
    )   # Create market_features.csv and stock_features.csv
    create_datasets(
        stock_features_path = './data/stock',
        market_features_path = './data/market',
        dataset_path = './data/datasets',
    )  # Create train, validation and test datasets


if __name__ == '__main__':
    main()