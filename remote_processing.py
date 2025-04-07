import logging
import argparse

from data_processing.create_datasets import create_datasets
from data_processing.process_market_data import process_market_data
from data_processing.process_stock_data import fetch_stock_data

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(filename)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="Data processing script with configurable paths.")
    parser.add_argument('--reference_data_path', type=str, default='./data/reference',
                        help='Path to the reference data directory')
    parser.add_argument('--stock_data_path', type=str, default='./data/stock',
                        help='Path to the stock data directory')
    parser.add_argument('--market_data_path', type=str, default='./data/market',
                        help='Path to the market data directory')
    parser.add_argument('--dataset_path', type=str, default='./data/datasets',
                        help='Path to the output dataset directory')
    parser.add_argument('--save_csv', type=bool, default=False,
                        help='Save datasets to CSV files prior to windows.')

    args = parser.parse_args()

    # process_market_data()  # Already on GitHub
    fetch_stock_data(
        reference_data_path=args.reference_data_path,
        stock_data_path=args.stock_data_path
    )
    create_datasets(
        stock_data_path=args.stock_data_path,
        market_data_path=args.market_data_path,
        dataset_path=args.dataset_path,
        save_csv=args.save_csv
    )

if __name__ == '__main__':
    main()
