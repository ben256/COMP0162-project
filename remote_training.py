import logging
import argparse

from training.train import train

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(filename)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Training script with configurable paths.")
    parser.add_argument('--dataset_path', type=str, default='/scratch0/bnaylor/datasets',
                        help='Path to the dataset directory')
    parser.add_argument('--output_dir', type=str, default='/scratch0/bnaylor/output',
                        help='Path to the output directory')
    args = parser.parse_args()

    train(
        dataset_path = args.dataset_path,
        output_dir = args.output_dir
    )

if __name__ == '__main__':
    main()