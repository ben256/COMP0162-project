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
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs for training')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                        help='Number of epochs to wait after last improvement')
    parser.add_argument('--early_stopping_delta', type=float, default=0.0,
                        help='Minimum change in validation loss to qualify as an improvement')
    parser.add_argument('--early_stopping_offset', type=int, default=20,
                        help='Number of epochs before early stopping starts')

    args = parser.parse_args()

    train(
        batch_size = args.batch_size,
        learning_rate = args.learning_rate,
        epochs = args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_delta=args.early_stopping_delta,
        early_stopping_offset=args.early_stopping_offset,
        dataset_path = args.dataset_path,
        output_dir = args.output_dir
    )

if __name__ == '__main__':
    main()