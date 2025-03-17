import logging
import argparse

from training.train import train

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(filename)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Training script with configurable paths.")
    parser.add_argument('--dataset_path', type=str, default='./data/datasets',
                        help='Path to the dataset directory')
    parser.add_argument('--output_dir', type=str, default='./data/output',
                        help='Path to the output directory')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs for training')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                        help='Number of epochs to wait after last improvement')
    parser.add_argument('--early_stopping_delta', type=float, default=0.0,
                        help='Minimum change in validation loss to qualify as an improvement')
    parser.add_argument('--early_stopping_offset', type=int, default=20,
                        help='Number of epochs before early stopping starts')
    parser.add_argument('--shuffle_train_data', type=bool, default=True,
                        help='Shuffle training data')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Number of transformer layers')
    parser.add_argument('--num_head', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--prediction_type', type=str, default='last',
                        help='Type of prediction head')

    args = parser.parse_args()

    train(
        batch_size = args.batch_size,
        learning_rate = args.learning_rate,
        epochs = args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_delta=args.early_stopping_delta,
        early_stopping_offset=args.early_stopping_offset,
        shuffle_train_data=args.shuffle_train_data,
        dropout=args.dropout,
        num_layers=args.num_layers,
        num_head=args.num_head,
        prediction_type=args.prediction_type,
        dataset_path = args.dataset_path,
        output_dir = args.output_dir
    )

if __name__ == '__main__':
    main()