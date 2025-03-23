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
    parser.add_argument('--batch_size', type=int, default=200,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-6,
                        help='Learning rate for training')
    parser.add_argument('--num_training_epochs', type=int, default=100,
                        help='Number of epochs for training')
    parser.add_argument('--num_warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs')
    parser.add_argument('--fusion_type', type=str, default='cross_attn',
                        help='Type of fusion layer')
    parser.add_argument('--embed_dim', type=int, default=128,
                        help='Embedding dimension')
    parser.add_argument('--ff_hidden_dim', type=int, default=256,
                        help='Feedforward hidden dimension')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                        help='Number of epochs to wait after last improvement')
    parser.add_argument('--early_stopping_delta', type=float, default=0.0,
                        help='Minimum change in validation loss to qualify as an improvement')
    parser.add_argument('--early_stopping_offset', type=int, default=5,
                        help='Number of epochs before early stopping starts')
    parser.add_argument('--shuffle_train_data', type=bool, default=True,
                        help='Shuffle training data')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--prediction_type', type=str, default='attn_pool',
                        help='Type of prediction head')

    args = parser.parse_args()

    train(batch_size=args.batch_size, learning_rate=args.learning_rate, num_training_epochs=args.num_training_epochs,
          num_warmup_epochs=args.num_warmup_epochs, fusion_type=args.fusion_type, embed_dim=args.embed_dim,
          ff_hidden_dim=args.ff_hidden_dim, early_stopping_patience=args.early_stopping_patience,
          early_stopping_delta=args.early_stopping_delta, early_stopping_offset=args.early_stopping_offset,
          shuffle_train_data=args.shuffle_train_data, dropout=args.dropout, num_layers=args.num_layers,
          num_heads=args.num_heads, prediction_type=args.prediction_type, dataset_path=args.dataset_path,
          output_dir=args.output_dir
          )


if __name__ == '__main__':
    main()