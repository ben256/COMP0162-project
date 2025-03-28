import argparse

from training.ablation_training import run_ablation


def main():
    parser = argparse.ArgumentParser(description="LSTM Training script with configurable paths.")
    parser.add_argument('--dataset_path', type=str, default='../data/datasets',
                        help='Path to the dataset directory')
    parser.add_argument('--output_base_dir', type=str, default='../output/ablation',
                        help='Path to the output')

    args = parser.parse_args()

    run_ablation(
        dataset_path=args.dataset_path,
        output_base_dir=args.output_base_dir
    )


if __name__ == '__main__':
    main()