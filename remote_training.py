import logging

from training.train import train

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(filename)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)


def main():
    train(
        dataset_path = '/scratch0/bnaylor/datasets',
        output_dir = '/scratch0/bnaylor/output'
    )

if __name__ == '__main__':
    main()