import json
import logging
import copy
import os

import numpy as np
import torch
from torch import nn, optim

from torch.utils.data import DataLoader

from data_processing.dataset import CustomDataset
from model.model import ReturnsModel
from training.utils import create_training_folder

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(filename)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)


class EarlyStopping:
    def __init__(self, patience: int = 5, verbose: bool = False, delta: float = 0.0):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in validation loss to qualify as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta
        self.best_model_state = None

    def __call__(self, val_loss, model, epoch=None):
        # First epoch
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            if self.verbose:
                logging.info(f"Initial validation loss: {val_loss:.6f}. Saving model.")
        elif val_loss < self.best_loss - self.delta:
            # Improvement has been made
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            if self.verbose:
                logging.info(f"Validation loss improved to {val_loss:.6f}. Saving model.")
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement in validation loss. Counter: {self.counter + 1}/{self.patience}")
            if epoch >= 20:
                if self.counter >= self.patience:
                    self.early_stop = True


def train(
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        epochs: int = 100,
        dataset_path: str = '../data/datasets',
):
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    output_dir = create_training_folder('../output')
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')

    train_dataset = CustomDataset(np.load(f'{dataset_path}/train.npy'))
    val_dataset = CustomDataset(np.load(f'{dataset_path}/validation.npy'))
    test_dataset = CustomDataset(np.load(f'{dataset_path}/test.npy'))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    model = ReturnsModel()
    model.to(device)

    loss_function = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=2, verbose=True)

    epoch = 0
    train_loss_history, validation_loss_history = [], []
    early_stopping = EarlyStopping(patience=5, verbose=True, delta=1e-4)

    try:
        for epoch in range(epochs):
            model.train()
            epoch_train_loss = 0.0
            for batch_idx, (stock, market, target) in enumerate(train_dataloader):
                stock = stock.to(device)
                market = market.to(device)
                target = target.to(device)

                optimiser.zero_grad()
                output = model(stock, market)
                loss = loss_function(output, target)
                loss.backward()
                optimiser.step()

                epoch_train_loss += loss.item()

                if batch_idx % 1000 == 0:
                    progress = 100. * (batch_idx + 1) / len(train_dataloader)
                    logging.info(
                        f"Train Epoch: {epoch} [{batch_idx + 1}/{len(train_dataloader)} batches ({progress:.2f}%)]\tLoss: {loss.item():.6f}"
                    )

            avg_train_loss = epoch_train_loss / len(train_dataloader)
            train_loss_history.append(avg_train_loss)

            # Validation
            model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for batch_idx, (stock, market, target) in enumerate(validation_dataloader):
                    stock = stock.to(device)
                    market = market.to(device)
                    target = target.to(device)

                    output = model(stock, market)
                    epoch_val_loss += loss_function(output, target).item()

            avg_val_loss = epoch_val_loss / len(validation_dataloader)
            validation_loss_history.append(avg_val_loss)
            logging.info(f"Validation: Average loss: {avg_val_loss:.6f}")

            # Step the learning rate scheduler with the validation loss
            scheduler.step(avg_val_loss)

            # Check early stopping condition
            early_stopping(avg_val_loss, model, epoch)
            if early_stopping.early_stop:
                logging.info("Early stopping triggered. Exiting training loop.")
                break

    except Exception as e:
        # Save checkpoint if an exception occurs, then re-raise
        error_checkpoint_path = os.path.join(checkpoint_dir, f"error_checkpoint_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimiser.state_dict(),
            'train_loss_history': train_loss_history,
            'validation_loss_history': validation_loss_history,
        }, error_checkpoint_path)
        logging.error(f"Error encountered. Saved checkpoint: {error_checkpoint_path}")
        raise e

    # Optionally load the best model state (if early stopping was triggered)
    if early_stopping.best_model_state is not None:
        model.load_state_dict(early_stopping.best_model_state)

    with open(f'{output_dir}/loss.json', 'w') as f:
        json.dump({'train': train_loss_history, 'validation': validation_loss_history}, f)

    final_model_path = os.path.join(output_dir, 'best_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
        'train_loss_history': train_loss_history,
        'validation_loss_history': validation_loss_history,
    }, final_model_path)
    logging.info(f"Training complete. Final model saved to: {final_model_path}")
#
#
# if __name__ == '__main__':
#     train()
