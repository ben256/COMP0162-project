import json
import copy
import math
import os

import numpy as np
import torch
from torch import nn, optim

from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from data_processing.dataset import CustomDataset
from model.model import ReturnsModel
from training.utils import create_training_folder, setup_logging


def get_cosine_schedule_with_warmup(optimizer, num_warmup_epochs, num_training_epochs):
    """
    Creates a schedule with a learning rate that increases linearly during the warmup period,
    then decays following a cosine function.

    Args:
        optimizer: The optimizer for which to schedule the learning rate.
        num_warmup_epochs: The number of epochs to linearly increase the LR.
        num_training_epochs: Total number of epochs for training.

    Returns:
        A LambdaLR scheduler.
    """
    def lr_lambda(current_epoch):
        if current_epoch < num_warmup_epochs:
            # Linear warmup: current_epoch / num_warmup_epochs
            return float(current_epoch) / float(max(1, num_warmup_epochs))
        else:
            # Cosine annealing: progress runs from 0 to 1 over the remaining epochs.
            progress = float(current_epoch - num_warmup_epochs) / float(max(1, num_training_epochs - num_warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


class EarlyStopping:
    def __init__(
            self,
            patience: int = 5,
            delta: float = 0.0,
            offset: int = 20,
            verbose: bool = False,
            logger = None,
    ):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in validation loss to qualify as an improvement.
        """
        self.patience = patience
        self.delta = delta
        self.offset = offset
        self.verbose = verbose
        self.logger = logger
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model, epoch=None):
        # First epoch
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            if self.verbose:
                self.logger.info(f"Initial validation loss: {val_loss:.6f}. Saving model.")
        elif val_loss < self.best_loss - self.delta:
            # Improvement has been made
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            if self.verbose:
                self.logger.info(f"Validation loss improved to {val_loss:.6f}. Saving model.")
        else:
            if epoch >= self.offset:
                self.counter += 1
                if self.verbose:
                    self.logger.info(f"No improvement in validation loss. Counter: {self.counter}/{self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True


def train(
        batch_size: int = 128,
        learning_rate: float = 1e-5,
        num_training_epochs: int = 100,
        num_warmup_epochs: int = 5,
        fusion_type: str = 'cross_attn',
        early_stopping_patience: int = 5,
        early_stopping_delta: float = 1e-4,
        early_stopping_offset: int = 20,
        shuffle_train_data: bool = True,
        dropout: float = 0.1,
        num_layers: int = 2,
        num_heads: int = 4,
        prediction_type: str = 'attn_pool',
        dataset_path: str = '../data/datasets',
        output_dir: str = '../output'
):
    output_dir = create_training_folder(output_dir)
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    logger = setup_logging(output_dir)
    
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    train_dataset = CustomDataset(np.load(f'{dataset_path}/train.npy'))
    val_dataset = CustomDataset(np.load(f'{dataset_path}/validation.npy'))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train_data)
    validation_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    model = ReturnsModel(
        fusion_type=fusion_type,
        prediction_type=prediction_type,
        stock_input_dim=5,
        market_input_dim=4,
        embed_dim=128,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        ff_hidden_dim=256
    )
    model.to(device)

    loss_function = nn.MSELoss()
    optimiser = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_cosine_schedule_with_warmup(optimiser, num_warmup_epochs, num_training_epochs)

    epoch = 0
    train_loss_history, validation_loss_history = [], []
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        delta=early_stopping_delta,
        offset=early_stopping_offset,
        verbose=True,
        logger=logger,
    )

    logger.info("Starting training loop")
    logger.info(f"Training for {num_training_epochs} epochs")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Early stopping: Patience: {early_stopping_patience}, Delta: {early_stopping_delta}, Offset: {early_stopping_offset}")
    logger.info(f"Shuffle training data: {shuffle_train_data}")
    logger.info(f"Dropout: {dropout}")
    logger.info(f"Number of transformer layers: {num_layers}")
    logger.info(f"Number of attention heads: {num_heads}")
    logger.info(f"Prediction type: {prediction_type}")
    logger.info(f"Fusion type: {fusion_type}")
    logger.info(f"Dataset path: {dataset_path}")
    logger.info(f"Output directory: {output_dir}")

    try:
        for epoch in range(num_training_epochs):
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
                    logger.info(
                        f"Train Epoch: {epoch} [{batch_idx + 1}/{len(train_dataloader)} batches ({progress:.2f}%)]\tLoss: {loss.item():.8f}"
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
            logger.info(f"Validation: Average loss: {avg_val_loss:.8f}")

            # Step the learning rate scheduler with the validation loss
            scheduler.step()

            with open(f'{output_dir}/loss.json', 'w') as f:
                json.dump({'train': train_loss_history, 'validation': validation_loss_history}, f)

            # Check early stopping condition
            early_stopping(avg_val_loss, model, epoch)
            if early_stopping.early_stop:
                logger.info("Early stopping triggered. Exiting training loop.")
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
        logger.error(f"Error encountered. Saved checkpoint: {error_checkpoint_path}")
        raise e

    # Optionally load the best model state (if early stopping was triggered)
    if early_stopping.best_model_state is not None:
        model.load_state_dict(early_stopping.best_model_state)

    final_model_path = os.path.join(output_dir, 'best_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
        'train_loss_history': train_loss_history,
        'validation_loss_history': validation_loss_history,
    }, final_model_path)
    logger.info(f"Training complete. Final model saved to: {final_model_path}")


if __name__ == '__main__':
    train()
