import json
import copy
import os

import numpy as np
import torch
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from torch import nn, optim

from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from data_processing.dataset import CustomDataset
from model.benchmarks import LSTMModel
from training.utils import create_training_folder, setup_logging


def directional_accuracy(targets, predictions):
    """
    Calculate the directional accuracy of the model.
    """
    predictions = np.array(predictions)
    targets = np.array(targets)

    pred_sign = np.sign(predictions)
    target_sign = np.sign(targets)

    return np.mean(pred_sign == target_sign)


def evaluate_model(
        model,
        dataloader,
        device,
):
    predictions = []
    targets = []

    model.eval()

    with torch.no_grad():
        for stock, market, target in dataloader:
            stock = stock.to(device)
            market = market.to(device)
            target = target.to(device)

            output = model(stock, market)

            if isinstance(output, torch.Tensor):
                output = output.cpu().numpy()

            if isinstance(target, torch.Tensor):
                target = target.cpu().numpy()

            predictions.append(output)
            targets.append(target)

    return predictions, targets


class EarlyStopping:
    def __init__(
            self,
            patience: int = 5,
            delta: float = 0.0,
            offset: int = 5,
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
        batch_size: int = 64,
        learning_rate: float = 0.001,
        num_training_epochs: int = 50,
        early_stopping_patience: int = 5,
        early_stopping_delta: float = 0.0,
        early_stopping_offset: int = 5,
        shuffle_train_data: bool = True,
        weight_decay: float = 0.01,
        dropout: float = 0.1,
        num_layers: int = 2,
        embed_dim: int = 128,
        dataset_path: str = '../data/datasets',
        output_dir: str = '../output/lstm_training',
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

    model = LSTMModel(
        stock_input_dim=22,
        embed_dim=embed_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    model.to(device)

    criterion = nn.MSELoss()
    optimiser = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.3, patience=5, threshold=5e-5, verbose=True)

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
    logger.info(f"Number of training epochs: {num_training_epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Early stopping: Patience: {early_stopping_patience}, Delta: {early_stopping_delta}, Offset: {early_stopping_offset}")
    logger.info(f"Shuffle training data: {shuffle_train_data}")
    logger.info(f"Weight decay: {weight_decay}")
    logger.info(f"Dropout: {dropout}")
    logger.info(f"Number of transformer layers: {num_layers}")
    logger.info(f"Embedding dimension: {embed_dim}")
    logger.info(f"Dataset path: {dataset_path}")
    logger.info(f"Output directory: {output_dir}")

    try:
        for epoch in range(num_training_epochs):
            model.train()
            epoch_train_loss = 0.0
            for batch_idx, (stock, market, target) in enumerate(train_dataloader):
                stock = stock.to(device)
                target = target.to(device)

                optimiser.zero_grad()
                output = model(stock)
                loss = criterion(output, target)

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
                    epoch_val_loss += criterion(output, target).item()

            avg_val_loss = epoch_val_loss / len(validation_dataloader)
            validation_loss_history.append(avg_val_loss)
            logger.info(f"Validation: Average loss: {avg_val_loss:.8f}")

            # Step the learning rate scheduler with the validation loss
            scheduler.step(avg_val_loss)

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

    # run on test set
    try:
        test_dataset = CustomDataset(np.load(f'{dataset_path}/test.npy'))
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
        predictions, targets = evaluate_model(model, test_dataloader, device)
        rmse = root_mean_squared_error(np.concatenate(targets), np.concatenate(predictions))
        mae = mean_absolute_error(np.concatenate(targets), np.concatenate(predictions))
        mape = mean_absolute_percentage_error(np.concatenate(targets), np.concatenate(predictions))
        da = directional_accuracy(np.concatenate(targets), np.concatenate(predictions))

        logger.info(f"Transformers model test metrics:")
        logger.info(f"RMSE: {rmse:.6f}")
        logger.info(f"MAE: {mae:.6f}")
        logger.info(f"MAPE: {mape:.6f}")
        logger.info(f"Directional accuracy: {da:.6f}")
    except Exception as e:
        logger.error('Error encountered while evaluating on test set')
        logger.error(e)

    final_model_path = os.path.join(output_dir, 'best_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
        'train_loss_history': train_loss_history,
        'validation_loss_history': validation_loss_history,
    }, final_model_path)
    logger.info(f"Training complete. Final model saved to: {final_model_path}")


# if __name__ == '__main__':
#     train()
