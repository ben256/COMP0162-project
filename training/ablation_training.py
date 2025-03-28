import logging
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from model.ablation import FusionAblationModel, PredictionTypeAblationModel, MarketContextAblationModel
from data_processing.dataset import CustomDataset

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(filename)s - %(message)s')
logger = logging.getLogger(__name__)


def directional_accuracy(targets, predictions):
    """Calculate the directional accuracy of the model."""
    predictions = np.array(predictions)
    targets = np.array(targets)

    pred_sign = np.sign(predictions)
    target_sign = np.sign(targets)

    return np.mean(pred_sign == target_sign)


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for stock, market, target in dataloader:
        stock = stock.to(device)
        market = market.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        output = model(stock, market)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for stock, market, target in dataloader:
            stock = stock.to(device)
            market = market.to(device)
            target = target.to(device)

            output = model(stock, market)
            loss = criterion(output, target)

            total_loss += loss.item()
            predictions.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())

    val_loss = total_loss / len(dataloader)

    # Calculate metrics
    all_preds = np.concatenate(predictions)
    all_targets = np.concatenate(targets)
    rmse = root_mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    da = directional_accuracy(all_targets, all_preds)

    return val_loss, rmse, mae, da


def train_model(model, train_dataloader, val_dataloader, config, device, output_dir):
    model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    rmses = []
    maes = []
    das = []

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(config['epochs']):
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device)
        val_loss, rmse, mae, da = validate(model, val_dataloader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        rmses.append(rmse)
        maes.append(mae)
        das.append(da)

        scheduler.step(val_loss)

        logger.info(f"Epoch {epoch+1}/{config['epochs']}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        logger.info(f"RMSE: {rmse:.6f}, MAE: {mae:.6f}, DA: {da:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'rmse': rmse,
                'mae': mae,
                'da': da
            }, os.path.join(output_dir, "best_model.pth"))
            logger.info("Saved best model checkpoint")

    # Plot training curves
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss Curves')

    plt.subplot(2, 2, 2)
    plt.plot(rmses, label='RMSE')
    plt.legend()
    plt.title('RMSE')

    plt.subplot(2, 2, 3)
    plt.plot(maes, label='MAE')
    plt.legend()
    plt.title('MAE')

    plt.subplot(2, 2, 4)
    plt.plot(das, label='Directional Accuracy')
    plt.legend()
    plt.title('Directional Accuracy')

    plt.savefig(os.path.join(output_dir, "training_curves.png"))

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'rmses': rmses,
        'maes': maes,
        'das': das,
        'best_val_loss': best_val_loss,
        'final_rmse': rmses[-1],
        'final_mae': maes[-1],
        'final_da': das[-1]
    }


def run_ablation():
    torch.manual_seed(42)
    np.random.seed(42)

    # Configuration
    config = {
        'batch_size': 128,
        'epochs': 30,
        'lr': 0.00001,
        'stock_input_dim': 22,
        'market_input_dim': 24,
        'embed_dim': 128,
        'num_heads': 4
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load datasets
    dataset_path = '../data/datasets'
    train_dataset = CustomDataset(np.load(os.path.join(dataset_path, 'train.npy')))
    val_dataset = CustomDataset(np.load(os.path.join(dataset_path, 'validation.npy')))

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    results = {}
    output_base_dir = '../output/ablation'

    # 1. Fusion Type Ablation
    fusion_types = ['concat', 'cross-attn']
    for fusion_type in fusion_types:
        logger.info(f"Running ablation for fusion type: {fusion_type}")
        output_dir = os.path.join(output_base_dir, f"fusion_{fusion_type}")

        model = FusionAblationModel(
            fusion_type=fusion_type,
            stock_input_dim=config['stock_input_dim'],
            market_input_dim=config['market_input_dim'],
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads']
        )

        result = train_model(model, train_dataloader, val_dataloader, config, device, output_dir)
        results[f"fusion_{fusion_type}"] = result

    # 2. Prediction Type Ablation
    prediction_types = ['last', 'attn_pool']
    for pred_type in prediction_types:
        logger.info(f"Running ablation for prediction type: {pred_type}")
        output_dir = os.path.join(output_base_dir, f"prediction_{pred_type}")

        model = PredictionTypeAblationModel(
            prediction_type=pred_type,
            stock_input_dim=config['stock_input_dim'],
            market_input_dim=config['market_input_dim'],
            embed_dim=config['embed_dim']
        )

        result = train_model(model, train_dataloader, val_dataloader, config, device, output_dir)
        results[f"prediction_{pred_type}"] = result

    # 3. Market Context Ablation
    market_contexts = [True, False]
    for use_market in market_contexts:
        context_name = "with_market" if use_market else "without_market"
        logger.info(f"Running ablation for market context: {context_name}")
        output_dir = os.path.join(output_base_dir, f"context_{context_name}")

        model = MarketContextAblationModel(
            include_market_context=use_market,
            stock_input_dim=config['stock_input_dim'],
            market_input_dim=config['market_input_dim'],
            embed_dim=config['embed_dim']
        )

        result = train_model(model, train_dataloader, val_dataloader, config, device, output_dir)
        results[f"context_{context_name}"] = result

    # Summarize results
    logger.info("\n========== ABLATION STUDY RESULTS ==========")

    logger.info("\n1. Fusion Type Comparison:")
    for fusion_type in fusion_types:
        result = results[f"fusion_{fusion_type}"]
        logger.info(f"{fusion_type}: RMSE={result['final_rmse']:.6f}, MAE={result['final_mae']:.6f}, DA={result['final_da']:.6f}")

    logger.info("\n2. Prediction Type Comparison:")
    for pred_type in prediction_types:
        result = results[f"prediction_{pred_type}"]
        logger.info(f"{pred_type}: RMSE={result['final_rmse']:.6f}, MAE={result['final_mae']:.6f}, DA={result['final_da']:.6f}")

    logger.info("\n3. Market Context Comparison:")
    for use_market in market_contexts:
        context_name = "with_market" if use_market else "without_market"
        result = results[f"context_{context_name}"]
        logger.info(f"{context_name}: RMSE={result['final_rmse']:.6f}, MAE={result['final_mae']:.6f}, DA={result['final_da']:.6f}")


if __name__ == '__main__':
    run_ablation()