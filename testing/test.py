import logging
import os
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from model.benchmarks import NullModel, NaiveModel, ARIMAModel
from model.model import ReturnsModel
from data_processing.dataset import CustomDataset

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(filename)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)


def directional_accuracy(predictions, targets):
    """
    Calculate the directional accuracy of the model.
    """
    predictions = np.array(predictions)
    targets = np.array(targets)

    pred_sign = np.sign(predictions)
    target_sign = np.sign(targets)

    return np.mean(pred_sign == target_sign)


def evalute_model(
        model,
        dataloader,
        device,
):
    predictions = []
    targets = []

    with torch.no_grad():
        for stock, market, target in dataloader:
            stock = stock.to(device)
            market = market.to(device)
            target = target.to(device)

            output = model(stock, market)

            if isinstance(output, torch.Tensor):
                output = output.cpu().numpy()

            predictions.append(output)
            targets.append(target)

    return predictions, targets


def test(
        dataset_path='../data/datasets',  # Path to your dataset folder
        checkpoint_path='../output/best_model.pth',  # Path to your best model checkpoint
        batch_size=200
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    best_model = torch.load('../output/remote_0/best_model.pth', map_location=torch.device('cpu'))

    test_dataset = CustomDataset(np.load(os.path.join(dataset_path, 'test.npy')))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    null_model = NullModel()
    naive_model = NaiveModel()
    arima_model = ARIMAModel()
    transformers_model = ReturnsModel(
        fusion_type = 'cross_attn',
        prediction_type = 'attn_pool',
        stock_input_dim = 5,
        market_input_dim = 4,
        embed_dim = 128,
        num_layers = 3,
        num_heads = 4,
        dropout = 0.1,
        ff_hidden_dim = 256
    )
    transformers_model.load_state_dict(best_model['model_state_dict'])
    transformers_model.eval()

    # models = [null_model, naive_model, arima_model, transformers_model]
    # model_names = ['Null', 'Naive', 'ARIMA', 'Transformers']
    models = [naive_model]
    model_names = ['Naive']


    for model, model_name in zip(models, model_names):
        predictions, targets = evalute_model(model, test_dataloader, device)
        rmse = root_mean_squared_error(np.concatenate(targets), np.concatenate(predictions))
        mae = mean_absolute_error(np.concatenate(targets), np.concatenate(predictions))
        mape = mean_absolute_percentage_error(np.concatenate(targets), np.concatenate(predictions))
        da = directional_accuracy(np.concatenate(predictions), np.concatenate(targets))

        logging.info('---')
        logging.info(f"{model_name} model:")
        logging.info(f"RMSE: {rmse:.6f}")
        logging.info(f"MAE: {mae:.6f}")
        logging.info(f"MAPE: {mape:.6f}")
        logging.info(f"Directional accuracy: {da:.6f}")


if __name__ == '__main__':
    test()
