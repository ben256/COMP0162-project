import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, unified_features, target_type='return'):
        """
        Args:
            unified_features (array-like): with shape [n, 11]:
            - stock_data: Data for stock features with shape [n, 5].
            - market_data : Data for market features with shape [n, 4].
            - targets: Target values (next day’s return ratio and return sign) with shape [n, 2].
        """
        features = torch.tensor(unified_features, dtype=torch.float32)

        self.stock_data = features[:, :, :5]
        self.market_data = features[:, :, 5:9]
        self.target_return = features[:, :, 9]
        self.target_sign = features[:, :, 10]

        if target_type == 'return':
            self.target = self.target_return
        elif target_type == 'sign':
            self.target = self.target_sign
        else:
            raise 'Invalid target type'

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        sample = {
            "stock_data": self.stock_data[idx],
            "market_data": self.market_data[idx],
            "target": self.target[idx]
        }
        return sample
