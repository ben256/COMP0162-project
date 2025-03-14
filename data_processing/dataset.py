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

        self.stock_features = features[:, :, :5]
        self.market_features = features[:, :, 5:9]
        self.target_return = features[:, -1, 9].unsqueeze(1)
        self.target_sign = features[:, -1, 10].unsqueeze(1)

        if target_type == 'return':
            self.target = self.target_return
        elif target_type == 'sign':
            self.target = self.target_sign
        else:
            raise 'Invalid target type'

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.stock_features[idx], self.market_features[idx], self.target[idx],
