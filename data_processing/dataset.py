import logging

import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, unified_features, target_type='return'):
        """
        Args:
            unified_features: (batch_size, sequence_length = 60, n_features = 48)
            - stock_data: stock_features = 22, unified_features[:, :, :22]
            - market_data: market_features = 23, unified_features[:, :, 22:46]
            - target_return: unified features[:, -1, 46]
            - target_sign: unified_features[:, -1, 47]
        """
        features = torch.tensor(unified_features, dtype=torch.float32)

        assert features.shape[2] == 51, "Expected 48 features in the last dimension"
        assert torch.all((features[:, -1, 50] == 1.0) | (features[:, -1, 50] == 0.0)), "Target sign values must be either 0 or 1"

        self.stock_features = features[:, :, :25]
        self.market_features = features[:, :, 25:49]
        self.target_return = features[:, -1, 49].unsqueeze(1)
        self.target_sign = features[:, -1, 50].unsqueeze(1)
        self.feature_list = ['z_open_stock', 'z_high_stock', 'z_low_stock',
            'z_returns_stock', 'z_volume_stock', 'z_returns_ma_5_stock', 'z_returns_std_5_stock','z_volume_ma_5_stock',
            'z_volume_std_5_stock', 'z_returns_ma_10_stock', 'z_returns_std_10_stock', 'z_volume_ma_10_stock',
            'z_volume_std_10_stock', 'z_returns_ma_20_stock', 'z_returns_std_20_stock', 'z_volume_ma_20_stock',
            'z_volume_std_20_stock', 'z_returns_ma_30_stock', 'z_returns_std_30_stock', 'z_volume_ma_30_stock',
            'z_volume_std_30_stock', 'z_returns_ma_60_stock', 'z_returns_std_60_stock', 'z_volume_ma_60_stock',
            'z_volume_std_60_stock', 'z_returns_market', 'z_volume_market', 'z_vix_market', 'z_cor1m_market',
            'z_returns_ma_5_market', 'z_returns_std_5_market', 'z_volume_ma_5_market', 'z_volume_std_5_market',
            'z_returns_ma_10_market', 'z_returns_std_10_market', 'z_volume_ma_10_market', 'z_volume_std_10_market',
            'z_returns_ma_20_market', 'z_returns_std_20_market', 'z_volume_ma_20_market', 'z_volume_std_20_market',
            'z_returns_ma_30_market', 'z_returns_std_30_market', 'z_volume_ma_30_market', 'z_volume_std_30_market',
            'z_returns_ma_60_market', 'z_returns_std_60_market', 'z_volume_ma_60_market', 'z_volume_std_60_market',
            'target', 'target_sign'
        ]

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

    def print_info(self):
        for index, feature in enumerate(self.feature_list):
            split_i = feature.split('_')
            if feature == 'target':
                logging.info(f'{index}: Target - target')
            elif feature == 'target_sign':
                logging.info(f'{index}: Target - target_sign')
            elif split_i[-1] == 'stock':
                logging.info(f'{index}: Stock - {"_".join(split_i[1:-1])}')
            elif split_i[-1] == 'market':
                logging.info(f'{index}: Market - {"_".join(split_i[1:-1])}')
            else:
                logging.info('what')


class CustomDatasetLegacy(Dataset):
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
