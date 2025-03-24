import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(
            self,
            embed_dim: int = 128,
            dropout: float = 0.1,
            max_len: int = 5000,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Parameters:
            x: Tensor of shape (batch_size, sequence_length, embed_dim)
        Returns:
            Tensor of the same shape as x after adding positional encodings
        """
        # [batch_size, 60, 128]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class PositionWiseFFN(nn.Module):
    def __init__(
            self,
            embed_dim: int = 128,
            hidden_dim: int = 256,
            dropout: float = 0.1
    ):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Parameters:
            x: Tensor of shape (batch_size, sequence_length, embed_dim)
        Returns:
            Tensor of shape (batch_size, sequence_length, embed_dim)
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(
            self,
            embed_dim: int = 128,
            num_heads: int = 8,
            dropout: float = 0.1,
            ff_hidden_dim: int = 256
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.multi_head_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = PositionWiseFFN(embed_dim, ff_hidden_dim, dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Parameters:
            x: Tensor of shape (batch_size, sequence_length, embed_dim)
            mask: Not really sure what this does
        Returns:
            Tensor of shape (batch_size, sequence_length, embed_dim)
        """
        x_norm = self.norm1(x)
        attn_output, _ = self.multi_head_attention(x_norm, x_norm, x_norm, key_padding_mask=mask)
        x = x + self.dropout1(attn_output)

        # Pre-LN for the feedforward sub-layer
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = x + self.dropout2(ffn_output)

        return x


class StockEncoder(nn.Module):
    def __init__(
            self,
            input_dim: int = 22,
            embed_dim: int = 128,
            num_layers: int = 3,
            num_heads: int = 8,
            dropout: float = 0.1,
            ff_hidden_dim: int = 256
    ):
        super().__init__()
        self.linear_projection = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, dropout, ff_hidden_dim)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        Parameters:
            x: Tensor of shape (batch_size, sequence_length, input_dim)
        Returns:
            Tensor of shape (batch_size, sequence_length, embed_dim)
        """
        x = self.linear_projection(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)

        return x


class MarketEncoder(nn.Module):
    def __init__(
            self,
            input_dim: int = 24,
            embed_dim: int = 128,
            num_layers: int = 3,
            num_heads: int = 8,
            ff_hidden_dim: int = 256,
            dropout: float = 0.1
    ):
        super().__init__()
        self.linear_projection = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, dropout, ff_hidden_dim)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        Parameters:
            x: Tensor of shape (batch_size, sequence_length, input_dim)
        Returns:
            Tensor of shape (batch_size, sequence_length, embed_dim)
        """
        x = self.linear_projection(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)

        return x


class Fusion(nn.Module):
    def __init__(
            self,
            fusion_type: str = 'concat',
            stock_input_dim: int = 22,
            market_input_dim: int = 24,
            embed_dim: int = 128,
            num_layers: int = 3,
            num_heads: int = 8,
            dropout: float = 0.1,
            ff_hidden_dim: int = 256
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.stock_encoder = StockEncoder(
            input_dim=stock_input_dim,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            ff_hidden_dim=ff_hidden_dim
        )
        self.market_encoder = MarketEncoder(
            input_dim=market_input_dim,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_hidden_dim=ff_hidden_dim,
            dropout=dropout
        )
        self.linear_fusion = nn.Linear(embed_dim*2, embed_dim)
        self.relu = nn.ReLU()

        if self.fusion_type == 'cross_attn':
            self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)


    def forward(self, x, y):
        """
        Parameters:
            x: Stock embeddings, tensor of shape (batch_size, sequence_length, input_dim)
            y: Market embeddings, tensor of shape (batch_size, sequence_length, input_dim)
        Returns:
            Tensor of shape (batch_size, sequence_length, embed_dim)
        """
        x = self.stock_encoder(x)
        y = self.market_encoder(y)

        if self.fusion_type == 'concat':
            out = torch.cat([x, y], -1)
            out = self.linear_fusion(out)
            out = self.relu(out)
            return out

        elif self.fusion_type == 'cross_attn':
            # Use stock embeddings as queries, market embeddings as keys/values
            attn_output, _ = self.cross_attn(query=x, key=y, value=y)
            # Apply a residual connection with stock embeddings
            out = x + attn_output
            out = self.relu(out)
            return out

        else:
            raise ValueError("Invalid fusion type")


class PredictionHead(nn.Module):
    def __init__(
            self,
            prediction_type: str = 'attn_pool',
            embed_dim: int = 128,
            dropout: float = 0.1
    ):
        super().__init__()
        self.prediction_type = prediction_type
        self.fc1 = nn.Linear(embed_dim, 1)

        if self.prediction_type == 'attn_pool':
            self.attn_vector = nn.Linear(embed_dim, 1)

    def forward(self, x):
        """
        Parameters:
            x: Tensor of shape (batch_size, sequence_length, embed_dim)
        Returns:
            Tensor of shape (batch_size, 1) containing the predicted percentage change
        """
        if self.prediction_type == 'last':
            x_last = x[:, -1, :]  # shape: (batch_size, embed_dim)
        elif self.prediction_type == 'pool':
            x_last = torch.mean(x, dim=1) # shape: (batch_size, embed_dim)
        elif self.prediction_type == 'attn_pool':
            attn_scores = self.attn_vector(x)  # (batch_size, sequence_length, 1)
            attn_weights = torch.softmax(attn_scores, dim=1)  # (batch_size, sequence_length, 1)
            x_last = torch.sum(attn_weights * x, dim=1)  # (batch_size, embed_dim)
        else:
            raise ValueError("Invalid prediction type")

        x = self.fc1(x_last)
        return x


class MCST(nn.Module):
    def __init__(
            self,
            fusion_type: str = 'cross_attn',
            prediction_type: str = 'attn_pool',
            stock_input_dim: int = 22,
            market_input_dim: int = 24,
            embed_dim: int = 128,
            num_layers: int = 3,
            num_heads: int = 8,
            dropout: float = 0.1,
            ff_hidden_dim: int = 256
    ):
        super().__init__()
        self.fusion = Fusion(
            fusion_type=fusion_type,
            stock_input_dim=stock_input_dim,
            market_input_dim=market_input_dim,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            ff_hidden_dim=ff_hidden_dim
        )
        self.prediction_head = PredictionHead(
            prediction_type=prediction_type,
            embed_dim=embed_dim,
            dropout=dropout
        )

    def forward(self, stock_data, market_data):
        """
        Parameters:
            stock_data: Tensor of shape (batch_size, sequence_length, 5)
            market_data: Tensor of shape (batch_size, sequence_length, 4)
        Returns:
            Tensor of shape (batch_size, 1) containing the predicted percentage change
        """
        x = self.fusion(stock_data, market_data)
        return self.prediction_head(x)

    def predict(self, stock_data, market_data, targets):
        """
        Parameters:
            stock_data: Tensor of shape (batch_size, sequence_length, 5)
            market_data: Tensor of shape (batch_size, sequence_length, 4)
        Returns:
            Tensor of shape (batch_size, 1) containing the predicted percentage change
        """
        x = self.fusion(stock_data, market_data)
        return self.prediction_head(x)

