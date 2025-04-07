from torch import nn
import torch


# Simple Components
class SimpleEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        return self.proj(x)


class SimpleFusion(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fusion = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, stock_emb, market_emb):
        fused = torch.cat([stock_emb, market_emb], dim=-1)
        return self.fusion(fused)


# Fusion Type
class FusionAblationModel(nn.Module):
    def __init__(self, fusion_type='concat', stock_input_dim=22, market_input_dim=24, embed_dim=128, num_heads=4):
        super().__init__()
        self.stock_encoder = SimpleEncoder(stock_input_dim, embed_dim)
        self.market_encoder = SimpleEncoder(market_input_dim, embed_dim)
        self.fusion_type = fusion_type

        if fusion_type == 'concat':
            # For concatenation, we simply combine and project
            self.fusion = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU()
            )
        elif fusion_type == 'cross-attn':
            # For cross-attention, we use a simple MultiheadAttention layer
            self.fusion = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        else:
            raise ValueError("Invalid fusion type")

        # A simple prediction head
        self.prediction_head = nn.Linear(embed_dim, 1)

    def forward(self, stock_data, market_data):
        stock_emb = self.stock_encoder(stock_data)  # shape: (batch_size, seq_len, embed_dim)
        market_emb = self.market_encoder(market_data)  # shape: (batch_size, seq_len, embed_dim)

        if self.fusion_type == 'concat':
            fused = torch.cat([stock_emb, market_emb], dim=-1)
            fused = self.fusion(fused)
        else:
            fused, _ = self.fusion(stock_emb, market_emb, market_emb)

        fused_last = fused[:, -1, :]
        return self.prediction_head(fused_last)


# Prediction Type
class PredictionTypeAblationModel(nn.Module):
    def __init__(
            self,
            prediction_type: str = 'attn_pool',
            stock_input_dim: int = 22,
            market_input_dim: int = 24,
            embed_dim: int = 128,
    ):
        super().__init__()
        self.prediction_type = prediction_type
        self.stock_encoder = SimpleEncoder(stock_input_dim, embed_dim)
        self.market_encoder = SimpleEncoder(market_input_dim, embed_dim)
        self.fusion = SimpleFusion(embed_dim)
        self.prediction_head = nn.Linear(embed_dim, 1)

        if prediction_type == 'attn_pool':
            self.attn_vector = nn.Linear(embed_dim, 1)

    def forward(self, stock_data, market_data):

        stock_emb = self.stock_encoder(stock_data)
        market_emb = self.market_encoder(market_data)
        emb = self.fusion(stock_emb, market_emb)

        if self.prediction_type == 'last':
            emb_last = emb[:, -1, :]  # shape: (batch_size, embed_dim)

        elif self.prediction_type == 'attn_pool':
            attn_scores = self.attn_vector(emb)  # (batch_size, sequence_length, 1)
            attn_weights = torch.softmax(attn_scores, dim=1)  # (batch_size, sequence_length, 1)
            emb_last = torch.sum(attn_weights * emb, dim=1)  # (batch_size, embed_dim)

        else:
            raise ValueError("Invalid prediction type")

        return self.prediction_head(emb_last)


# Market Context
class MarketContextAblationModel(nn.Module):
    def __init__(
            self,
            include_market_context: bool = True,
            stock_input_dim: int = 22,
            market_input_dim: int = 24,
            embed_dim: int = 128
    ):
        super().__init__()
        self.include_market_context = include_market_context

        self.stock_encoder = SimpleEncoder(stock_input_dim, embed_dim)
        if include_market_context:
            self.market_encoder = SimpleEncoder(market_input_dim, embed_dim)
            self.fusion = SimpleFusion(embed_dim)

        self.prediction_head = nn.Linear(embed_dim, 1)

    def forward(self, stock_data, market_data):
        stock_emb = self.stock_encoder(stock_data)

        if self.include_market_context:
            market_emb = self.market_encoder(market_data)
            output = self.fusion(stock_emb, market_emb)
        else:
            output = stock_emb

        output_last = output[:, -1, :]

        return self.prediction_head(output_last)
