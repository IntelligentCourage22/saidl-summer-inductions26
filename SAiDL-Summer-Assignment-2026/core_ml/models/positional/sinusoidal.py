
import torch
import torch.nn as nn
import math


class SinusoidalPE(nn.Module):
    """
    Fixed sinusoidal positional encoding from 'Attention Is All You Need'.
    Not learned — purely deterministic based on position.
    """
    def __init__(self, d_model, max_seq_len=4096, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # build the encoding matrix once and register as buffer
        # (buffer = not a parameter, but moves with .to(device))
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims
        pe = pe.unsqueeze(0)  # [1, max_seq_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, T, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
