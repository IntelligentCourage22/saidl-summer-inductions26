import math

import torch
import torch.nn as nn


class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding from Attention Is All You Need."""

    def __init__(self, d_model, max_seq_len=4096, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout)

        pe = self._build_pe(max_seq_len, d_model)
        self.register_buffer("pe", pe.unsqueeze(0))

    @staticmethod
    def _build_pe(max_seq_len, d_model, device=None):
        pe = torch.zeros(max_seq_len, d_model, device=device)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        if device is not None:
            position = position.to(device)
            div_term = div_term.to(device)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def resize_max_seq_len(self, max_seq_len):
        """Extend the deterministic cache for longer-context evaluation."""
        if max_seq_len <= self.max_seq_len:
            return

        pe = self._build_pe(max_seq_len, self.d_model, self.pe.device)
        pe = pe.to(dtype=self.pe.dtype)
        self.pe = pe.unsqueeze(0)
        self.max_seq_len = max_seq_len

    def forward(self, x):
        if x.size(1) > self.max_seq_len:
            self.resize_max_seq_len(x.size(1))
        return self.dropout(x + self.pe[:, : x.size(1), :])
