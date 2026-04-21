
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class StandardAttention(nn.Module):
    """
    Vanilla multi-head causal self-attention from 'Attention Is All You Need'.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.scale    = math.sqrt(self.d_head)

        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape   # batch, seq_len, d_model

        # project and split into heads
        Q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        # Q, K, V shape: [B, n_heads, T, d_head]

        # scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # scores shape: [B, n_heads, T, T]

        # causal mask — upper triangle is -inf so future tokens are invisible
        if mask is None:
            mask = torch.triu(
                torch.ones(T, T, device=x.device), diagonal=1
            ).bool()
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # weighted sum of values
        out = torch.matmul(attn, V)                          # [B, n_heads, T, d_head]
        out = out.transpose(1, 2).contiguous().view(B, T, C) # [B, T, d_model]
        return self.out_proj(out)
