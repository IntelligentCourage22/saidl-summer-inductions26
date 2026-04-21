
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SlidingWindowAttention(nn.Module):
    def __init__(self, d_model, n_heads, window_size=256, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model     = d_model
        self.n_heads     = n_heads
        self.d_head      = d_model // n_heads
        self.scale       = math.sqrt(self.d_head)
        self.window_size = window_size
        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        W = min(self.window_size, T)
        Q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        idx  = torch.arange(T, device=x.device)
        row  = idx.unsqueeze(1)
        col  = idx.unsqueeze(0)
        causal_mask = col > row
        local_mask  = (row - col) >= W
        combined    = causal_mask | local_mask
        scores = scores.masked_fill(combined.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)
        out  = torch.matmul(attn, V)
        out  = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)
