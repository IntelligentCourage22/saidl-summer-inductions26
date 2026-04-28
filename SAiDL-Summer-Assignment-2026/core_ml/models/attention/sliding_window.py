import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.positional.alibi import AlibiBias
from models.positional.relative import RelativePositionBias
from models.positional.rope import RotaryEmbedding


class SlidingWindowAttention(nn.Module):
    """Causal local attention over a fixed-size window."""

    def __init__(
        self,
        d_model,
        n_heads,
        window_size=256,
        dropout=0.1,
        positional_encoding="sinusoidal",
        max_seq_len=4096,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)
        self.window_size = window_size

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.rotary = (
            RotaryEmbedding(self.d_head, max_seq_len)
            if positional_encoding == "rope"
            else None
        )
        self.alibi = AlibiBias(n_heads) if positional_encoding == "alibi" else None
        self.relative_bias = (
            RelativePositionBias(n_heads, max_seq_len)
            if positional_encoding == "relative"
            else None
        )

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        window_size = min(self.window_size, seq_len)

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.rotary is not None:
            q, k = self.rotary(q, k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if self.alibi is not None:
            scores = scores + self.alibi(seq_len, x.device, scores.dtype)
        if self.relative_bias is not None:
            scores = scores + self.relative_bias(seq_len, x.device)

        idx = torch.arange(seq_len, device=x.device)
        row = idx.unsqueeze(1)
        col = idx.unsqueeze(0)
        causal_mask = col > row
        local_mask = (row - col) >= window_size
        combined_mask = causal_mask | local_mask
        if mask is not None:
            combined_mask = combined_mask | mask

        scores = scores.masked_fill(
            combined_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.out_proj(out)
