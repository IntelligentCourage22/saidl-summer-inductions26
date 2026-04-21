import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core_ml.models.positional.alibi import AlibiBias
from core_ml.models.positional.relative import RelativePositionBias
from core_ml.models.positional.rope import RotaryEmbedding


class GroupedQueryAttention(nn.Module):
    """Grouped-query attention. Set n_kv_heads=1 for MQA."""

    def __init__(
        self,
        d_model,
        n_heads,
        n_kv_heads=2,
        dropout=0.1,
        positional_encoding="sinusoidal",
        max_seq_len=4096,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)

        self.q_proj = nn.Linear(d_model, n_heads * self.d_head)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head)
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

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.d_head)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.d_head)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.rotary is not None:
            q, k = self.rotary(q, k)

        k = (
            k.unsqueeze(2)
            .expand(batch_size, self.n_kv_heads, self.n_groups, seq_len, self.d_head)
            .reshape(batch_size, self.n_heads, seq_len, self.d_head)
        )
        v = (
            v.unsqueeze(2)
            .expand(batch_size, self.n_kv_heads, self.n_groups, seq_len, self.d_head)
            .reshape(batch_size, self.n_heads, seq_len, self.d_head)
        )

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if self.alibi is not None:
            scores = scores + self.alibi(seq_len, x.device, scores.dtype)
        if self.relative_bias is not None:
            scores = scores + self.relative_bias(seq_len, x.device)

        if mask is None:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device), diagonal=1
            ).bool()
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.out_proj(out)
