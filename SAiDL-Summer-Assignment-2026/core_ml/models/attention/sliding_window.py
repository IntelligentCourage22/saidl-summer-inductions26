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

    def _local_relative_bias(self, query_positions, key_positions, device):
        rel_pos = key_positions - query_positions
        rel_pos = rel_pos.clamp(
            min=-(self.relative_bias.max_seq_len - 1),
            max=self.relative_bias.max_seq_len - 1,
        )
        rel_pos = rel_pos + self.relative_bias.max_seq_len - 1
        bias = self.relative_bias.bias(rel_pos.to(device))
        return bias.permute(2, 0, 1).unsqueeze(0)

    def _local_alibi_bias(self, query_positions, key_positions, dtype):
        distance = (query_positions - key_positions).clamp(min=0).to(dtype)
        slopes = self.alibi.slopes.to(device=query_positions.device, dtype=dtype)
        return -slopes * distance.view(1, 1, *distance.shape)

    @staticmethod
    def _gather_local_mask(mask, key_positions):
        seq_len, window_size = key_positions.shape
        safe_key_positions = key_positions.clamp(0, seq_len - 1).to(mask.device)

        if mask.dim() == 2:
            query_positions = torch.arange(seq_len, device=mask.device).unsqueeze(1)
            return mask[query_positions, safe_key_positions]

        if mask.dim() == 3:
            gather_idx = safe_key_positions.unsqueeze(0).expand(mask.size(0), -1, -1)
            return torch.gather(mask, dim=2, index=gather_idx)

        if mask.dim() == 4:
            gather_idx = safe_key_positions.view(1, 1, seq_len, window_size).expand(
                mask.size(0), mask.size(1), -1, -1
            )
            return torch.gather(mask, dim=3, index=gather_idx)

        raise ValueError("mask must have shape [T,T], [B,T,T], or [B,H,T,T]")

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

        pad_len = window_size - 1
        k_windows = F.pad(k, (0, 0, pad_len, 0)).unfold(2, window_size, 1)
        v_windows = F.pad(v, (0, 0, pad_len, 0)).unfold(2, window_size, 1)
        k_windows = k_windows.permute(0, 1, 2, 4, 3)
        v_windows = v_windows.permute(0, 1, 2, 4, 3)

        scores = torch.einsum("bhtd,bhtwd->bhtw", q, k_windows) / self.scale

        query_positions = torch.arange(seq_len, device=x.device).unsqueeze(1)
        window_offsets = torch.arange(window_size, device=x.device).unsqueeze(0)
        key_positions = query_positions - window_size + 1 + window_offsets
        combined_mask = key_positions < 0

        if self.alibi is not None:
            scores = scores + self._local_alibi_bias(
                query_positions, key_positions, scores.dtype
            )
        if self.relative_bias is not None:
            scores = scores + self._local_relative_bias(
                query_positions, key_positions, x.device
            )

        if mask is not None:
            local_mask = self._gather_local_mask(mask.bool(), key_positions)
            if local_mask.dim() == 2:
                local_mask = local_mask.unsqueeze(0).unsqueeze(0)
            elif local_mask.dim() == 3:
                local_mask = local_mask.unsqueeze(1)
            combined_mask = combined_mask.view(1, 1, seq_len, window_size) | local_mask
        else:
            combined_mask = combined_mask.view(1, 1, seq_len, window_size)

        scores = scores.masked_fill(combined_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        out = torch.einsum("bhtw,bhtwd->bhtd", attn, v_windows)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.out_proj(out)
