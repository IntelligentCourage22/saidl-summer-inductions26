import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings applied directly to query/key heads."""

    def __init__(self, dim, max_seq_len=4096, base=10000):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension.")

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)

        self.register_buffer("cos", freqs.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", freqs.sin()[None, None, :, :], persistent=False)

    def forward(self, q, k):
        seq_len = q.size(-2)
        cos = self.cos[:, :, :seq_len, :].to(dtype=q.dtype, device=q.device)
        sin = self.sin[:, :, :seq_len, :].to(dtype=q.dtype, device=q.device)
        return self._apply(q, cos, sin), self._apply(k, cos, sin)

    @staticmethod
    def _apply(x, cos, sin):
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x_even * cos - x_odd * sin
        out[..., 1::2] = x_even * sin + x_odd * cos
        return out
