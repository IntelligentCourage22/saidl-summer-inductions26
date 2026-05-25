import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings applied directly to query/key heads."""

    def __init__(self, dim, max_seq_len=4096, base=10000):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension.")

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        cos, sin = self._build_cache(max_seq_len, device=None)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _build_cache(self, max_seq_len, device):
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.float32, device=device)
                / self.dim
            )
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        return freqs.cos()[None, None, :, :], freqs.sin()[None, None, :, :]

    def resize_max_seq_len(self, max_seq_len):
        """Extend cached rotary frequencies for longer-context evaluation."""
        if max_seq_len <= self.max_seq_len:
            return

        cos, sin = self._build_cache(max_seq_len, self.cos.device)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.max_seq_len = max_seq_len

    def forward(self, q, k):
        seq_len = q.size(-2)
        if seq_len > self.max_seq_len:
            self.resize_max_seq_len(seq_len)
        cos = self.cos[:, :, :seq_len, :].to(dtype=q.dtype, device=q.device)
        sin = self.sin[:, :, :seq_len, :].to(dtype=q.dtype, device=q.device)
        return self.apply_rotary(q, cos, sin), self.apply_rotary(k, cos, sin)

    @staticmethod
    def apply_rotary(x, cos, sin):
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x_even * cos - x_odd * sin
        out[..., 1::2] = x_odd * cos + x_even * sin
        return out
