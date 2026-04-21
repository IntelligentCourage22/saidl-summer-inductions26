import torch
import torch.nn as nn
import torch.nn.functional as F

from core_ml.models.positional.rope import RotaryEmbedding


class LinearAttention(nn.Module):
    """Causal kernelized linear attention with an ELU+1 feature map."""

    def __init__(
        self,
        d_model,
        n_heads,
        dropout=0.1,
        eps=1e-6,
        positional_encoding="sinusoidal",
        max_seq_len=4096,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.eps = eps

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

    @staticmethod
    def kernel(x):
        return F.elu(x) + 1

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.rotary is not None:
            q, k = self.rotary(q, k)

        q = self.kernel(q)
        k = self.kernel(k)

        kv = torch.einsum("bhtd,bhte->bhtde", k, v)
        kv_cum = torch.cumsum(kv, dim=2)
        numerator = torch.einsum("bhtd,bhtde->bhte", q, kv_cum)

        k_cum = torch.cumsum(k, dim=2)
        denominator = torch.einsum("bhtd,bhtd->bht", q, k_cum)
        denominator = denominator.unsqueeze(-1).clamp(min=self.eps)

        out = numerator / denominator
        out = self.dropout(out)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.out_proj(out)
