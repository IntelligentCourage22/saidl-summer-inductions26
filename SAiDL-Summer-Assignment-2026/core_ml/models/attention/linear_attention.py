
import torch, torch.nn as nn, torch.nn.functional as F

class LinearAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, eps=1e-6):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.eps     = eps
        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout  = nn.Dropout(dropout)

    def kernel(self, x):
        return F.elu(x) + 1

    def forward(self, x, mask=None):
        B, T, C = x.shape
        Q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1,2)
        K = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1,2)
        V = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1,2)
        Q, K = self.kernel(Q), self.kernel(K)
        KV     = torch.einsum("bhti,bhtj->bhtij", K, V)
        KV_cum = torch.cumsum(KV, dim=2)
        num    = torch.einsum("bhti,bhtij->bhtj", Q, KV_cum)
        K_cum  = torch.cumsum(K, dim=2)
        den    = torch.einsum("bhti,bhti->bht", Q, K_cum).unsqueeze(-1).clamp(min=self.eps)
        out    = self.dropout(num / den)
        out    = out.transpose(1,2).contiguous().view(B, T, C)
        return self.out_proj(out)
