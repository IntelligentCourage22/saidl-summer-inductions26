
import torch, torch.nn as nn, torch.nn.functional as F, math

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads=2, dropout=0.1):
        super().__init__()
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups   = n_heads // n_kv_heads
        self.d_head     = d_model // n_heads
        self.scale      = math.sqrt(self.d_head)
        self.q_proj   = nn.Linear(d_model, n_heads * self.d_head)
        self.k_proj   = nn.Linear(d_model, n_kv_heads * self.d_head)
        self.v_proj   = nn.Linear(d_model, n_kv_heads * self.d_head)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        Q = self.q_proj(x).view(B, T, self.n_heads,    self.d_head).transpose(1,2)
        K = self.k_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1,2)
        V = self.v_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1,2)
        K = K.unsqueeze(2).expand(B, self.n_kv_heads, self.n_groups, T, self.d_head).reshape(B, self.n_heads, T, self.d_head)
        V = V.unsqueeze(2).expand(B, self.n_kv_heads, self.n_groups, T, self.d_head).reshape(B, self.n_heads, T, self.d_head)
        scores = torch.matmul(Q, K.transpose(-2,-1)) / self.scale
        if mask is None:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out  = torch.matmul(attn, V).transpose(1,2).contiguous().view(B, T, C)
        return self.out_proj(out)
