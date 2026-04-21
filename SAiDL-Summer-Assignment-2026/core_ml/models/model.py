
import torch, torch.nn as nn, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from models.attention.standard        import StandardAttention
from models.attention.sliding_window  import SlidingWindowAttention
from models.attention.gqa             import GroupedQueryAttention
from models.attention.linear_attention import LinearAttention
from models.positional.sinusoidal     import SinusoidalPE

def build_attention(t, d, h, drop):
    if t == "standard":       return StandardAttention(d, h, drop)
    if t == "sliding_window": return SlidingWindowAttention(d, h, 256, drop)
    if t == "gqa":            return GroupedQueryAttention(d, h, 2, drop)
    if t == "mqa":            return GroupedQueryAttention(d, h, 1, drop)
    if t == "linear":         return LinearAttention(d, h, drop)
    raise ValueError(f"Unknown attention: {t}")

def build_pe(t, d, m, drop):
    if t == "sinusoidal": return SinusoidalPE(d, m, drop)
    raise ValueError(f"Unknown PE: {t}")

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, attn_type):
        super().__init__()
        self.attn    = build_attention(attn_type, d_model, n_heads, dropout)
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.ff(self.norm2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.model.d_model
        self.token_emb = nn.Embedding(cfg.model.vocab_size, d)
        self.pos_enc   = build_pe(cfg.model.positional_encoding, d, cfg.model.max_seq_len, cfg.model.dropout)
        self.blocks    = nn.ModuleList([
            TransformerBlock(d, cfg.model.n_heads, cfg.model.d_ff, cfg.model.dropout, cfg.model.attention_type)
            for _ in range(cfg.model.n_layers)
        ])
        self.norm    = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, cfg.model.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, x, targets=None):
        B, T = x.shape
        x = self.pos_enc(self.token_emb(x))
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        for block in self.blocks:
            x = block(x, mask)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
