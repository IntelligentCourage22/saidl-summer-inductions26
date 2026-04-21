
import torch
import torch.nn as nn
import sys
import os

# make sure local modules are importable
sys.path.append(
    os.path.join(os.path.dirname(__file__), '..', '..')
)

from models.attention.standard import StandardAttention
from models.positional.sinusoidal import SinusoidalPE


# ─────────────────────────────────────────────
# Attention factory — add new types here later
# ─────────────────────────────────────────────
def build_attention(attention_type, d_model, n_heads, dropout):
    if attention_type == "standard":
        return StandardAttention(d_model, n_heads, dropout)
    # Part 2 — we'll add sliding_window, sparse, linear, mqa, gqa here
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")


# ─────────────────────────────────────────────
# Positional encoding factory
# ─────────────────────────────────────────────
def build_positional_encoding(pe_type, d_model, max_seq_len, dropout):
    if pe_type == "sinusoidal":
        return SinusoidalPE(d_model, max_seq_len, dropout)
    # Part 3 — we'll add rope, alibi, relative here
    else:
        raise ValueError(f"Unknown positional encoding: {pe_type}")


# ─────────────────────────────────────────────
# Single Transformer Block
# ─────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, attention_type):
        super().__init__()

        self.attn    = build_attention(attention_type, d_model, n_heads, dropout)
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        # pre-norm architecture (more stable than post-norm)
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.ff(self.norm2(x))
        return x


# ─────────────────────────────────────────────
# Full Transformer Language Model
# ─────────────────────────────────────────────
class TransformerLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        d_model  = cfg.model.d_model
        n_heads  = cfg.model.n_heads
        n_layers = cfg.model.n_layers
        d_ff     = cfg.model.d_ff
        dropout  = cfg.model.dropout
        vocab    = cfg.model.vocab_size
        max_len  = cfg.model.max_seq_len
        attn_t   = cfg.model.attention_type
        pe_type  = cfg.model.positional_encoding

        # token embedding
        self.token_emb = nn.Embedding(vocab, d_model)

        # positional encoding
        self.pos_enc = build_positional_encoding(pe_type, d_model, max_len, dropout)

        # stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, attn_t)
            for _ in range(n_layers)
        ])

        self.norm     = nn.LayerNorm(d_model)
        self.lm_head  = nn.Linear(d_model, vocab, bias=False)

        # weight tying — share token embedding and lm_head weights
        # this is standard practice, saves ~25M params
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        # GPT-2 style initialization
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        B, T = x.shape

        # embed tokens then add positional encoding
        x = self.token_emb(x)   # [B, T, d_model]
        x = self.pos_enc(x)     # [B, T, d_model]

        # causal mask — built once, shared across all blocks
        mask = torch.triu(
            torch.ones(T, T, device=x.device), diagonal=1
        ).bool()

        for block in self.blocks:
            x = block(x, mask)

        x = self.norm(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]

        # compute loss if targets provided
        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
