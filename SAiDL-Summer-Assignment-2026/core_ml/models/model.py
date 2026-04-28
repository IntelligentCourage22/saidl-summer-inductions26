import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention.gqa import GroupedQueryAttention
from models.attention.linear_attention import LinearAttention
from models.attention.sliding_window import SlidingWindowAttention
from models.attention.standard import StandardAttention
from models.positional.sinusoidal import SinusoidalPE


def cfg_get(cfg, key, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class IdentityPE(nn.Module):
    def forward(self, x):
        return x


def build_attention(attention_type, d_model, n_heads, dropout, model_cfg):
    positional_encoding = cfg_get(model_cfg, "positional_encoding", "sinusoidal")
    max_seq_len = cfg_get(model_cfg, "max_seq_len", 4096)

    if attention_type == "standard":
        return StandardAttention(
            d_model, n_heads, dropout, positional_encoding, max_seq_len
        )
    if attention_type == "sliding_window":
        return SlidingWindowAttention(
            d_model,
            n_heads,
            cfg_get(model_cfg, "window_size", 256),
            dropout,
            positional_encoding,
            max_seq_len,
        )
    if attention_type == "linear":
        return LinearAttention(
            d_model,
            n_heads,
            dropout,
            positional_encoding=positional_encoding,
            max_seq_len=max_seq_len,
        )
    if attention_type == "gqa":
        return GroupedQueryAttention(
            d_model,
            n_heads,
            cfg_get(model_cfg, "n_kv_heads", max(1, n_heads // 4)),
            dropout,
            positional_encoding,
            max_seq_len,
        )
    if attention_type == "mqa":
        return GroupedQueryAttention(
            d_model, n_heads, 1, dropout, positional_encoding, max_seq_len
        )

    raise ValueError(f"Unknown attention type: {attention_type}")


def build_positional_encoding(pe_type, d_model, max_seq_len, dropout):
    if pe_type == "sinusoidal":
        return SinusoidalPE(d_model, max_seq_len, dropout)
    if pe_type in {"rope", "alibi", "relative", "none"}:
        return IdentityPE()

    raise ValueError(f"Unknown positional encoding: {pe_type}")


class CausalDepthwiseConv1d(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.left_padding = kernel_size - 1
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=0,
            groups=channels,
        )

    def forward(self, x):
        return self.conv(F.pad(x, (self.left_padding, 0)))


class DepthwiseConvBlock(nn.Module):
    """Lightweight causal 1D convolution branch for local token mixing."""

    def __init__(self, d_model, kernel_size=5, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            CausalDepthwiseConv1d(d_model, kernel_size),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class GatedConvFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, kernel_size=5, dropout=0.1):
        super().__init__()
        self.proj_in = nn.Linear(d_model, 2 * d_ff)
        self.depthwise = CausalDepthwiseConv1d(d_ff, kernel_size)
        self.proj_out = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate, value = self.proj_in(x).chunk(2, dim=-1)
        value = self.depthwise(value.transpose(1, 2)).transpose(1, 2)
        value = F.gelu(value) * torch.sigmoid(gate)
        return self.dropout(self.proj_out(value))


class TransformerBlock(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        d_model = cfg_get(model_cfg, "d_model")
        n_heads = cfg_get(model_cfg, "n_heads")
        d_ff = cfg_get(model_cfg, "d_ff")
        dropout = cfg_get(model_cfg, "dropout")
        attention_type = cfg_get(model_cfg, "attention_type")
        self.block_type = cfg_get(model_cfg, "block_type", "standard")

        self.attn = build_attention(attention_type, d_model, n_heads, dropout, model_cfg)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.conv_norm = None
        self.conv = None
        if self.block_type in {"conv_before", "interleaved"}:
            self.conv_norm = nn.LayerNorm(d_model)
            self.conv = DepthwiseConvBlock(
                d_model,
                cfg_get(model_cfg, "conv_kernel_size", 5),
                dropout,
            )

        if self.block_type == "gated_conv":
            self.ff = GatedConvFeedForward(
                d_model,
                d_ff,
                cfg_get(model_cfg, "conv_kernel_size", 5),
                dropout,
            )
        else:
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )

    def forward(self, x, mask=None):
        if self.block_type == "conv_before":
            x = x + self.conv(self.conv_norm(x))

        x = x + self.dropout(self.attn(self.norm1(x), mask))

        if self.block_type == "interleaved":
            x = x + self.conv(self.conv_norm(x))

        x = x + self.ff(self.norm2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        model_cfg = cfg_get(cfg, "model")
        d_model = cfg_get(model_cfg, "d_model")
        n_layers = cfg_get(model_cfg, "n_layers")
        vocab_size = cfg_get(model_cfg, "vocab_size")
        max_seq_len = cfg_get(model_cfg, "max_seq_len")
        dropout = cfg_get(model_cfg, "dropout")
        pe_type = cfg_get(model_cfg, "positional_encoding")

        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = build_positional_encoding(pe_type, d_model, max_seq_len, dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(model_cfg) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        _, seq_len = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}."
            )

        x = self.token_emb(x)
        x = self.pos_enc(x)

        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
        mask = mask.bool()

        for block in self.blocks:
            x = block(x, mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )

        return logits, loss

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
