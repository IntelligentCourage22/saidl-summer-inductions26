from types import SimpleNamespace

import pytest
import torch

from core_ml.models.model import TransformerLM


def make_cfg(attention_type="standard", positional_encoding="sinusoidal", block_type="standard"):
    return SimpleNamespace(
        model=SimpleNamespace(
            vocab_size=128,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            n_layers=2,
            d_ff=64,
            dropout=0.0,
            max_seq_len=32,
            window_size=8,
            conv_kernel_size=3,
            attention_type=attention_type,
            positional_encoding=positional_encoding,
            block_type=block_type,
        )
    )


@pytest.mark.parametrize(
    "attention_type",
    ["standard", "sliding_window", "linear", "mqa", "gqa"],
)
def test_attention_variants_forward(attention_type):
    model = TransformerLM(make_cfg(attention_type=attention_type))
    x = torch.randint(0, 128, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, 128)
    assert loss.item() > 0


@pytest.mark.parametrize(
    "positional_encoding",
    ["sinusoidal", "rope", "alibi", "relative", "none"],
)
def test_positional_variants_forward(positional_encoding):
    model = TransformerLM(make_cfg(positional_encoding=positional_encoding))
    x = torch.randint(0, 128, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, 128)
    assert loss.item() > 0


@pytest.mark.parametrize(
    "block_type",
    ["standard", "conv_before", "interleaved", "gated_conv"],
)
def test_block_variants_forward(block_type):
    model = TransformerLM(make_cfg(block_type=block_type))
    x = torch.randint(0, 128, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, 128)
    assert loss.item() > 0
