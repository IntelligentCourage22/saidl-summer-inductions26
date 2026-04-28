from types import SimpleNamespace

import pytest
import torch

from models.model import TransformerLM


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


# ---------------------------------------------------------------------------
# Existing shape / smoke tests (unchanged)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# P5 — Causal masking test
# Ensures logits at position i are independent of tokens at positions j > i.
# ---------------------------------------------------------------------------

def test_causal_masking():
    model = TransformerLM(make_cfg())
    model.eval()
    x = torch.randint(0, 128, (1, 16))
    x_corrupted = x.clone()
    x_corrupted[0, -1] = (x[0, -1] + 1) % 128
    with torch.no_grad():
        logits_orig, _ = model(x)
        logits_corrupt, _ = model(x_corrupted)
    # Changing the last token must NOT affect logits at position 0
    assert torch.allclose(logits_orig[0, 0], logits_corrupt[0, 0], atol=1e-5), (
        "Causal masking violated: logits at position 0 changed when the last token was modified."
    )


# ---------------------------------------------------------------------------
# P6 — RoPE norm preservation test
# RoPE applies an orthogonal rotation, which must preserve vector norms.
# ---------------------------------------------------------------------------

def test_rope_norm_preservation():
    from models.positional.rope import RotaryEmbedding

    rope = RotaryEmbedding(dim=8, max_seq_len=16)
    q = torch.randn(1, 1, 4, 8)
    k = torch.randn(1, 1, 4, 8)
    q_rot, k_rot = rope(q, k)
    assert torch.allclose(q_rot.norm(dim=-1), q.norm(dim=-1), atol=1e-5), (
        "RoPE rotation did not preserve query vector norms."
    )
    assert torch.allclose(k_rot.norm(dim=-1), k.norm(dim=-1), atol=1e-5), (
        "RoPE rotation did not preserve key vector norms."
    )


# ---------------------------------------------------------------------------
# P7 — Gradient flow tests
# Ensures no dead paths exist in any variant (catches detached tensors,
# zero-initialised layers that do not receive gradients, etc.)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "attention_type",
    ["standard", "sliding_window", "linear", "gqa", "mqa"],
)
def test_gradients_flow(attention_type):
    model = TransformerLM(make_cfg(attention_type=attention_type))
    x = torch.randint(0, 128, (2, 16))
    _, loss = model(x, x)
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"


# ---------------------------------------------------------------------------
# P8a — max_seq_len guard test
# Verifies that TransformerLM.forward() raises ValueError for inputs
# exceeding max_seq_len.
# ---------------------------------------------------------------------------

def test_exceeds_max_seq_len_raises():
    model = TransformerLM(make_cfg())  # max_seq_len=32
    x = torch.randint(0, 128, (1, 33))
    with pytest.raises(ValueError, match="max_seq_len"):
        model(x)


# ---------------------------------------------------------------------------
# P8b — GQA invalid head configuration test
# Verifies that GQA rejects n_heads % n_kv_heads != 0.
# ---------------------------------------------------------------------------

def test_gqa_invalid_heads_raises():
    cfg = make_cfg(attention_type="gqa")
    cfg.model.n_heads = 4
    cfg.model.n_kv_heads = 3  # 4 % 3 != 0
    with pytest.raises(AssertionError):
        TransformerLM(cfg)
