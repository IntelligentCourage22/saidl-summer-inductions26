from types import SimpleNamespace
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn.functional as F

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


def test_linear_attention_rejects_incompatible_positional_encodings():
    for positional_encoding in ["rope", "alibi", "relative"]:
        cfg = make_cfg(
            attention_type="linear",
            positional_encoding=positional_encoding,
        )
        with pytest.raises(NotImplementedError):
            TransformerLM(cfg)


def test_position_resize_keeps_buffers_registered():
    model = TransformerLM(make_cfg(positional_encoding="rope"))
    model.resize_position_buffers(64)
    rotary = model.blocks[0].attn.rotary
    assert "cos" in rotary._buffers
    assert "sin" in rotary._buffers
    assert rotary.cos.size(-2) == 64

    model = TransformerLM(make_cfg(positional_encoding="sinusoidal"))
    model.resize_position_buffers(64)
    assert "pe" in model.pos_enc._buffers
    assert model.pos_enc.pe.size(1) == 64


def test_alibi_slopes_match_paper_formula():
    from models.positional.alibi import AlibiBias

    slopes = AlibiBias._get_slopes(8)
    expected = [2 ** (-(i + 1)) for i in range(8)]
    assert slopes == pytest.approx(expected)


def _full_matrix_sliding_window_reference(attn, x):
    batch_size, seq_len, d_model = x.shape
    window_size = min(attn.window_size, seq_len)

    q = attn.q_proj(x).view(batch_size, seq_len, attn.n_heads, attn.d_head)
    k = attn.k_proj(x).view(batch_size, seq_len, attn.n_heads, attn.d_head)
    v = attn.v_proj(x).view(batch_size, seq_len, attn.n_heads, attn.d_head)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    if attn.rotary is not None:
        q, k = attn.rotary(q, k)

    scores = torch.matmul(q, k.transpose(-2, -1)) / attn.scale
    if attn.alibi is not None:
        scores = scores + attn.alibi(seq_len, x.device, scores.dtype)
    if attn.relative_bias is not None:
        scores = scores + attn.relative_bias(seq_len, x.device)

    idx = torch.arange(seq_len, device=x.device)
    row = idx.unsqueeze(1)
    col = idx.unsqueeze(0)
    mask = (col > row) | ((row - col) >= window_size)
    scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
    out = torch.matmul(attn_weights, v)
    out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
    return attn.out_proj(out)


@pytest.mark.parametrize(
    "positional_encoding",
    ["sinusoidal", "rope", "alibi", "relative"],
)
def test_sliding_window_matches_full_matrix_reference(positional_encoding):
    from models.attention.sliding_window import SlidingWindowAttention

    torch.manual_seed(0)
    attn = SlidingWindowAttention(
        d_model=32,
        n_heads=4,
        window_size=3,
        dropout=0.0,
        positional_encoding=positional_encoding,
        max_seq_len=16,
    )
    attn.eval()

    x = torch.randn(2, 7, 32)
    actual = attn(x)
    expected = _full_matrix_sliding_window_reference(attn, x)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_sliding_window_scope_excludes_old_tokens():
    from models.attention.sliding_window import SlidingWindowAttention

    torch.manual_seed(1)
    attn = SlidingWindowAttention(
        d_model=32,
        n_heads=4,
        window_size=2,
        dropout=0.0,
        positional_encoding="sinusoidal",
        max_seq_len=16,
    )
    attn.eval()

    x = torch.randn(1, 5, 32)
    x_corrupted = x.clone()
    x_corrupted[:, 0] = x_corrupted[:, 0] + 100.0

    with torch.no_grad():
        out = attn(x)
        out_corrupted = attn(x_corrupted)

    assert torch.allclose(out[:, 4], out_corrupted[:, 4], atol=1e-5)
