# Core ML

This module contains the long-context language-modeling task for the SAiDL
Summer Assignment. It trains a modular causal Transformer on WikiText-2 with
swappable attention mechanisms, positional encodings, and convolutional hybrid
blocks.

## Main Commands

From `SAiDL-Summer-Assignment-2026/core_ml`:

```bash
python train.py --set training.max_steps=20 --set training.eval_every=10 --set training.batch_size=2 --set training.seq_len=128 --set model.max_seq_len=128
python train.py --set model.attention_type=gqa --set model.positional_encoding=alibi
python eval_extrapolation.py --checkpoint checkpoints/core_ml/final.pt --eval_seq_lens 512 1024 2048
python benchmark_latency.py --attention_type gqa --positional_encoding alibi --seq_lens 512 1024 2048
```

From the repository root, the same entry points also work as modules:

```bash
python -m core_ml.train --set training.max_steps=20 --set training.eval_every=10 --set training.batch_size=2 --set training.seq_len=128 --set model.max_seq_len=128
python -m core_ml.eval_extrapolation --checkpoint checkpoints/core_ml/final.pt --eval_seq_lens 512 1024 2048
python -m core_ml.benchmark_latency --attention_type gqa --positional_encoding alibi --seq_lens 512 1024 2048
```

## Implemented Variants

- Attention: `standard`, `sliding_window`, `linear`, `mqa`, `gqa`.
- Positional encoding: `sinusoidal`, `rope`, `alibi`, `relative`, `none`.
- Blocks: `standard`, `conv_before`, `interleaved`, `gated_conv`.

`linear` attention intentionally supports only `sinusoidal` and `none`; ALiBi
and learned relative bias require an explicit score matrix, and RoPE's
dot-product relative-position property does not carry cleanly through the
nonlinear ELU+1 kernel map.

## Results

Small result artifacts are stored under `../results/core_ml/`, with the report
in `../reports/core_ml_report.tex`. Large checkpoints, raw W&B folders, and
temporary output directories are intentionally excluded from git.

W&B is disabled by default so the scripts run cleanly in a fresh environment.
To log a training run, pass `--set wandb.enabled=true`. To log latency or
extrapolation runs, add `--wandb` to those scripts.

## Notes

The sliding-window implementation builds local key/value windows directly, so
the score and probability tensors scale with `sequence_length * window_size`
rather than a full quadratic attention matrix. The shared full causal mask is
only constructed for the standard, MQA, and GQA attention paths that need it.

`benchmark_latency.py` measures dense forward-pass latency on random token
inputs. It is intended for architecture-level scaling comparisons, not cached
autoregressive decoding throughput.
