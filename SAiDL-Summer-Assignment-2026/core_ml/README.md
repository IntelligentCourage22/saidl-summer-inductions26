# Core ML

This module contains the long-context language-modeling task for the SAiDL
Summer Assignment. It trains a modular causal Transformer on WikiText-2 with
swappable attention mechanisms, positional encodings, and convolutional hybrid
blocks.

## Main Commands

From `SAiDL-Summer-Assignment-2026/core_ml`:

```bash
python train.py
python train.py --set model.attention_type=gqa --set model.positional_encoding=alibi
python eval_extrapolation.py --checkpoint checkpoints/core_ml/final.pt --eval_seq_lens 512 1024 2048
python benchmark_latency.py --attention_type gqa --positional_encoding alibi --seq_lens 512 1024 2048 --wandb
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

## Notes

The sliding-window implementation is a locality-mask ablation over a full
attention matrix. It is semantically local attention, but it is not a true sparse
kernel and therefore does not provide the memory savings of Longformer-style
blocked attention.
