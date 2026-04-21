# SAiDL Summer Assignment 2026

This repository currently focuses on the compulsory Core ML task: modular
long-context language modeling on WikiText-2.

## Current Core ML Status

- Standard Transformer language model implemented.
- Attention variants wired through config: `standard`, `sliding_window`, `linear`,
  `mqa`, and `gqa`.
- Positional variants wired through config: `sinusoidal`, `rope`, `alibi`,
  `relative`, and `none`.
- Hybrid block variants wired through config: `standard`, `conv_before`,
  `interleaved`, and `gated_conv`.
- Training script logs JSONL metrics for loss, perplexity, throughput, and peak
  GPU memory.
- LaTeX report scaffold lives in `reports/core_ml_report.tex`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run A Small Smoke Experiment

```powershell
python -m core_ml.train --set training.batch_size=2 --set training.seq_len=128 --set model.max_seq_len=128 --set training.max_steps=20 --set training.eval_every=10
```

## Example Experiment Overrides

```powershell
python -m core_ml.train --set model.attention_type=sliding_window --set model.window_size=256
python -m core_ml.train --set model.attention_type=linear
python -m core_ml.train --set model.attention_type=gqa --set model.n_kv_heads=2
python -m core_ml.train --set model.positional_encoding=rope
python -m core_ml.train --set model.positional_encoding=alibi
python -m core_ml.train --set model.block_type=conv_before
python -m core_ml.train --set model.block_type=gated_conv
```

## Suggested Experiment Plan

1. Establish the baseline at context length 1024 with `standard` attention and
   `sinusoidal` positions.
2. Run attention ablations at context lengths 512, 1024, and 2048 if memory
   allows: `sliding_window`, `linear`, and `gqa`.
3. Run positional extrapolation: train at 512, then evaluate or retrain configs
   with `sinusoidal`, `rope`, `alibi`, and `relative` at longer contexts.
4. Combine the best attention and positional choices with `conv_before` and
   `gated_conv`.
5. Copy the final metrics into the LaTeX tables and include short analysis of
   stability, memory, and throughput tradeoffs.
