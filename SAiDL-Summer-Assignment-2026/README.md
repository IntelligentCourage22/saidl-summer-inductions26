# SAiDL Summer Assignment 2026

This repository currently focuses on the compulsory Core ML task: modular
long-context language modeling on WikiText-2. It also includes the start of the
Diffusion domain task: a compute-aware latent DiT baseline plus RACD scaffolding.

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

## Diffusion Task

The diffusion code lives in `diffusion/`. It is designed for Kaggle free-tier
experiments, but the default config now follows the assignment baseline:
DiT-B/8 on `32x32` VAE latents (`hidden_size=768`, `depth=12`,
`num_heads=12`, `patch_size=8`). A smaller debug config is available at
`diffusion/configs/dit_landscape_compact.yaml` for smoke tests only; report
numbers should use the default DiT-B/8 config.

### Kaggle Setup

```python
%cd /kaggle/working
!git clone https://github.com/IntelligentCourage22/saidl-summer-inductions26.git repo
%cd /kaggle/working/repo/SAiDL-Summer-Assignment-2026
!pip install -q -r requirements.txt
```

### Train The Baseline Latent DiT

Set `data.root` to the Kaggle landscape dataset directory that exists in your
notebook. For the SAiDL diffusion task, attach the Kaggle dataset
`arnaud58/landscape-pictures`; Kaggle usually mounts it at
`/kaggle/input/landscape-pictures`.

```python
!python diffusion/train_dit.py \
  --set data.root="/kaggle/input/landscape-pictures" \
  --set training.max_steps=2000 \
  --set training.output_dir="results/diffusion/baseline_dit" \
  --set training.checkpoint_dir="checkpoints/diffusion/baseline_dit"
```

For a fast code-path smoke test, add:

```python
--config diffusion/configs/dit_landscape_compact.yaml
```

### Generate Baseline And Global Cyclic Samples

```python
!python diffusion/sample.py \
  --checkpoint checkpoints/diffusion/baseline_dit/final.pt \
  --mode baseline \
  --set sampling.num_images=64 \
  --set sampling.output_dir="results/diffusion/samples"

!python diffusion/sample.py \
  --checkpoint checkpoints/diffusion/baseline_dit/final.pt \
  --mode global_cyclic \
  --set sampling.num_images=64 \
  --set sampling.output_dir="results/diffusion/samples"
```

### Difficulty Predictor And RACD

```python
!python diffusion/generate_supervision.py \
  --checkpoint checkpoints/diffusion/baseline_dit/final.pt \
  --set predictor.supervision_dir="results/diffusion/difficulty_supervision"

!python diffusion/train_predictor.py \
  --set predictor.supervision_dir="results/diffusion/difficulty_supervision" \
  --set predictor.output_dir="results/diffusion/difficulty_predictor"

!python diffusion/sample.py \
  --checkpoint checkpoints/diffusion/baseline_dit/final.pt \
  --mode racd \
  --predictor-checkpoint results/diffusion/difficulty_predictor/difficulty_predictor.pt \
  --tau 0.5 \
  --set sampling.num_images=64 \
  --set sampling.output_dir="results/diffusion/samples"
```

### Evaluate Generated Folders

```python
!python diffusion/export_real_images.py \
  --set data.root="/kaggle/input/landscape-pictures" \
  --output-dir="results/diffusion/real_val" \
  --max-images=1000

!python diffusion/evaluate.py \
  --real-dir results/diffusion/real_val \
  --generated-dir results/diffusion/samples/baseline \
  --output results/diffusion/eval_baseline.json
```

### Required Tau Sweep

This runs baseline, global cyclic refinement, and RACD for each threshold,
then writes `tau_sweep_results.json`, `tau_sweep_results.csv`, and the required
`cmmd_vs_time.png` plot.

```python
!python diffusion/run_tau_sweep.py \
  --checkpoint checkpoints/diffusion/baseline_dit/final.pt \
  --predictor-checkpoint results/diffusion/difficulty_predictor/difficulty_predictor.pt \
  --real-dir results/diffusion/real_val \
  --output-dir results/diffusion/tau_sweep \
  --num-images 256 \
  --taus 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
```

Create the image grid for the LaTeX report:

```python
!python diffusion/make_sample_grid.py \
  --folders results/diffusion/tau_sweep/samples/baseline results/diffusion/tau_sweep/samples/global_cyclic results/diffusion/tau_sweep/samples/racd_tau_0p5/racd \
  --labels Baseline Global-Cyclic RACD \
  --output results/diffusion/sample_grid.png
```
