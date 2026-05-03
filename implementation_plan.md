# SAiDL Diffusion Task — RACD Implementation Plan

## Goal

Implement **Regionally-Adaptive Cyclic Diffusion (RACD)**: a latent diffusion system built on a **DiT-B/8** backbone that uses a learned **Difficulty Predictor** to selectively apply cyclic refinement only to hard spatial regions, maximising fidelity while minimising compute. Evaluate using both FID and CMMD against baseline and global-cyclic-refinement ablations.

---

## Task Breakdown (from PDF)

| # | Deliverable | Type |
|---|-------------|------|
| 1 | Baseline DiT + Global Cyclic Refinement | Mandatory |
| 2 | Difficulty Predictor → RACD pipeline | Mandatory |
| 3 | FID + CMMD evaluation, Fidelity-vs-Compute plot | Mandatory |
| B | RL Policy Network (AdaDiff-style) for adaptive τ | **BONUS** |

---

## Phase 0 — Foundations & Environment

### 0.1 Theory Prerequisites

Before writing a single line of code, internalise these concepts (in order):

1. **DDPM forward/reverse process** — the noise schedule `β_t`, the closed-form `q(x_t | x_0)`, the MSE noise-prediction loss `L_simple = E[||ε − ε_θ(x_t, t)||²]`
2. **Latent Diffusion Models (LDM)** — why we train in VAE latent space (`z = E(x)`) instead of pixel space; the pretrained VAE encoder/decoder is frozen
3. **DiT architecture** — patchify layer, adaLN-Zero conditioning, final linear decoder; how patch size controls token count
4. **Cyclic refinement** — re-inject noise to `t_start` and re-denoise; trades compute for error correction
5. **CMMD** — CLIP embeddings + MMD kernel distance (distribution-free alternative to FID)

> [!TIP]
> **Reading order**: DDPM paper → Latent Diffusion paper → DiT paper → ABCD paper (cyclic diffusion) → CMMD paper. Watch the recommended video (till 41:40) first if diffusion is new to you.

### 0.2 Environment Setup

```
Platform        : Kaggle Notebook (free T4/P100) or Google Colab
Python          : 3.10+
Key packages    : torch>=2.1, diffusers, transformers, torchvision,
                  clean-fid, clip-mmd (or manual CMMD), accelerate,
                  wandb (logging), timm
```

### 0.3 Dataset — Landscape

- Source: [Kaggle Landscape Pictures](https://www.kaggle.com/datasets/arnaud58/landscape-pictures), mounted on Kaggle as `/kaggle/input/landscape-pictures`
- **Preprocessing**: Resize all images to **256×256**, normalize to `[-1, 1]`
- **No class conditioning** — unconditional generation
- Split: ~90% train / ~10% held-out for FID/CMMD reference

---

## Phase 1 — Baseline DiT on Landscape (Core Deliverable 1a)

### 1.1 Architecture: DiT-B/8

| Parameter | Value |
|-----------|-------|
| Hidden dim | 768 |
| Heads | 12 |
| Transformer depth | 12 layers |
| Patch size `p` | 8 |
| Latent shape (after VAE) | 32 × 32 × 4 |
| Token sequence length | (32/8)² = 16 tokens |
| Conditioning | Timestep `t` only (no class) |
| Conditioning mechanism | adaLN-Zero |
| Final layer | Linear projection → patch predictions |

> [!IMPORTANT]
> Use the **pretrained VAE from the [facebookresearch/DiT](https://github.com/facebookresearch/DiT) repo** (or equivalently `stabilityai/sd-vae-ft-ema` via `diffusers.AutoencoderKL`). Do NOT train a VAE — freeze it entirely.

### 1.2 Implementation Strategy

**Option A (Recommended):** Fork the official `facebookresearch/DiT` repo. It already has:
- `models.py` with DiT-S/B/L/XL definitions
- `train.py` with DDP training loop
- VAE loading utilities
- Adapt to: remove class conditioning, point at landscape dataset

**Option B:** Build from scratch in PyTorch using `timm` ViT blocks + custom adaLN-Zero. More educational but slower.

### 1.3 Training Configuration

```yaml
# Diffusion
num_timesteps: 1000
beta_schedule: linear          # β_1=1e-4 to β_T=0.02
prediction_target: epsilon     # predict noise (standard DDPM)

# Optimisation
optimizer: AdamW
learning_rate: 1e-4
weight_decay: 0.0
batch_size: 32                 # adjust for GPU memory
ema_decay: 0.9999
epochs: ~200-400               # target ~100k-200k steps
gradient_accumulation: 2-4     # if batch_size limited by VRAM

# Data
image_size: 256
latent_size: 32                # 256/8 (VAE downsamples 8x)
```

### 1.4 Sampling (Baseline)

Use **DDPM sampling** (1000 steps) or **DDIM** (50-250 steps) for faster iteration:

```
z_T ~ N(0, I)                          # sample noise in latent space
for t = T, T-1, ..., 1:
    ε_pred = DiT(z_t, t)               # predict noise
    z_{t-1} = reverse_step(z_t, ε_pred) # DDPM or DDIM step
x_0 = VAE.decode(z_0)                  # decode to pixel space
```

### 1.5 Global Cyclic Refinement (Deliverable 1b)

After standard sampling reaches `z_0`:

```
# Re-inject noise to t=200
z_200 = sqrt(ᾱ_200) * z_0 + sqrt(1 - ᾱ_200) * ε,   ε ~ N(0,I)

# Re-denoise from t=200 back to t=0
for t = 200, 199, ..., 1:
    ε_pred = DiT(z_t, t)
    z_{t-1} = reverse_step(z_t, ε_pred)

x_0_refined = VAE.decode(z_0)
```

**Measure**: generation time (wall-clock per image), FID, CMMD. Document the compute trade-off.

---

## Phase 2 — Difficulty Predictor & RACD (Core Deliverable 2)

### 2.1 Concept

The Difficulty Predictor answers: *"which spatial regions of the latent still have high reconstruction error after the first denoising pass?"*

### 2.2 Supervision Signal Generation

With the **frozen, trained DiT**, run the full denoising chain on training images. At an intermediate timestep `t_mid` (e.g., `t=500`), extract:

1. **Intermediate x̂₀ prediction**: Using the Tweedie formula:
   ```
   x̂₀(t_mid) = (z_{t_mid} - sqrt(1 - ᾱ_{t_mid}) · ε_θ(z_{t_mid}, t_mid)) / sqrt(ᾱ_{t_mid})
   ```
2. **Ground-truth z₀**: The VAE-encoded clean latent
3. **Spatial error map** (supervision target):
   ```
   E(h, w) = || x̂₀(t_mid)[h, w, :] - z₀[h, w, :] ||²    (per spatial location)
   ```
4. **Normalize** to `[0, 1]` → this becomes the ground-truth difficulty map `M_gt`

### 2.3 Difficulty Predictor Architecture

A lightweight CNN that takes intermediate DiT features as input:

```
Input:  DiT intermediate features at t_mid  →  shape (B, 768, 4, 4)
        (extracted from e.g. layer 6 of the 12-layer DiT)

Network:
  ConvTranspose2d(768, 256, k=2, s=2)  →  (B, 256, 8, 8)
  GroupNorm(32) + GELU
  ConvTranspose2d(256, 128, k=2, s=2)  →  (B, 128, 16, 16)
  GroupNorm(32) + GELU
  ConvTranspose2d(128, 64, k=2, s=2)   →  (B, 64, 32, 32)
  GroupNorm(32) + GELU
  Conv2d(64, 1, k=3, p=1) + Sigmoid    →  (B, 1, 32, 32)

Output: M ∈ [0, 1]^{32×32}  (spatial difficulty map in latent resolution)
```

**Loss**: Binary cross-entropy or MSE between predicted `M` and normalised `M_gt`.

> [!IMPORTANT]
> **Two-stage training is mandatory**: First train DiT to convergence. Then freeze DiT weights and train ONLY the Difficulty Predictor on top of frozen DiT features.

### 2.4 RACD Inference Pipeline

```python
# Stage 1: Standard denoising
z_0, intermediate_features = dit_sample_with_features(z_T)

# Stage 2: Predict difficulty
M = difficulty_predictor(intermediate_features)  # (1, 1, 32, 32)

# Stage 3: Selective cyclic refinement
mask = (M > tau).float()  # binary mask at threshold τ

# Re-inject noise ONLY to masked regions
z_200_masked = mask * (sqrt(a_200) * z_0 + sqrt(1-a_200) * eps) + (1 - mask) * z_0

# Re-denoise (full pass, but masked regions benefit most)
for t = 200, ..., 1:
    z_{t-1} = reverse_step(z_t, DiT(z_t, t))

x_final = VAE.decode(z_0)
```

### 2.5 Threshold Sweep

Sweep `τ ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}` and record FID, CMMD, and generation time for each. This feeds directly into the Fidelity-vs-Compute plot.

---

## Phase 3 — Evaluation (Core Deliverable 3)

### 3.1 FID (Fréchet Inception Distance)

Use `clean-fid` for reproducible scores:
```python
from cleanfid import fid
score = fid.compute_fid(real_dir, gen_dir, mode="clean")
```
- Generate **5,000–10,000** images for reliable statistics
- Use the held-out real images as the reference set

### 3.2 CMMD (CLIP Maximum Mean Discrepancy)

Implement manually or use `clip-mmd` package:
```python
# 1. Extract CLIP embeddings (ViT-L/14@336px)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")

# 2. Embed real and generated images
real_embs = encode_images(clip_model, real_images)   # (N, D)
gen_embs  = encode_images(clip_model, gen_images)    # (N, D)

# 3. Compute MMD with Gaussian RBF kernel
K_rr = rbf_kernel(real_embs, real_embs)
K_gg = rbf_kernel(gen_embs, gen_embs)
K_rg = rbf_kernel(real_embs, gen_embs)
cmmd = K_rr.mean() + K_gg.mean() - 2 * K_rg.mean()
```

### 3.3 Required Comparison Table

| Method | FID ↓ | CMMD ↓ | Gen Time (s/img) | Refinement % |
|--------|-------|--------|-------------------|-------------|
| Baseline DiT | — | — | — | 0% |
| Global Cyclic (t=200) | — | — | — | 100% |
| RACD τ=0.3 | — | — | — | ~X% |
| RACD τ=0.5 | — | — | — | ~Y% |
| RACD τ=0.7 | — | — | — | ~Z% |

### 3.4 Fidelity vs. Compute Plot

- **X-axis**: Average generation time per image (seconds)
- **Y-axis**: CMMD score (lower = better)
- Plot points for: baseline, global cyclic, RACD at each τ
- Identify the **Pareto-optimal τ** and explain why in the report

---

## Phase 4 — BONUS: RL Policy Network

### 4.1 Problem Formulation

Instead of a fixed threshold τ, train a policy network that outputs a **continuous action** `t_start ∈ [0, T]` per patch:

| MDP Component | Definition |
|---------------|------------|
| State `s` | Intermediate DiT features at a patch location |
| Action `a` | `t_start` — how far back to re-inject noise (0 = skip, T = full re-denoise) |
| Reward `R` | `−CMMD_error − λ · compute_time` |

### 4.2 Policy Architecture

Replace the Difficulty Predictor's final Sigmoid with a continuous output head:
```
Same CNN backbone as Difficulty Predictor
→ Conv2d(64, 1, k=3, p=1)
→ Sigmoid * T                    # output t_start ∈ [0, T] per patch
```

### 4.3 Training with REINFORCE

```python
for episode in range(num_episodes):
    # 1. Standard denoising pass
    z_0, features = dit_sample_with_features(z_T)

    # 2. Policy predicts per-patch t_start
    t_start_map = policy_net(features)  # (1, 1, H, W)

    # 3. Per-patch cyclic refinement with variable t_start
    z_refined = patch_wise_cyclic_refine(z_0, t_start_map)
    x_gen = vae.decode(z_refined)

    # 4. Compute reward
    cmmd_err = compute_cmmd(x_gen, real_batch)
    time_cost = measure_time(t_start_map)
    reward = -cmmd_err - lambda_ * time_cost

    # 5. REINFORCE update
    log_prob = policy_net.log_prob(t_start_map, features)
    loss = -(reward - baseline) * log_prob
    loss.backward()
    optimizer.step()
```

---

## Codebase Structure

```
diffusion/
├── configs/
│   └── dit_landscape.yaml          # all hyperparameters
├── data/
│   ├── dataset.py                  # landscape dataset loader (resize 256²)
│   └── download.py                 # download + preprocess script
├── models/
│   ├── dit.py                      # DiT-B/8 (fork from facebookresearch/DiT)
│   ├── vae.py                      # frozen VAE wrapper (load sd-vae-ft-ema)
│   ├── difficulty_predictor.py     # lightweight CNN predictor
│   └── policy_net.py               # RL policy (BONUS)
├── diffusion/
│   ├── gaussian_diffusion.py       # noise schedule, q_sample, p_sample
│   ├── samplers.py                 # DDPM + DDIM samplers
│   └── cyclic_refinement.py        # global + region-aware cyclic logic
├── evaluation/
│   ├── fid.py                      # clean-fid wrapper
│   ├── cmmd.py                     # CLIP-MMD implementation
│   └── visualize.py                # sample grids, difficulty maps, plots
├── train_dit.py                    # Phase 1: train base DiT
├── generate_supervision.py         # Phase 2a: generate difficulty maps
├── train_predictor.py              # Phase 2b: train difficulty predictor
├── train_policy.py                 # Phase 4: RL training (BONUS)
├── sample.py                       # unified sampling (baseline/global/RACD)
├── evaluate.py                     # run FID + CMMD + timing
└── report/
    └── diffusion_report.tex        # LaTeX report
```

---

## Timeline & Compute Budget

| Phase | Duration | GPU Hours (T4) | Deliverable |
|-------|----------|----------------|-------------|
| 0: Setup + theory | 2–3 days | 0 | Environment ready, papers read |
| 1a: DiT training | 4–5 days | ~30–50h | Converged DiT, baseline samples |
| 1b: Global cyclic | 1 day | ~2h | Cyclic samples + timing |
| 2a: Supervision gen | 1 day | ~4h | Difficulty map dataset |
| 2b: Predictor train | 1–2 days | ~5h | Trained predictor |
| 2c: RACD pipeline | 1 day | ~2h | End-to-end RACD sampling |
| 3: Evaluation | 2 days | ~8h | FID, CMMD, plots, τ sweep |
| 4: BONUS (RL) | 3–4 days | ~15h | Policy network results |
| 5: Report | 2–3 days | 0 | LaTeX writeup |
| **Total** | **~2.5–3 weeks** | **~65–85h** | |

> [!WARNING]
> **DiT training is the bottleneck.** On a single T4, expect ~200k steps to take 30–50 hours. Use mixed precision (`fp16`/`bf16`), gradient accumulation, and EMA. Consider using DDIM (50 steps) during evaluation to save time on sampling.

---

## Key Risk Mitigations

| Risk | Mitigation |
|------|-----------|
| DiT doesn't converge on small dataset | Use strong augmentation (random crop, flip, color jitter); consider starting from a pretrained DiT checkpoint and fine-tuning |
| Difficulty maps are too uniform | Try multiple `t_mid` values (300, 500, 700); use per-channel error rather than just L2 |
| CMMD implementation bugs | Validate against the official Google Research CMMD repo; test on known distributions |
| OOM on T4 (16GB) | Reduce batch size to 8–16; use gradient checkpointing; accumulate gradients |
| Cyclic refinement shows no improvement | Tune `t_start` for re-injection (try 100, 200, 300); ensure VAE decode quality is not the bottleneck |

---

## Report Structure (LaTeX)

1. **Introduction** — Problem statement, motivation for adaptive refinement
2. **Background** — DDPM, LDM, DiT, cyclic refinement, CMMD vs FID
3. **Method**
   - 3.1 Baseline DiT training on landscape
   - 3.2 Global cyclic refinement
   - 3.3 Difficulty Predictor design and supervision
   - 3.4 RACD pipeline
   - 3.5 RL policy (BONUS)
4. **Experiments**
   - Comparison table (FID, CMMD, time)
   - Fidelity-vs-Compute Pareto plot
   - Generated image grids (baseline vs global vs RACD)
   - Difficulty map visualizations
5. **Analysis** — optimal τ discussion, compute trade-off, failure cases
6. **Conclusion**

---

## Open Questions

> [!IMPORTANT]
> **Dataset confirmation**: Use the Kaggle `arnaud58/landscape-pictures` dataset for the SAiDL diffusion task.

> [!IMPORTANT]
> **Compute platform**: Are you using Kaggle (free T4, 30h/week) or Colab Pro? This will affect batch sizes and whether we need checkpoint-resume logic.

> [!NOTE]
> **DiT from scratch vs fork**: Do you want to build DiT from scratch for learning purposes, or fork the official repo and adapt it? The fork approach is ~3x faster to get running.
