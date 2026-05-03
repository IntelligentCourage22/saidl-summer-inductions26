# Diffusion Implementation — Deep Critical Review

## 1. Assignment Alignment

The PDF (Section: Diffusion, pages 8–9) requires:

| Requirement | Status | Notes |
|---|---|---|
| Baseline DiT on landscape dataset | ✅ Implemented | But undersized vs spec |
| Global cyclic refinement (noise to t=200, denoise to t=0) | ✅ Implemented | Sampler mismatch issue |
| Measure FID/CMMD + generation time | ⚠️ Scaffolded | Code exists, no results |
| Difficulty predictor on frozen DiT features | ✅ Implemented | Timestep mismatch bug |
| Spatial difficulty map M ∈ [0,1]^{h×w} | ✅ Implemented | |
| Supervision from x̂₀ vs ground-truth x₀ error | ✅ Implemented | |
| Integrate predictor, apply refinement where difficulty > τ | ✅ Implemented | Mask blending flaw |
| CMMD evaluation (CLIP + MMD) | ✅ Implemented | |
| Compare RACD vs baseline vs global cyclic (FID, CMMD, time) | ❌ Missing | Report is empty |
| **Plot Fidelity (CMMD) vs Compute for varying τ** | ❌ Missing | Explicitly required |
| **Find optimal τ and explain why** | ❌ Missing | Explicitly required |
| Use pretrained VAE, latent space only | ✅ Correct | |
| Images resized to 256×256 | ✅ Correct | |
| Fix DiT config (e.g. DiT-B/8) | ⚠️ Partial | Uses DiT-S/4 instead |
| LaTeX report with generated images | ❌ Empty | All values are `--` |

> [!WARNING]
> The two most visible missing deliverables are the **τ-sweep plot** and the **completed report with numbers and image grids**. These are explicitly required and their absence would be immediately noticed by a reviewer.

---

## 2. Critical Issues (P0 — Must Fix)

### C1: Sampler Mismatch Between Initial Pass and Cyclic Refinement

**Files:** [samplers.py](file:///c:/Users/ansh0/Desktop/saidl-summer-inductions/saidl-summer-inductions26/SAiDL-Summer-Assignment-2026/diffusion/diffusion/samplers.py#L9-L28) vs [samplers.py:L33-L38](file:///c:/Users/ansh0/Desktop/saidl-summer-inductions/saidl-summer-inductions26/SAiDL-Summer-Assignment-2026/diffusion/diffusion/samplers.py#L33-L38)

`sample_loop` generates initial samples using **DDIM** (50 steps). But `global_cyclic_refine` → `denoise_from_t` uses **DDPM** `p_sample` (201 individual steps from t=200 to t=0).

**Why this is critical:**
- DDIM and DDPM have fundamentally different update rules. The cyclic refinement pass is 4× slower than the initial generation, making compute comparisons meaningless.
- A reviewer will immediately question why you used two different samplers and whether your timing comparison is fair.

**Fix:** Use DDIM for cyclic refinement too. Write a `ddim_denoise_from_t` that mirrors the DDIM stepping in `sample_loop`.

---

### C2: Frankenstein Input During Masked Refinement

**File:** [samplers.py:L48-L53](file:///c:/Users/ansh0/Desktop/saidl-summer-inductions/saidl-summer-inductions26/SAiDL-Summer-Assignment-2026/diffusion/diffusion/samplers.py#L48-L53) and [gaussian_diffusion.py:L107-L114](file:///c:/Users/ansh0/Desktop/saidl-summer-inductions/saidl-summer-inductions26/SAiDL-Summer-Assignment-2026/diffusion/diffusion/gaussian_diffusion.py#L107-L114)

`masked_cyclic_refine` creates a composite tensor: `mask * noised_z + (1-mask) * clean_z`. Then `denoise_from_t` denoises the **entire** tensor — including clean unmasked regions that are at t=0 while masked regions are at t=200.

**Why this is critical:** The model was trained on uniformly-noised inputs at a single timestep. Feeding it a spatially-heterogeneous noise level (some patches at t=200, others at t=0) is **out-of-distribution**. This will produce boundary artifacts and degrade quality in the exact regions you're trying to preserve.

**Fix:** Either:
1. Denoise only the masked region (extract patches, denoise, re-insert), or
2. Use RePaint-style alternating: at each denoising step, re-inject the known (unmasked) region from the forward process at the current timestep.

---

### C3: Feature Timestep Mismatch Between Training and Inference

**Training:** [generate_supervision.py:L77](file:///c:/Users/ansh0/Desktop/saidl-summer-inductions/saidl-summer-inductions26/SAiDL-Summer-Assignment-2026/diffusion/generate_supervision.py#L77) — features extracted at exactly `t_mid = 500`.

**Inference:** [samplers.py:L22](file:///c:/Users/ansh0/Desktop/saidl-summer-inductions/saidl-summer-inductions26/SAiDL-Summer-Assignment-2026/diffusion/diffusion/samplers.py#L22) — features extracted at the first DDIM step where `t ≤ 500`.

With 50 DDIM steps from 999→0, the actual timestep hitting this condition is approximately t≈489 (the 26th step in `linspace(999, 0, 50)`). The predictor was trained on t=500 features but receives t≈489 features at inference. This distributional shift degrades predictor accuracy.

**Fix:** Record the exact timestep used during supervision and match it at inference, or train the predictor on features from multiple timesteps for robustness.

---

### C4: Developer's Own "FATAL" Warnings Left in Code

**File:** [gaussian_diffusion.py:L52](file:///c:/Users/ansh0/Desktop/saidl-summer-inductions/saidl-summer-inductions26/SAiDL-Summer-Assignment-2026/diffusion/diffusion/gaussian_diffusion.py#L52) and [L89](file:///c:/Users/ansh0/Desktop/saidl-summer-inductions/saidl-summer-inductions26/SAiDL-Summer-Assignment-2026/diffusion/diffusion/gaussian_diffusion.py#L89)

```python
# refine this idk why ts doesnt work
# needs improvement ( FATAL ) need to check on alpha_t params
```

These comments signal to a reviewer that the author is not confident in core mathematical correctness. Even though the DDIM formula is actually correct upon inspection, these comments severely undermine credibility.

**Fix:** Remove these comments. The DDIM step and `predict_x0_from_eps` are mathematically correct — verify and clean up.

---

## 3. Major Issues (P1 — Should Fix)

### M1: Model Significantly Undersized vs Assignment Spec

The PDF says: *"Fix your DiT model size/config (e.g. DiT-B/8)"*

| Parameter | Your Config (DiT-S/4) | DiT-B/8 (Specified) |
|---|---|---|
| hidden_size | 384 | 768 |
| depth | 6 | 12 |
| num_heads | 6 | 12 |
| patch_size | 4 | 8 |
| Approx params | ~8M | ~130M |

Your model is **~16× smaller** than the specified baseline. While Kaggle constraints justify some reduction, the gap is enormous. A reviewer will question whether your RACD results transfer to a properly-sized model.

**Mitigation:** Either scale up to at least DiT-S/8 (~33M params) or explicitly justify the choice with a compute budget analysis in the report.

### M2: No Learning Rate Schedule

**File:** [train_dit.py:L80-L84](file:///c:/Users/ansh0/Desktop/saidl-summer-inductions/saidl-summer-inductions26/SAiDL-Summer-Assignment-2026/diffusion/train_dit.py#L80-L84)

Flat lr=1e-4 for 20k steps with no warmup. Diffusion transformers universally use warmup + cosine/linear decay. This likely leaves significant performance on the table.

### M3: No Validation Monitoring During Training

The training loop never evaluates on a held-out set. There is no early stopping, no validation loss tracking, no periodic sample generation. You cannot detect overfitting or determine optimal stopping point.

### M4: Minimal Data Augmentation for 5000 Images

Only `RandomHorizontalFlip` on 5000 images. For a dataset this small, add: random resized crop, color jitter, and potentially random rotation. The model will overfit quickly without stronger augmentation.

### M5: `weight_decay: 0.0` with AdamW

Using AdamW with zero weight decay is identical to Adam. Either use Adam (simpler) or set a real weight decay (0.01–0.05). This suggests a copy-paste config that wasn't tuned.

### M6: Dead Config Key

`diffusion.prediction_target: epsilon` in the YAML is never read by any code. This is cosmetic but signals to reviewers that the config and code are not in sync.

---

## 4. Technical Correctness Audit

### ✅ Correct

| Component | Verdict |
|---|---|
| Forward diffusion q_sample | Correct: `√ᾱ_t · x₀ + √(1-ᾱ_t) · ε` |
| predict_x0_from_eps | Correct: `(x_t - √(1-ᾱ_t)·ε) / √ᾱ_t` |
| DDPM p_sample (mean formula) | Correct: `√(1/α_t) · (x_t - β_t·ε/√(1-ᾱ_t))` |
| DDIM step (η=0) | Correct: `√ᾱ_{t-1}·x̂₀ + √(1-ᾱ_{t-1})·ε` |
| Posterior variance | Correct: `β_t · (1-ᾱ_{t-1})/(1-ᾱ_t)` with ᾱ_{-1}=1 |
| AdaLN-Zero conditioning | Correct: shift/scale/gate modulation |
| VAE encode/decode with scaling factor | Correct: 0.18215 matches SD-VAE |
| 2D sincos position embedding | Correct |
| CMMD with RBF kernel | Correct implementation |
| EMA update rule | Correct: `shadow = decay·shadow + (1-decay)·new` |

### ⚠️ Fragile

| Component | Issue |
|---|---|
| DDIM timestep schedule | `linspace().long().unique_consecutive()` — number of actual steps is unpredictable after dedup |
| `p_sample` at t=0 | Variance clamp at `1e-20` prevents NaN, but the `nonzero_mask` already handles this. Redundant but safe. |
| `GaussianDiffusion.to()` | Reconstructs from float-converted beta endpoints — potential tiny numerical drift |

---

## 5. Architecture & Design Quality

### DiT Architecture: B+
The `LatentDiT` follows the original DiT paper correctly: patchify → positional embedding → transformer blocks with AdaLN-Zero → final layer → unpatchify. The `unpatchify` using `einsum` is clean. Zero-initialization of AdaLN outputs and final layer is correct per the DiT paper.

### Difficulty Predictor: B-
Reasonable design (ConvTranspose upsampling from 8×8 → 32×32), but:
- The upsampling path (8→16→32) introduces spatial imprecision
- No skip connections from the DiT's spatial structure
- Sigmoid output is correct for [0,1] difficulty maps

### Training Pipeline: C+
- Missing: LR schedule, validation, resume-from-checkpoint logic (config has `resume: null` but no code reads it)
- Present: Gradient accumulation, AMP, EMA, gradient clipping, JSONL logging, W&B integration
- The deprecated `torch.cuda.amp` API will emit warnings

### Sampling Pipeline: C
- Correct DDIM sampling loop
- Feature extraction hook is well-designed
- But the DDIM/DDPM mismatch and masked refinement issues (C1, C2) are fundamental

### Evaluation: B
- CMMD is correctly implemented with CLIP embeddings + RBF-MMD
- FID delegates to `cleanfid` which is a strong choice
- Missing: the τ-sweep plot that the assignment explicitly requires

---

## 6. What a Top-Tier Reviewer Would Criticize

1. **"Your report is empty."** — This alone would fail the submission. No numbers, no images, no analysis.
2. **"Why did you use two different samplers?"** — The DDIM/DDPM inconsistency is hard to defend.
3. **"How do you handle boundary artifacts in masked refinement?"** — No discussion of the out-of-distribution issue.
4. **"Your model is 16× smaller than specified. How do results transfer?"** — Needs explicit justification.
5. **"Where is the τ vs CMMD plot?"** — Explicitly required by the assignment.
6. **"What about the FATAL comment in your own codebase?"** — Instant credibility loss.

---

## 7. Suggested Improvements (Ranked by Impact)

### High Impact (do these first)
1. **Fix the sampler mismatch** — Use DDIM everywhere. Write a `ddim_denoise_from_t` helper.
2. **Fix masked refinement** — Use RePaint-style blending at each denoising step.
3. **Add LR warmup + cosine decay** — 500-step warmup, cosine to 0. Expect 10-20% quality improvement.
4. **Remove self-doubt comments** — Clean up all TODO/FATAL markers.
5. **Run experiments and fill the report** — This is the #1 priority for the submission.

### Medium Impact
6. **Add stronger augmentation** — RandomResizedCrop, ColorJitter.
7. **Add validation loop** — Sample every N steps, compute FID on a small val set.
8. **Match feature timestep** — Use the same `t_mid` in supervision and inference.
9. **Implement τ sweep script** — Automate τ ∈ {0.1, 0.2, ..., 0.9} and plot CMMD vs time.
10. **Add `weight_decay: 0.01`** — Standard for transformer training.

### Polish
11. Scale model closer to DiT-B/8 or justify the gap.
12. Train predictor on multi-timestep features for robustness.
13. Add CFG (classifier-free guidance) — even unconditional models benefit from null-conditioning training.
14. Use `torch.amp` instead of deprecated `torch.cuda.amp`.

---

## 8. Final Verdict

### Score: 5.5 / 10

**Justification:**

| Dimension | Score | Weight |
|---|---|---|
| Assignment completeness | 4/10 | 25% — Report empty, no results, missing τ plot |
| Technical correctness | 6/10 | 25% — Core math is right but sampler mismatch and mask blending are real bugs |
| Code quality | 7/10 | 15% — Well-structured, modular, but FATAL comments and dead config |
| Architecture design | 7/10 | 15% — Follows DiT paper correctly, difficulty predictor is reasonable |
| Research presentation | 3/10 | 20% — Empty report, no analysis, no generated images |

**The code infrastructure is solid** — you have a complete end-to-end pipeline (train → sample → evaluate) with proper config management, EMA, AMP, logging, and wandb. This is genuinely good engineering.

**But the submission is incomplete.** The assignment grades on documented results, analysis, and understanding. An empty report with `--` in every cell cannot score well regardless of code quality.

> [!IMPORTANT]
> **To reach 8+/10:** Fix C1-C3, run all experiments on Kaggle, fill the report with real numbers and image grids, produce the τ-sweep plot, and write 2-3 paragraphs of analysis per experiment. The code is ~80% there; the deliverable gap is primarily in execution and documentation.
