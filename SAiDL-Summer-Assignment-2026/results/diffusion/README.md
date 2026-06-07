# Diffusion Results

The final reported diffusion experiment uses the fixed DiT-S/4 latent
configuration from `diffusion/configs/dit_landscape_compact.yaml`.
The final S/4 checkpoint was trained to 100k optimizer steps on Kaggle, with
the full checkpoint/history tracked through W&B artifacts. Large weights are
not committed to this repository.

Final report artifacts:

- `tau_sweep_s4/tau_sweep_results.json`: baseline, global cyclic refinement,
  and RACD threshold sweep used in `reports/diffusion_report.tex`.
- `tau_sweep_s4/tau_sweep_results.csv`: tabular version of the same sweep.
- `tau_sweep_s4/cmmd_vs_time.png`: CMMD-vs-generation-time plot.
- `final_s4_sample_grid.png`: generated samples from baseline, global cyclic,
  and RACD at the selected threshold.

The older `tau_sweep/`, `baseline_dit/`, and small `eval_*_16.json` artifacts
are retained as diagnostic evidence from earlier undertrained runs. They are
not the final numbers used in the submitted report.
