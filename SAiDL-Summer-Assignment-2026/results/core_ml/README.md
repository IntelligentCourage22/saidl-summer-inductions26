# Core ML Results

This directory stores small, versioned experiment artifacts used by
`reports/core_ml_report.tex`. Large checkpoints, W&B directories, and raw output
folders are intentionally excluded from git.

## Included Artifacts

- `gqa_rope_full_rerun/`: Full metric tail and final metrics for the completed
  GQA + RoPE rerun.
- `gqa_alibi_512/`: Final metrics, best metric snapshot, available metric tail,
  and 512-to-2048 length extrapolation results for GQA + ALiBi.
- `gqa_rope_512/`: Final metrics, best metric snapshot, available metric tail,
  and 512-to-2048 length extrapolation results for GQA + RoPE.
- `gqa_relative_512/`: Final metrics, best metric snapshot, available metric
  tail, and 512-to-2048 length extrapolation results for GQA + learned relative
  position bias.
- `standard_sinusoidal_512/`: Standard Transformer + sinusoidal baseline at
  context length 512 and its 512-to-2048 extrapolation results.
- `benchmarks/latency_*.json`: Inference latency benchmarks for standard,
  sliding-window, linear, and GQA + ALiBi attention at context lengths 512,
  1024, and 2048.
- `experiment_summary.json`: Compact aggregate table containing the values used
  across the Core ML report.

The `metrics_tail.jsonl` files for the 512-context runs contain the available
pasted tail rows from the Colab/W&B runs rather than complete raw training logs.
