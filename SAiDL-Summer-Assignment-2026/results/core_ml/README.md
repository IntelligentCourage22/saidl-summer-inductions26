# Core ML Results

This directory stores small, versioned experiment artifacts used by
`reports/core_ml_report.tex`. Large checkpoints, W&B directories, and raw output
folders are intentionally excluded from git.

## Included Artifacts

- `gqa_rope_full_rerun/`: Full metric tail and final metrics for the completed
  GQA + RoPE rerun.
- `gqa_alibi_512/`: Final metrics, best metric snapshot, available metric tail,
  and 512-to-2048 length extrapolation results for GQA + ALiBi.
- `benchmarks/latency_gqa_alibi.json`: Inference latency benchmark for GQA +
  ALiBi at context lengths 512, 1024, and 2048.
- `experiment_summary.json`: Compact aggregate table containing the values used
  across the Core ML report.

The `metrics_tail.jsonl` file for `gqa_alibi_512` contains the available pasted
tail rows from the Colab/W&B run rather than the complete raw training log.
