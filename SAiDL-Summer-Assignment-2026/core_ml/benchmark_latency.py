"""
Inference latency benchmark script (Task 2c).

Measures tokens/sec and peak GPU memory across different sequence lengths
and attention/PE variants.

Usage:
    python -m core_ml.benchmark_latency \
        --attention_type standard \
        --positional_encoding sinusoidal \
        --seq_lens 512 1024 2048 4096 \
        --output_dir outputs/benchmarks
"""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from core_ml.models.model import TransformerLM


def make_cfg(
    attention_type="standard",
    positional_encoding="sinusoidal",
    block_type="standard",
    max_seq_len=4096,
    n_kv_heads=2,
):
    """Build a model config matching the project defaults."""
    return SimpleNamespace(
        model=SimpleNamespace(
            vocab_size=50257,
            d_model=512,
            n_heads=8,
            n_kv_heads=n_kv_heads,
            n_layers=6,
            d_ff=2048,
            dropout=0.0,  # no dropout for deterministic benchmarking
            max_seq_len=max_seq_len,
            window_size=256,
            conv_kernel_size=5,
            attention_type=attention_type,
            positional_encoding=positional_encoding,
            block_type=block_type,
        )
    )


@torch.no_grad()
def benchmark_forward(model, seq_len, batch_size, device, warmup=3, repeats=10):
    """
    Benchmark forward pass latency and throughput.

    Returns dict with tokens_per_sec, avg_ms, peak_memory_mb.
    """
    model.eval()
    x = torch.randint(0, 50257, (batch_size, seq_len), device=device)

    # Warmup passes (not timed)
    for _ in range(warmup):
        _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    avg_time = sum(times) / len(times)
    total_tokens = batch_size * seq_len
    tokens_per_sec = total_tokens / avg_time

    peak_memory_mb = 0.0
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "avg_ms": round(avg_time * 1000, 2),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "peak_memory_mb": round(peak_memory_mb, 1),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark inference latency and throughput (Task 2c)",
    )
    parser.add_argument(
        "--attention_type",
        default="standard",
        choices=["standard", "sliding_window", "linear", "gqa", "mqa"],
        help="Attention variant to benchmark (default: standard)",
    )
    parser.add_argument(
        "--positional_encoding",
        default="sinusoidal",
        choices=["sinusoidal", "rope", "alibi", "relative", "none"],
        help="Positional encoding to use (default: sinusoidal)",
    )
    parser.add_argument(
        "--block_type",
        default="standard",
        choices=["standard", "conv_before", "interleaved", "gated_conv"],
        help="Block type (default: standard)",
    )
    parser.add_argument(
        "--seq_lens",
        type=int,
        nargs="+",
        default=[512, 1024, 2048],
        help="Sequence lengths to benchmark (default: 512 1024 2048)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for benchmarking (default: 8)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup passes (default: 3)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of timed forward passes (default: 10)",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/benchmarks",
        help="Directory to save results (default: outputs/benchmarks)",
    )
    parser.add_argument(
        "--n_kv_heads",
        type=int,
        default=2,
        help="Number of KV heads for GQA (default: 2)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_seq_len = max(args.seq_lens)

    print(f"=== Inference Latency Benchmark ===")
    print(f"Attention:   {args.attention_type}")
    print(f"PE:          {args.positional_encoding}")
    print(f"Block type:  {args.block_type}")
    print(f"Batch size:  {args.batch_size}")
    print(f"Seq lengths: {args.seq_lens}")
    print(f"Device:      {device}")
    print()

    cfg = make_cfg(
        attention_type=args.attention_type,
        positional_encoding=args.positional_encoding,
        block_type=args.block_type,
        max_seq_len=max_seq_len,
        n_kv_heads=args.n_kv_heads,
    )
    model = TransformerLM(cfg).to(device)
    param_count = model.count_parameters()
    print(f"Parameters:  {param_count:,}")
    print()

    results = {
        "attention_type": args.attention_type,
        "positional_encoding": args.positional_encoding,
        "block_type": args.block_type,
        "batch_size": args.batch_size,
        "param_count": param_count,
        "device": str(device),
        "benchmarks": [],
    }

    print(
        f"{'Seq Len':>8}  {'Tokens/s':>12}  {'Avg (ms)':>10}  {'Peak Mem (MB)':>14}"
    )
    print("-" * 50)

    for seq_len in args.seq_lens:
        try:
            bench = benchmark_forward(
                model, seq_len, args.batch_size, device, args.warmup, args.repeats
            )
            results["benchmarks"].append(bench)
            print(
                f"{bench['seq_len']:>8}  "
                f"{bench['tokens_per_sec']:>12,.1f}  "
                f"{bench['avg_ms']:>10.2f}  "
                f"{bench['peak_memory_mb']:>14.1f}"
            )
        except RuntimeError as e:
            error_msg = str(e)
            if "out of memory" in error_msg.lower():
                print(f"{seq_len:>8}  {'OOM':>12}  {'-':>10}  {'-':>14}")
                results["benchmarks"].append(
                    {"seq_len": seq_len, "error": "OOM"}
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            else:
                raise

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"latency_{args.attention_type}_{args.positional_encoding}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
