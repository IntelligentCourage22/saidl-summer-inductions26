"""
Extrapolation evaluation script (Task 3c).

Tests whether positional encodings can generalise beyond their training length.
Train at seq_len=512, evaluate at 512, 1024, and 2048.

Usage:
    python -m core_ml.eval_extrapolation \
        --checkpoint checkpoints/core_ml/final.pt \
        --eval_seq_lens 512 1024 2048 \
        --output_dir outputs/extrapolation
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import get_dataloaders
from models.model import TransformerLM


def to_namespace(value):
    """Recursively convert dicts to SimpleNamespace."""
    if isinstance(value, dict):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [to_namespace(item) for item in value]
    return value


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    """Evaluate model perplexity on a dataloader."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        _, loss = model(x, y)
        total_loss += loss.item() * x.numel()
        total_tokens += x.numel()

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 100))
    return {"loss": avg_loss, "perplexity": perplexity, "total_tokens": total_tokens}


def load_checkpoint(checkpoint_path, device):
    """Load model from a checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    cfg = to_namespace(config)
    model = TransformerLM(cfg).to(device)
    state_dict = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in ckpt["model"].items()
    }
    model.load_state_dict(state_dict)
    model.eval()
    return model, config, cfg


def maybe_init_wandb(config, args, pe_type, attn_type, train_seq_len):
    """Reuse the checkpoint's W&B settings for extrapolation runs."""
    wandb_cfg = config.get("wandb", {})
    if not args.wandb:
        return None

    try:
        import wandb
    except ImportError:
        print("wandb not installed; continuing without W&B logging.")
        return None

    return wandb.init(
        project=wandb_cfg.get("project", "SAiDL-LongContext"),
        entity=wandb_cfg.get("entity"),
        job_type="core_ml_extrapolation",
        name=f"extrapolation_{attn_type}_{pe_type}_train{train_seq_len}",
        config={
            "checkpoint": args.checkpoint,
            "eval_seq_lens": args.eval_seq_lens,
            "batch_size": args.batch_size,
            "max_batches": args.max_batches,
            "attention_type": attn_type,
            "positional_encoding": pe_type,
            "train_seq_len": train_seq_len,
        },
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate positional encoding extrapolation (Task 3c)",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--eval_seq_lens",
        type=int,
        nargs="+",
        default=[512, 1024, 2048],
        help="Sequence lengths to evaluate at (default: 512 1024 2048)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation (default: 8)",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/extrapolation",
        help="Directory to save results (default: outputs/extrapolation)",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Max batches per eval (default: full validation set)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log extrapolation metrics to W&B.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, cfg = load_checkpoint(args.checkpoint, device)

    pe_type = config.get("model", {}).get("positional_encoding", "unknown")
    attn_type = config.get("model", {}).get("attention_type", "unknown")
    train_seq_len = config.get("training", {}).get("seq_len", "unknown")

    print(f"=== Extrapolation Evaluation ===")
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"PE type:     {pe_type}")
    print(f"Attn type:   {attn_type}")
    print(f"Train len:   {train_seq_len}")
    print(f"Eval lens:   {args.eval_seq_lens}")
    print(f"Device:      {device}")
    print()

    run = maybe_init_wandb(config, args, pe_type, attn_type, train_seq_len)

    results = {
        "checkpoint": args.checkpoint,
        "positional_encoding": pe_type,
        "attention_type": attn_type,
        "train_seq_len": train_seq_len,
        "evaluations": [],
    }

    for seq_len in args.eval_seq_lens:
        print(f"Evaluating at seq_len={seq_len}... ", end="", flush=True)

        model.resize_position_buffers(seq_len)

        try:
            _, val_loader, _ = get_dataloaders(
                seq_len=seq_len,
                batch_size=args.batch_size,
                num_workers=0,  # avoid multiprocessing issues during eval
                dataset_name=config.get("data", {}).get("dataset_name", "wikitext"),
                dataset_config=config.get("data", {}).get(
                    "dataset_config", "wikitext-2-raw-v1"
                ),
                tokenizer_name=config.get("data", {}).get("tokenizer_name", "gpt2"),
            )

            start = time.time()
            metrics = evaluate(model, val_loader, device, args.max_batches)
            elapsed = time.time() - start

            metrics["seq_len"] = seq_len
            metrics["eval_time_sec"] = round(elapsed, 2)
            results["evaluations"].append(metrics)
            if run is not None:
                run.log(
                    {
                        "eval_seq_len": seq_len,
                        "extrapolation/loss": metrics["loss"],
                        "extrapolation/perplexity": metrics["perplexity"],
                        "extrapolation/total_tokens": metrics["total_tokens"],
                        "extrapolation/eval_time_sec": metrics["eval_time_sec"],
                    },
                    step=seq_len,
                )

            print(
                f"perplexity={metrics['perplexity']:.2f}  "
                f"loss={metrics['loss']:.4f}  "
                f"time={elapsed:.1f}s"
            )
        except Exception as e:
            print(f"FAILED: {e}")
            results["evaluations"].append(
                {"seq_len": seq_len, "error": str(e)}
            )

    # Print summary table
    print()
    print(f"{'Seq Len':>8}  {'Perplexity':>12}  {'Loss':>8}  {'Time (s)':>10}")
    print("-" * 45)
    for ev in results["evaluations"]:
        if "error" in ev:
            print(f"{ev['seq_len']:>8}  {'ERROR':>12}  {'-':>8}  {'-':>10}")
        else:
            print(
                f"{ev['seq_len']:>8}  {ev['perplexity']:>12.2f}  "
                f"{ev['loss']:>8.4f}  {ev['eval_time_sec']:>10.1f}"
            )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"extrapolation_{pe_type}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
