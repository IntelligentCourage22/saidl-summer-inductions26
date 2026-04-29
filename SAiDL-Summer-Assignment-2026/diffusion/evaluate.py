import argparse
import json
import os
from pathlib import Path

sys_path = os.path.dirname(os.path.abspath(__file__))
import sys

sys.path.insert(0, sys_path)

from evaluation.cmmd import compute_cmmd
from evaluation.fid import compute_fid
from utils import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--generated-dir", required=True)
    parser.add_argument("--output", default="results/diffusion/eval_metrics.json")
    parser.add_argument("--skip-fid", action="store_true")
    parser.add_argument("--skip-cmmd", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = {
        "real_dir": args.real_dir,
        "generated_dir": args.generated_dir,
    }
    if not args.skip_fid:
        metrics["fid"] = compute_fid(args.real_dir, args.generated_dir)
    if not args.skip_cmmd:
        metrics["cmmd"] = compute_cmmd(args.real_dir, args.generated_dir, args.batch_size)

    output = Path(args.output)
    ensure_dir(output.parent)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
