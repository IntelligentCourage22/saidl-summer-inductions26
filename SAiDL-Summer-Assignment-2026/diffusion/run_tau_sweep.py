import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

sys_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, sys_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--predictor-checkpoint", required=True)
    parser.add_argument("--real-dir", required=True)
    parser.add_argument(
        "--config", default=os.path.join(sys_path, "configs", "dit_landscape.yaml")
    )
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--output-dir", default="results/diffusion/tau_sweep")
    parser.add_argument(
        "--taus",
        nargs="+",
        type=float,
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    parser.add_argument("--num-images", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--reuse-samples", action="store_true")
    parser.add_argument("--skip-fid", action="store_true")
    parser.add_argument("--skip-cmmd", action="store_true")
    return parser.parse_args()


def run_command(command):
    print(" ".join(str(part) for part in command), flush=True)
    subprocess.run(command, check=True)


def tau_label(tau):
    return f"{tau:.2f}".rstrip("0").rstrip(".").replace(".", "p")


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def sample_dir_for(output_dir, mode, tau=None):
    if mode == "racd":
        return output_dir / "samples" / f"racd_tau_{tau_label(tau)}" / "racd"
    return output_dir / "samples" / mode


def run_sampling(args, mode, generated_dir, tau=None):
    if args.reuse_samples and generated_dir.exists() and any(generated_dir.glob("*.png")):
        return

    sample_root = generated_dir.parent if mode == "racd" else generated_dir.parent
    command = [
        sys.executable,
        os.path.join(sys_path, "sample.py"),
        "--checkpoint",
        args.checkpoint,
        "--mode",
        mode,
        "--config",
        args.config,
        "--set",
        f"sampling.output_dir={sample_root.as_posix()}",
    ]
    if args.num_images is not None:
        command.extend(["--set", f"sampling.num_images={args.num_images}"])
    if args.batch_size is not None:
        command.extend(["--set", f"sampling.batch_size={args.batch_size}"])
    for override in args.set:
        if override.startswith("sampling.output_dir="):
            continue
        command.extend(["--set", override])
    if mode == "racd":
        command.extend(
            [
                "--predictor-checkpoint",
                args.predictor_checkpoint,
                "--tau",
                str(tau),
            ]
        )
    run_command(command)


def run_evaluation(args, method_name, generated_dir, eval_dir):
    output = eval_dir / f"{method_name}.json"
    if args.reuse_samples and output.exists():
        with open(output, "r", encoding="utf-8") as handle:
            return json.load(handle)

    command = [
        sys.executable,
        os.path.join(sys_path, "evaluate.py"),
        "--real-dir",
        args.real_dir,
        "--generated-dir",
        str(generated_dir),
        "--output",
        str(output),
        "--batch-size",
        str(args.eval_batch_size),
    ]
    if args.skip_fid:
        command.append("--skip-fid")
    if args.skip_cmmd:
        command.append("--skip-cmmd")
    run_command(command)
    with open(output, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_sampling_summary(generated_dir):
    summary_path = generated_dir / "sampling_summary.json"
    if not summary_path.exists():
        return {}
    with open(summary_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path, records):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "tau",
                "fid",
                "cmmd",
                "mean_time_sec_per_image",
                "mean_refinement_fraction",
                "generated_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(records)


def choose_optimal_tau(records):
    racd = [
        record
        for record in records
        if record["method"] == "racd" and record.get("cmmd") is not None
    ]
    if not racd:
        return None

    cmmd_values = [record["cmmd"] for record in racd]
    time_values = [record["mean_time_sec_per_image"] for record in racd]
    cmmd_min, cmmd_max = min(cmmd_values), max(cmmd_values)
    time_min, time_max = min(time_values), max(time_values)

    def normalized_score(record):
        cmmd_span = max(cmmd_max - cmmd_min, 1e-12)
        time_span = max(time_max - time_min, 1e-12)
        cmmd_score = (record["cmmd"] - cmmd_min) / cmmd_span
        time_score = (record["mean_time_sec_per_image"] - time_min) / time_span
        return cmmd_score + time_score

    return min(racd, key=normalized_score)


def plot_tradeoff(path, records, optimal):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 5))
    for method, marker in [("baseline", "o"), ("global_cyclic", "s"), ("racd", "^")]:
        subset = [
            record
            for record in records
            if record["method"] == method and record.get("cmmd") is not None
        ]
        if not subset:
            continue
        times = [record["mean_time_sec_per_image"] for record in subset]
        cmmds = [record["cmmd"] for record in subset]
        plt.scatter(times, cmmds, marker=marker, label=method)
        for record in subset:
            if record["tau"] is not None:
                plt.annotate(
                    f"tau={record['tau']:.2f}",
                    (record["mean_time_sec_per_image"], record["cmmd"]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                )

    if optimal is not None:
        plt.scatter(
            [optimal["mean_time_sec_per_image"]],
            [optimal["cmmd"]],
            facecolors="none",
            edgecolors="black",
            s=140,
            linewidths=1.5,
            label="selected tau",
        )

    plt.xlabel("Mean generation time (s/image)")
    plt.ylabel("CMMD (lower is better)")
    plt.title("Fidelity vs Compute for RACD Threshold Sweep")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)


def main():
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    eval_dir = ensure_dir(output_dir / "eval")

    records = []
    methods = [("baseline", None), ("global_cyclic", None)]
    methods.extend(("racd", tau) for tau in args.taus)

    for mode, tau in methods:
        method_name = mode if tau is None else f"racd_tau_{tau_label(tau)}"
        generated_dir = sample_dir_for(output_dir, mode, tau)
        run_sampling(args, mode, generated_dir, tau)
        metrics = run_evaluation(args, method_name, generated_dir, eval_dir)
        summary = read_sampling_summary(generated_dir)

        records.append(
            {
                "method": mode,
                "tau": tau,
                "fid": metrics.get("fid"),
                "cmmd": metrics.get("cmmd"),
                "mean_time_sec_per_image": summary.get("mean_time_sec_per_image"),
                "mean_refinement_fraction": summary.get("mean_refinement_fraction"),
                "generated_dir": str(generated_dir),
            }
        )

    optimal = choose_optimal_tau(records)
    result = {
        "records": records,
        "optimal_tau": optimal["tau"] if optimal is not None else None,
        "selection_rule": "minimum normalized CMMD plus normalized generation time among RACD thresholds",
    }

    with open(output_dir / "tau_sweep_results.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    write_csv(output_dir / "tau_sweep_results.csv", records)
    if not args.skip_cmmd:
        plot_tradeoff(output_dir / "cmmd_vs_time.png", records, optimal)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
