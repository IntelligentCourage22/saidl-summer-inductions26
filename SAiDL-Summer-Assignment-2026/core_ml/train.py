import argparse
import json
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from core_ml.data.dataset import get_dataloaders
from core_ml.models.model import TransformerLM
from core_ml.utils.metrics import MetricsTracker, compute_grad_norm


def to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: to_namespace(val) for key, val in value.items()})
    if isinstance(value, list):
        return [to_namespace(item) for item in value]
    return value


def set_nested(config, dotted_key, value):
    current = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="core_ml/configs/config.yaml")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config values, e.g. --set model.attention_type=gqa",
    )
    return parser.parse_args()


def load_config(path, overrides):
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override {override!r}; expected key=value.")
        key, raw_value = override.split("=", 1)
        set_nested(config, key, yaml.safe_load(raw_value))

    return config


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    model.eval()
    tracker = MetricsTracker()

    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        _, loss = model(x, y)
        tracker.update(loss.item(), x.numel())

    return tracker.get_summary()


def save_checkpoint(path, model, optimizer, scheduler, step, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "config": config,
        },
        path,
    )


def maybe_init_wandb(config):
    if not config.get("wandb", {}).get("enabled", False):
        return None

    import wandb

    return wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"].get("entity"),
        config=config,
    )


def build_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return max(1, step) / warmup_steps

        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    args = parse_args()
    config = load_config(args.config, args.set)
    cfg = to_namespace(config)

    set_seed(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg.training.output_dir)
    checkpoint_dir = Path(cfg.training.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders(
        seq_len=cfg.training.seq_len,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        dataset_name=cfg.data.dataset_name,
        dataset_config=cfg.data.dataset_config,
        tokenizer_name=cfg.data.tokenizer_name,
    )

    model = TransformerLM(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.learning_rate, betas=(0.9, 0.95)
    )
    total_steps = cfg.training.max_steps or (cfg.training.num_epochs * len(train_loader))
    scheduler = build_scheduler(optimizer, cfg.training.warmup_steps, total_steps)
    run = maybe_init_wandb(config)

    global_step = 0
    train_tracker = MetricsTracker()
    start_time = time.time()
    metrics_path = output_dir / "metrics.jsonl"

    with open(metrics_path, "a", encoding="utf-8") as metrics_file:
        for epoch in range(cfg.training.num_epochs):
            model.train()
            for x, y in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                _, loss = model(x, y)
                loss.backward()
                grad_norm = compute_grad_norm(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
                optimizer.step()
                scheduler.step()

                global_step += 1
                train_tracker.update(loss.item(), x.numel())

                if global_step % cfg.training.eval_every == 0:
                    train_summary = train_tracker.get_summary()
                    val_summary = evaluate(
                        model, val_loader, device, cfg.training.eval_batches
                    )
                    record = {
                        "step": global_step,
                        "epoch": epoch,
                        "grad_norm": grad_norm,
                        "lr": scheduler.get_last_lr()[0],
                        "elapsed_sec": time.time() - start_time,
                        "train": train_summary,
                        "validation": val_summary,
                        "model": {
                            "attention_type": cfg.model.attention_type,
                            "positional_encoding": cfg.model.positional_encoding,
                            "block_type": cfg.model.block_type,
                        },
                    }
                    metrics_file.write(json.dumps(record) + "\n")
                    metrics_file.flush()
                    if run is not None:
                        run.log(record, step=global_step)
                    train_tracker.reset()
                    model.train()

                if global_step % cfg.training.save_every == 0:
                    save_checkpoint(
                        checkpoint_dir / f"step_{global_step}.pt",
                        model,
                        optimizer,
                        scheduler,
                        global_step,
                        config,
                    )

                if cfg.training.max_steps and global_step >= cfg.training.max_steps:
                    break

            if cfg.training.max_steps and global_step >= cfg.training.max_steps:
                break

    final_val = evaluate(model, val_loader, device, cfg.training.eval_batches)
    final_test = evaluate(model, test_loader, device, cfg.training.eval_batches)
    final_record = {"validation": final_val, "test": final_test, "step": global_step}
    with open(output_dir / "final_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(final_record, handle, indent=2)

    save_checkpoint(
        checkpoint_dir / "final.pt", model, optimizer, scheduler, global_step, config
    )
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
