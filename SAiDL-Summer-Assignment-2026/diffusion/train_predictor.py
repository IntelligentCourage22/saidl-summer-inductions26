import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

sys_path = os.path.dirname(os.path.abspath(__file__))
import sys

sys.path.insert(0, sys_path)

from models.difficulty_predictor import DifficultyPredictor
from utils import append_jsonl, ensure_dir, load_config, maybe_init_wandb, set_seed, torch_load


class SupervisionDataset(Dataset):
    def __init__(self, root):
        self.files = sorted(Path(root).glob("batch_*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No supervision batches found under {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch_load(self.files[idx], map_location="cpu")


def collate_batches(records):
    features = torch.cat([record["features"].float() for record in records], dim=0)
    targets = torch.cat([record["target"].float() for record in records], dim=0)
    return features, targets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=os.path.join(sys_path, "configs", "dit_landscape.yaml")
    )
    parser.add_argument("--set", action="append", default=[])
    return parser.parse_args()


@torch.no_grad()
def evaluate(predictor, loader, device):
    predictor.eval()
    losses = []
    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        pred = predictor(features)
        losses.append(float(F.mse_loss(pred, targets).detach().cpu()))
    predictor.train()
    return sum(losses) / len(losses) if losses else None


def main():
    args = parse_args()
    config, cfg = load_config(args.config, args.set)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = ensure_dir(cfg.predictor.output_dir)

    dataset = SupervisionDataset(cfg.predictor.supervision_dir)
    val_len = int(len(dataset) * float(getattr(cfg.predictor, "val_fraction", 0.1)))
    val_len = min(max(val_len, 1), max(len(dataset) - 1, 0))
    train_len = len(dataset) - val_len
    generator = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=generator)
    loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_batches)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_batches)
    token_grid = cfg.model.latent_size // cfg.model.patch_size
    predictor = DifficultyPredictor(
        cfg.model.hidden_size, token_grid, cfg.model.latent_size
    ).to(device)
    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=float(cfg.predictor.learning_rate),
        weight_decay=float(getattr(cfg.predictor, "weight_decay", 0.0)),
    )
    run = maybe_init_wandb(config)

    step = 0
    start = time.time()
    for epoch in range(cfg.predictor.num_epochs):
        progress = tqdm(loader, desc=f"predictor epoch {epoch}")
        for features, targets in progress:
            features = features.to(device)
            targets = targets.to(device)
            pred = predictor(features)
            loss = F.mse_loss(pred, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1

            if step % 50 == 0:
                record = {
                    "step": step,
                    "epoch": epoch,
                    "predictor_loss": float(loss.detach().cpu()),
                    "elapsed_sec": time.time() - start,
                }
                append_jsonl(output_dir / "metrics.jsonl", record)
                if run is not None:
                    run.log(record, step=step)

            val_every = getattr(cfg.predictor, "val_every", 100)
            if val_every and step % val_every == 0:
                val_loss = evaluate(predictor, val_loader, device)
                if val_loss is not None:
                    record = {
                        "step": step,
                        "epoch": epoch,
                        "predictor_val_loss": val_loss,
                        "elapsed_sec": time.time() - start,
                    }
                    append_jsonl(output_dir / "metrics.jsonl", record)
                    if run is not None:
                        run.log(record, step=step)

            if cfg.predictor.max_steps and step >= cfg.predictor.max_steps:
                break
        if cfg.predictor.max_steps and step >= cfg.predictor.max_steps:
            break

    checkpoint = {
        "predictor": predictor.state_dict(),
        "config": config,
        "step": step,
    }
    torch.save(checkpoint, output_dir / "difficulty_predictor.pt")
    with open(output_dir / "final_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "step": step,
                "num_train_supervision_batches": len(train_ds),
                "num_val_supervision_batches": len(val_ds),
            },
            handle,
            indent=2,
        )
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
