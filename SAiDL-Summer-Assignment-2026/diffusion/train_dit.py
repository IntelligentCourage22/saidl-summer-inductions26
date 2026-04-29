import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys_path = os.path.dirname(os.path.abspath(__file__))
import sys

sys.path.insert(0, sys_path)

from data.dataset import LandscapeDataset, split_image_paths
from diffusion.gaussian_diffusion import GaussianDiffusion
from models.dit import LatentDiT
from models.vae import FrozenVAE
from utils import EMAModel, append_jsonl, ensure_dir, load_config, maybe_init_wandb, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(sys_path, "configs", "dit_landscape.yaml"))
    parser.add_argument("--set", action="append", default=[])
    return parser.parse_args()


def save_checkpoint(path, model, ema, optimizer, scaler, step, epoch, config):
    ensure_dir(Path(path).parent)
    torch.save(
        {
            "model": model.state_dict(),
            "model_ema": ema.state_dict() if ema is not None else None,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "step": step,
            "epoch": epoch,
            "config": config,
        },
        path,
    )


def main():
    args = parse_args()
    config, cfg = load_config(args.config, args.set)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = ensure_dir(cfg.training.output_dir)
    checkpoint_dir = ensure_dir(cfg.training.checkpoint_dir)

    train_paths, _ = split_image_paths(
        cfg.data.root,
        cfg.data.val_fraction,
        cfg.data.max_train_images,
        cfg.data.max_val_images,
        cfg.seed,
    )
    train_ds = LandscapeDataset(train_paths, cfg.data.image_size)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    vae = FrozenVAE(cfg.vae.model_name, cfg.vae.scaling_factor).to(device)
    model = LatentDiT(**vars(cfg.model)).to(device)
    diffusion = GaussianDiffusion(
        cfg.diffusion.num_timesteps,
        cfg.diffusion.beta_start,
        cfg.diffusion.beta_end,
        device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
    )
    ema = EMAModel(model, float(cfg.training.ema_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.training.amp) and torch.cuda.is_available())
    run = maybe_init_wandb(config)

    global_step = 0
    start = time.time()
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(cfg.training.num_epochs):
        progress = tqdm(train_loader, desc=f"epoch {epoch}")
        for batch_idx, batch in enumerate(progress):
            images = batch["image"].to(device, non_blocking=True)
            with torch.no_grad():
                latents = vae.encode(images)

            with torch.cuda.amp.autocast(enabled=bool(cfg.training.amp) and torch.cuda.is_available()):
                loss = diffusion.training_loss(model, latents) / cfg.training.grad_accum_steps

            scaler.scale(loss).backward()
            if (batch_idx + 1) % cfg.training.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.training.grad_clip))
                scaler.step(optimizer)
                scaler.update()
                ema.update(model)
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                loss_value = float(loss.detach().cpu()) * cfg.training.grad_accum_steps
                progress.set_postfix(loss=f"{loss_value:.4f}")

                if global_step % cfg.training.log_every == 0:
                    record = {
                        "step": global_step,
                        "epoch": epoch,
                        "loss": loss_value,
                        "elapsed_sec": time.time() - start,
                        "lr": optimizer.param_groups[0]["lr"],
                        "images_seen": global_step
                        * cfg.training.batch_size
                        * cfg.training.grad_accum_steps,
                    }
                    append_jsonl(output_dir / "metrics.jsonl", record)
                    if run is not None:
                        run.log(record, step=global_step)

                if global_step % cfg.training.save_every == 0:
                    save_checkpoint(
                        checkpoint_dir / f"step_{global_step}.pt",
                        model,
                        ema,
                        optimizer,
                        scaler,
                        global_step,
                        epoch,
                        config,
                    )

                if cfg.training.max_steps and global_step >= cfg.training.max_steps:
                    break

        if cfg.training.max_steps and global_step >= cfg.training.max_steps:
            break

    save_checkpoint(
        checkpoint_dir / "final.pt", model, ema, optimizer, scaler, global_step, epoch, config
    )
    with open(output_dir / "final_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "step": global_step,
                "epoch": epoch,
                "elapsed_sec": time.time() - start,
                "num_train_images": len(train_ds),
            },
            handle,
            indent=2,
        )
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
