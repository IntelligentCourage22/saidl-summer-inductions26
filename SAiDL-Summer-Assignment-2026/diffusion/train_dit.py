import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

sys_path = os.path.dirname(os.path.abspath(__file__))
import sys

sys.path.insert(0, sys_path)

from data.dataset import LandscapeDataset, split_image_paths
from diffusion.gaussian_diffusion import GaussianDiffusion
from diffusion.samplers import sample_loop
from models.dit import LatentDiT
from models.vae import FrozenVAE
from utils import (
    EMAModel,
    append_jsonl,
    ensure_dir,
    load_config,
    maybe_init_wandb,
    set_seed,
    torch_load,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=os.path.join(sys_path, "configs", "dit_landscape.yaml")
    )
    parser.add_argument("--set", action="append", default=[])
    return parser.parse_args()


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def save_checkpoint(path, model, ema, optimizer, scheduler, scaler, step, epoch, config):
    ensure_dir(Path(path).parent)
    raw_model = unwrap_model(model)
    torch.save(
        {
            "model": raw_model.state_dict(),
            "model_ema": ema.state_dict() if ema is not None else None,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "step": step,
            "epoch": epoch,
            "config": config,
        },
        path,
    )


@torch.no_grad()
def save_training_sample_grid(model, ema, vae, diffusion, cfg, device, step):
    raw_model = unwrap_model(model)
    sample_dir = ensure_dir(getattr(cfg.training, "sample_output_dir", "results/diffusion/training_samples"))
    num_images = int(getattr(cfg.training, "sample_num_images", 16))
    batch_size = int(getattr(cfg.training, "sample_batch_size", cfg.sampling.batch_size))
    ddim_steps = int(getattr(cfg.training, "sample_ddim_steps", cfg.sampling.ddim_steps))
    image_batches = []
    use_ema = bool(getattr(cfg.training, "sample_use_ema", False))
    original_state = {
        name: value.detach().cpu().clone()
        for name, value in raw_model.state_dict().items()
        if torch.is_floating_point(value)
    }

    try:
        if use_ema and ema is not None:
            ema.copy_to(raw_model)
        raw_model.eval()
        generated = 0
        while generated < num_images:
            current_batch = min(batch_size, num_images - generated)
            shape = (
                current_batch,
                cfg.model.in_channels,
                cfg.model.latent_size,
                cfg.model.latent_size,
            )
            latents = sample_loop(raw_model, diffusion, shape, device, ddim_steps)
            images = ((vae.decode(latents).detach().cpu() + 1) / 2).clamp(0, 1)
            image_batches.append(images)
            generated += current_batch
    finally:
        state = raw_model.state_dict()
        for name, value in original_state.items():
            state[name].copy_(value.to(device=state[name].device, dtype=state[name].dtype))
        raw_model.load_state_dict(state)
        model.train()

    images = torch.cat(image_batches, dim=0)
    nrow = min(4, num_images)
    grid = make_grid(images, nrow=nrow)
    path = sample_dir / f"step_{step:06d}.png"
    save_image(grid, path)
    return path


def log_checkpoint_artifact(run, cfg, checkpoint_path, step, sample_path=None):
    if run is None:
        return

    import wandb

    artifact_name = getattr(
        cfg.training, "artifact_name", "diffusion-baseline-dit-checkpoints"
    )
    artifact = wandb.Artifact(artifact_name, type="model")
    artifact.add_file(str(checkpoint_path), name=Path(checkpoint_path).name)
    if sample_path is not None and Path(sample_path).exists():
        artifact.add_file(str(sample_path), name=f"samples/{Path(sample_path).name}")

    logged = run.log_artifact(artifact, aliases=["latest", f"step-{step}"])
    if bool(getattr(cfg.training, "artifact_wait", True)):
        logged.wait()


def build_lr_scheduler(optimizer, max_steps, warmup_steps, min_lr_ratio=0.0):
    max_steps = max(1, int(max_steps or 1))
    warmup_steps = int(warmup_steps or 0)
    min_lr_ratio = float(min_lr_ratio)

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-8)
        decay_steps = max(1, max_steps - warmup_steps)
        progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate_validation_loss(
    model, vae, diffusion, loader, device, amp_enabled, max_batches=None
):
    model.eval()
    losses = []
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        latents = vae.encode(images)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            loss = diffusion.training_loss(model, latents)
        losses.append(float(loss.detach().cpu()))
    model.train()
    return sum(losses) / len(losses) if losses else None


def main():
    args = parse_args()
    config, cfg = load_config(args.config, args.set)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = ensure_dir(cfg.training.output_dir)
    checkpoint_dir = ensure_dir(cfg.training.checkpoint_dir)

    train_paths, val_paths = split_image_paths(
        cfg.data.root,
        cfg.data.val_fraction,
        cfg.data.max_train_images,
        cfg.data.max_val_images,
        cfg.seed,
    )
    train_ds = LandscapeDataset(
        train_paths,
        cfg.data.image_size,
        train=True,
        augment=bool(getattr(cfg.data, "train_augment", True)),
    )
    val_ds = LandscapeDataset(val_paths, cfg.data.image_size, train=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    vae = FrozenVAE(cfg.vae.model_name, cfg.vae.scaling_factor).to(device)
    model = LatentDiT(**vars(cfg.model)).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = torch.nn.DataParallel(model)
    diffusion = GaussianDiffusion(
        cfg.diffusion.num_timesteps,
        cfg.diffusion.beta_start,
        cfg.diffusion.beta_end,
        device,
        getattr(cfg.diffusion, "prediction_target", "epsilon"),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
    )
    min_lr_ratio = float(getattr(cfg.training, "min_learning_rate", 0.0)) / float(
        cfg.training.learning_rate
    )
    steps_per_epoch = max(1, len(train_loader) // int(cfg.training.grad_accum_steps))
    target_steps = cfg.training.max_steps or steps_per_epoch * int(
        cfg.training.num_epochs
    )
    scheduler = build_lr_scheduler(
        optimizer,
        target_steps,
        getattr(cfg.training, "warmup_steps", 0),
        min_lr_ratio,
    )
    ema = EMAModel(unwrap_model(model), float(cfg.training.ema_decay))
    amp_enabled = bool(cfg.training.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    run = maybe_init_wandb(config)

    global_step = 0
    start_epoch = 0
    if cfg.training.resume:
        checkpoint = torch_load(cfg.training.resume, map_location=device)
        unwrap_model(model).load_state_dict(checkpoint["model"])
        if checkpoint.get("model_ema") is not None:
            ema.load_state_dict(checkpoint["model_ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        global_step = int(checkpoint.get("step", 0))
        start_epoch = int(checkpoint.get("epoch", 0))

    start = time.time()
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, cfg.training.num_epochs):
        progress = tqdm(train_loader, desc=f"epoch {epoch}")
        for batch_idx, batch in enumerate(progress):
            images = batch["image"].to(device, non_blocking=True)
            with torch.no_grad():
                latents = vae.encode(images)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                loss = (
                    diffusion.training_loss(model, latents)
                    / cfg.training.grad_accum_steps
                )

            scaler.scale(loss).backward()
            if (batch_idx + 1) % cfg.training.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.training.grad_clip))
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                ema.update(unwrap_model(model))
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

                val_every = getattr(cfg.training, "val_every", cfg.training.save_every)
                if val_every and global_step % val_every == 0:
                    val_loss = evaluate_validation_loss(
                        model,
                        vae,
                        diffusion,
                        val_loader,
                        device,
                        amp_enabled,
                        getattr(cfg.training, "max_val_batches", None),
                    )
                    if val_loss is not None:
                        record = {
                            "step": global_step,
                            "epoch": epoch,
                            "val_loss": val_loss,
                            "elapsed_sec": time.time() - start,
                        }
                        append_jsonl(output_dir / "metrics.jsonl", record)
                        if run is not None:
                            run.log(record, step=global_step)

                checkpoint_path = None
                if global_step % cfg.training.save_every == 0:
                    checkpoint_path = checkpoint_dir / f"step_{global_step}.pt"
                    save_checkpoint(
                        checkpoint_path,
                        model,
                        ema,
                        optimizer,
                        scheduler,
                        scaler,
                        global_step,
                        epoch,
                        config,
                    )

                sample_every = getattr(cfg.training, "sample_every", 0)
                sample_path = None
                if sample_every and global_step % sample_every == 0:
                    sample_path = save_training_sample_grid(
                        model,
                        ema,
                        vae,
                        diffusion,
                        cfg,
                        device,
                        global_step,
                    )

                artifact_every = getattr(cfg.training, "artifact_every", 0)
                if artifact_every and global_step % artifact_every == 0:
                    if checkpoint_path is None:
                        checkpoint_path = checkpoint_dir / f"step_{global_step}.pt"
                        save_checkpoint(
                            checkpoint_path,
                            model,
                            ema,
                            optimizer,
                            scheduler,
                            scaler,
                            global_step,
                            epoch,
                            config,
                        )
                    log_checkpoint_artifact(
                        run, cfg, checkpoint_path, global_step, sample_path
                    )

                if cfg.training.max_steps and global_step >= cfg.training.max_steps:
                    break

        if cfg.training.max_steps and global_step >= cfg.training.max_steps:
            break

    final_checkpoint_path = checkpoint_dir / "final.pt"
    save_checkpoint(
        final_checkpoint_path,
        model,
        ema,
        optimizer,
        scheduler,
        scaler,
        global_step,
        epoch,
        config,
    )
    final_sample_path = None
    if getattr(cfg.training, "sample_every", 0):
        final_sample_path = save_training_sample_grid(
            model,
            ema,
            vae,
            diffusion,
            cfg,
            device,
            global_step,
        )
    if getattr(cfg.training, "artifact_every", 0):
        log_checkpoint_artifact(
            run, cfg, final_checkpoint_path, global_step, final_sample_path
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
