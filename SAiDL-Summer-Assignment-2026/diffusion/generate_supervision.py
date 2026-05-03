import argparse
import os
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
from utils import EMAModel, ensure_dir, load_config, set_seed, torch_load


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=os.path.join(sys_path, "configs", "dit_landscape.yaml"))
    parser.add_argument("--set", action="append", default=[])
    return parser.parse_args()


def normalize_error_map(error):
    flat = error.flatten(1)
    mins = flat.min(dim=1).values[:, None, None, None]
    maxs = flat.max(dim=1).values[:, None, None, None]
    return (error - mins) / (maxs - mins).clamp(min=1e-8)


@torch.no_grad()
def main():
    args = parse_args()
    config, cfg = load_config(args.config, args.set)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ensure_dir(cfg.predictor.supervision_dir)

    train_paths, _ = split_image_paths(
        cfg.data.root,
        cfg.data.val_fraction,
        cfg.data.max_train_images,
        cfg.data.max_val_images,
        cfg.seed,
    )
    dataset = LandscapeDataset(train_paths, cfg.data.image_size)
    loader = DataLoader(
        dataset,
        batch_size=cfg.predictor.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    checkpoint = torch_load(args.checkpoint, map_location=device)
    model = LatentDiT(**checkpoint.get("config", config)["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    if checkpoint.get("model_ema") is not None:
        ema = EMAModel(model)
        ema.load_state_dict(checkpoint["model_ema"])
        ema.copy_to(model)
    model.eval()

    vae = FrozenVAE(cfg.vae.model_name, cfg.vae.scaling_factor).to(device)
    diffusion = GaussianDiffusion(
        cfg.diffusion.num_timesteps,
        cfg.diffusion.beta_start,
        cfg.diffusion.beta_end,
        device,
        getattr(cfg.diffusion, "prediction_target", "epsilon"),
    )

    t_mid = int(cfg.predictor.t_mid)
    sample_idx = 0
    for batch in tqdm(loader, desc="supervision"):
        images = batch["image"].to(device, non_blocking=True)
        z0 = vae.encode(images)
        timesteps = torch.full((z0.shape[0],), t_mid, device=device, dtype=torch.long)
        noise = torch.randn_like(z0)
        z_t = diffusion.q_sample(z0, timesteps, noise)
        pred_noise, features = model(z_t, timesteps, return_features=True)
        x0_hat = diffusion.predict_x0_from_eps(z_t, timesteps, pred_noise)
        error = (x0_hat - z0).pow(2).mean(dim=1, keepdim=True)
        target = normalize_error_map(error)

        record = {
            "features": features.detach().cpu().half(),
            "target": target.detach().cpu().half(),
            "timestep": t_mid,
        }
        torch.save(record, Path(out_dir) / f"batch_{sample_idx:05d}.pt")
        sample_idx += 1


if __name__ == "__main__":
    main()
