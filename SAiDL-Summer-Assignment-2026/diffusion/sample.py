import argparse
import json
import os
from pathlib import Path

import torch

sys_path = os.path.dirname(os.path.abspath(__file__))
import sys

sys.path.insert(0, sys_path)

from diffusion.gaussian_diffusion import GaussianDiffusion
from diffusion.samplers import (
    Timer,
    global_cyclic_refine,
    masked_cyclic_refine,
    sample_loop,
    save_decoded_batch,
)
from models.difficulty_predictor import DifficultyPredictor
from models.dit import LatentDiT
from models.vae import FrozenVAE
from utils import EMAModel, ensure_dir, load_config, set_seed, torch_load


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=["baseline", "global_cyclic", "racd"], default="baseline")
    parser.add_argument("--predictor-checkpoint", default=None)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--config", default=os.path.join(sys_path, "configs", "dit_landscape.yaml"))
    parser.add_argument("--set", action="append", default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    config, cfg = load_config(args.config, args.set)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ensure_dir(Path(cfg.sampling.output_dir) / args.mode)

    checkpoint = torch_load(args.checkpoint, map_location=device)
    model_cfg = checkpoint.get("config", config)["model"]
    model = LatentDiT(**model_cfg).to(device)
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
    )
    predictor = None
    if args.mode == "racd":
        if args.predictor_checkpoint is None:
            raise ValueError("--predictor-checkpoint is required for RACD sampling.")
        pred_ckpt = torch_load(args.predictor_checkpoint, map_location=device)
        token_grid = cfg.model.latent_size // cfg.model.patch_size
        predictor = DifficultyPredictor(cfg.model.hidden_size, token_grid, cfg.model.latent_size).to(device)
        predictor.load_state_dict(pred_ckpt["predictor"])
        predictor.eval()

    generated = 0
    times = []
    refinement_fracs = []
    while generated < cfg.sampling.num_images:
        batch_size = min(cfg.sampling.batch_size, cfg.sampling.num_images - generated)
        shape = (batch_size, cfg.model.in_channels, cfg.model.latent_size, cfg.model.latent_size)
        with Timer() as timer:
            if args.mode == "racd":
                z0, features = sample_loop(
                    model, diffusion, shape, device, cfg.sampling.ddim_steps, return_features=True
                )
                difficulty = predictor(features)
                mask = (difficulty > args.tau).float()
                refinement_fracs.append(float(mask.mean().detach().cpu()))
                z0 = masked_cyclic_refine(
                    model, diffusion, z0, cfg.sampling.cyclic_t_start, mask, device
                )
            else:
                z0 = sample_loop(model, diffusion, shape, device, cfg.sampling.ddim_steps)
            if args.mode == "global_cyclic":
                z0 = global_cyclic_refine(model, diffusion, z0, cfg.sampling.cyclic_t_start, device)
        times.append(timer.elapsed / batch_size)
        save_decoded_batch(vae, z0, out_dir, args.mode, generated)
        generated += batch_size

    summary = {
        "mode": args.mode,
        "num_images": cfg.sampling.num_images,
        "mean_time_sec_per_image": sum(times) / len(times),
        "tau": args.tau if args.mode == "racd" else None,
        "mean_refinement_fraction": sum(refinement_fracs) / len(refinement_fracs)
        if refinement_fracs
        else None,
    }
    with open(out_dir / "sampling_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
