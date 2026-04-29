import argparse
import os
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision.utils import save_image

sys_path = os.path.dirname(os.path.abspath(__file__))
import sys

sys.path.insert(0, sys_path)

from data.dataset import LandscapeDataset, split_image_paths
from utils import ensure_dir, load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(sys_path, "configs", "dit_landscape.yaml"))
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--output-dir", default="results/diffusion/real_val")
    parser.add_argument("--max-images", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    _, cfg = load_config(args.config, args.set)
    out_dir = ensure_dir(args.output_dir)
    _, val_paths = split_image_paths(
        cfg.data.root,
        cfg.data.val_fraction,
        cfg.data.max_train_images,
        args.max_images,
        cfg.seed,
    )
    dataset = LandscapeDataset(val_paths, cfg.data.image_size, train=False)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=cfg.data.num_workers)

    idx = 0
    for batch in loader:
        images = (batch["image"] + 1) / 2
        for image in images:
            save_image(image, Path(out_dir) / f"real_{idx:05d}.png")
            idx += 1


if __name__ == "__main__":
    main()
