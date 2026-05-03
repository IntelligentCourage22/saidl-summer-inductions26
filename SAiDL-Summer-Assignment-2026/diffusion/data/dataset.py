from pathlib import Path
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(root):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def split_image_paths(
    root,
    val_fraction=0.1,
    max_train_images=None,
    max_val_images=None,
    seed=42,
):
    paths = list_images(root)
    if not paths:
        raise ValueError(f"No images found under {root}")

    rng = random.Random(seed)
    rng.shuffle(paths)

    val_count = max(1, int(len(paths) * val_fraction))
    val_paths = paths[:val_count]
    train_paths = paths[val_count:]

    if max_train_images:
        train_paths = train_paths[:max_train_images]
    if max_val_images:
        val_paths = val_paths[:max_val_images]

    return train_paths, val_paths


class LandscapeDataset(Dataset):
    def __init__(self, image_paths, image_size=256, train=True):
        self.image_paths = list(image_paths)
        if train:
            transform_steps = [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.85, 1.0),
                    ratio=(0.9, 1.1),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.1,
                            contrast=0.1,
                            saturation=0.1,
                            hue=0.02,
                        )
                    ],
                    p=0.5,
                ),
                transforms.RandomRotation(
                    degrees=5,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
            ]
        else:
            transform_steps = [
                transforms.Resize(
                    image_size, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(image_size),
            ]
        transform_steps.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.transform = transforms.Compose(transform_steps)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        return {"image": self.transform(image), "path": str(path)}
