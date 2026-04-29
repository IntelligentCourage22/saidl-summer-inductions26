from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor


class ImageFolderDataset(Dataset):
    def __init__(self, root):
        self.paths = sorted(
            path
            for path in Path(root).rglob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found under {root}")
        self.to_pil = transforms.Lambda(lambda x: x)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return Image.open(self.paths[idx]).convert("RGB")


def pil_collate(batch):
    return batch


@torch.no_grad()
def encode_folder(root, batch_size=16, device="cuda", model_name="openai/clip-vit-base-patch32"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    loader = DataLoader(
        ImageFolderDataset(root), batch_size=batch_size, shuffle=False, collate_fn=pil_collate
    )
    embeddings = []
    for images in loader:
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        feats = model.get_image_features(**inputs)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        embeddings.append(feats.cpu())
    return torch.cat(embeddings, dim=0)


def rbf_kernel(x, y, gamma=None):
    distances = torch.cdist(x, y).pow(2)
    if gamma is None:
        gamma = 1.0 / x.shape[1]
    return torch.exp(-gamma * distances)


def compute_cmmd(real_dir, generated_dir, batch_size=16, device="cuda"):
    real = encode_folder(real_dir, batch_size, device)
    generated = encode_folder(generated_dir, batch_size, device)
    k_rr = rbf_kernel(real, real).mean()
    k_gg = rbf_kernel(generated, generated).mean()
    k_rg = rbf_kernel(real, generated).mean()
    return float((k_rr + k_gg - 2 * k_rg).item())
