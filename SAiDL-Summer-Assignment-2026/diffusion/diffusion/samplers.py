import time
from pathlib import Path

import torch
from torchvision.utils import save_image


@torch.no_grad()
def sample_loop(model, diffusion, shape, device, steps=None, return_features=False):
    model.eval()
    x = torch.randn(shape, device=device)
    steps = steps or diffusion.num_timesteps
    timesteps = torch.linspace(
        diffusion.num_timesteps - 1, 0, steps, device=device
    ).long().unique_consecutive()
    features = None

    for idx, t in enumerate(timesteps):
        t_int = int(t.item())
        prev_t = int(timesteps[idx + 1].item()) if idx + 1 < len(timesteps) else -1
        t_batch = torch.full((shape[0],), t_int, device=device, dtype=torch.long)
        if return_features and features is None and int(t.item()) <= diffusion.num_timesteps // 2:
            pred, features = model(x, t_batch, return_features=True)
            x = diffusion.ddim_step_from_pred_noise(pred, x, t_int, prev_t)
        else:
            pred = model(x, t_batch)
            x = diffusion.ddim_step_from_pred_noise(pred, x, t_int, prev_t)

    return (x, features) if return_features else x


@torch.no_grad()
def denoise_from_t(model, diffusion, x_t, t_start, device):
    x = x_t
    for t in range(t_start, -1, -1):
        t_batch = torch.full((x.shape[0],), t, device=device, dtype=torch.long)
        x = diffusion.p_sample(model, x, t_batch)
    return x


@torch.no_grad()
def global_cyclic_refine(model, diffusion, z0, t_start, device):
    z_t = diffusion.add_noise_to_timestep(z0, t_start)
    return denoise_from_t(model, diffusion, z_t, t_start, device)


@torch.no_grad()
def masked_cyclic_refine(model, diffusion, z0, t_start, mask, device):
    if mask.shape[-2:] != z0.shape[-2:]:
        mask = torch.nn.functional.interpolate(mask, size=z0.shape[-2:], mode="nearest")
    z_t = diffusion.add_noise_to_timestep(z0, t_start, mask)
    refined = denoise_from_t(model, diffusion, z_t, t_start, device)
    return mask * refined + (1 - mask) * z0


def save_decoded_batch(vae, latents, output_dir, prefix, start_idx=0):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images = vae.decode(latents).detach().cpu()
    images = (images + 1) / 2
    for i, image in enumerate(images):
        save_image(image, output_dir / f"{prefix}_{start_idx + i:05d}.png")


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.time() - self.start
