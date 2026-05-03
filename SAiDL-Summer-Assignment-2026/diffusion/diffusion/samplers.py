import time
from pathlib import Path

import torch
from torchvision.utils import save_image


@torch.no_grad()
def _ddim_timesteps(
    diffusion, steps=None, device=None, start_timestep=None, include_timestep=None
):
    device = device or diffusion.device
    steps = steps or diffusion.num_timesteps
    timesteps = torch.linspace(
        diffusion.num_timesteps - 1, 0, steps, device=device
    ).long().unique_consecutive()

    if include_timestep is not None:
        include_timestep = int(include_timestep)
        if (
            0 <= include_timestep < diffusion.num_timesteps
            and not (timesteps == include_timestep).any()
        ):
            replace_idx = torch.argmin((timesteps - include_timestep).abs())
            timesteps[replace_idx] = include_timestep
            timesteps = torch.sort(torch.unique(timesteps), descending=True).values

    if start_timestep is not None:
        start_timestep = int(start_timestep)
        lower_timesteps = timesteps[timesteps < start_timestep]
        start = torch.tensor([start_timestep], device=device, dtype=torch.long)
        timesteps = torch.cat([start, lower_timesteps]).unique_consecutive()

    return timesteps


@torch.no_grad()
def sample_loop(
    model,
    diffusion,
    shape,
    device,
    steps=None,
    return_features=False,
    feature_timestep=None,
):
    model.eval()
    x = torch.randn(shape, device=device)
    timesteps = _ddim_timesteps(
        diffusion,
        steps,
        device,
        include_timestep=feature_timestep if return_features else None,
    )
    features = None

    for idx, t in enumerate(timesteps):
        t_int = int(t.item())
        prev_t = int(timesteps[idx + 1].item()) if idx + 1 < len(timesteps) else -1
        t_batch = torch.full((shape[0],), t_int, device=device, dtype=torch.long)
        should_capture_features = return_features and features is None
        if feature_timestep is not None:
            should_capture_features = should_capture_features and t_int == int(
                feature_timestep
            )
        else:
            should_capture_features = (
                should_capture_features and t_int <= diffusion.num_timesteps // 2
            )

        if should_capture_features:
            pred, features = model(x, t_batch, return_features=True)
            x = diffusion.ddim_step_from_pred_noise(pred, x, t_int, prev_t)
        else:
            pred = model(x, t_batch)
            x = diffusion.ddim_step_from_pred_noise(pred, x, t_int, prev_t)

    return (x, features) if return_features else x


@torch.no_grad()
def ddim_denoise_from_t(model, diffusion, x_t, t_start, device, steps=None):
    x = x_t
    timesteps = _ddim_timesteps(diffusion, steps, device, start_timestep=t_start)
    for idx, t in enumerate(timesteps):
        t_int = int(t.item())
        prev_t = int(timesteps[idx + 1].item()) if idx + 1 < len(timesteps) else -1
        t_batch = torch.full((x.shape[0],), t_int, device=device, dtype=torch.long)
        pred = model(x, t_batch)
        x = diffusion.ddim_step_from_pred_noise(pred, x, t_int, prev_t)
    return x


@torch.no_grad()
def global_cyclic_refine(model, diffusion, z0, t_start, device, steps=None):
    z_t = diffusion.add_noise_to_timestep(z0, t_start)
    return ddim_denoise_from_t(model, diffusion, z_t, t_start, device, steps)


@torch.no_grad()
def masked_cyclic_refine(model, diffusion, z0, t_start, mask, device, steps=None):
    if mask.shape[-2:] != z0.shape[-2:]:
        mask = torch.nn.functional.interpolate(
            mask, size=z0.shape[-2:], mode="nearest"
        )
    mask = mask.to(device=device, dtype=z0.dtype).clamp(0, 1)

    known_noise = torch.randn_like(z0)
    timesteps = _ddim_timesteps(diffusion, steps, device, start_timestep=t_start)
    start_batch = torch.full(
        (z0.shape[0],), int(timesteps[0].item()), device=device, dtype=torch.long
    )
    x = diffusion.q_sample(z0, start_batch, known_noise)

    for idx, t in enumerate(timesteps):
        t_int = int(t.item())
        prev_t = int(timesteps[idx + 1].item()) if idx + 1 < len(timesteps) else -1
        t_batch = torch.full((z0.shape[0],), t_int, device=device, dtype=torch.long)
        known_t = diffusion.q_sample(z0, t_batch, known_noise)
        x = mask * x + (1 - mask) * known_t

        pred = model(x, t_batch)
        x = diffusion.ddim_step_from_pred_noise(pred, x, t_int, prev_t)

        if prev_t < 0:
            known_prev = z0
        else:
            prev_batch = torch.full(
                (z0.shape[0],), prev_t, device=device, dtype=torch.long
            )
            known_prev = diffusion.q_sample(z0, prev_batch, known_noise)
        x = mask * x + (1 - mask) * known_prev

    return x


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
