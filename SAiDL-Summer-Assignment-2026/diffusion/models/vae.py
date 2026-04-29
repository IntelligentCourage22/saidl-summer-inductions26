import torch
import torch.nn as nn
from diffusers import AutoencoderKL


class FrozenVAE(nn.Module):
    def __init__(self, model_name="stabilityai/sd-vae-ft-ema", scaling_factor=0.18215):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(model_name)
        self.scaling_factor = scaling_factor
        self.vae.requires_grad_(False)
        self.vae.eval()

    @torch.no_grad()
    def encode(self, images):
        posterior = self.vae.encode(images).latent_dist
        return posterior.sample() * self.scaling_factor

    @torch.no_grad()
    def decode(self, latents):
        images = self.vae.decode(latents / self.scaling_factor).sample
        return images.clamp(-1, 1)
