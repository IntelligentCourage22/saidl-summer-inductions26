import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class Mlp(nn.Module):
    def __init__(self, hidden_size, mlp_ratio=4.0):
        super().__init__()
        inner = int(hidden_size * mlp_ratio)
        self.fc1 = nn.Linear(hidden_size, inner)
        self.fc2 = nn.Linear(inner, hidden_size)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(hidden_size, mlp_ratio)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

    def forward(self, x, cond):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada(cond).chunk(6, dim=1)
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * h
        h = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = x + gate_mlp.unsqueeze(1) * h
        return x


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x, cond):
        shift, scale = self.ada(cond).chunk(2, dim=1)
        return self.linear(modulate(self.norm(x), shift, scale))


class LatentDiT(nn.Module):
    def __init__(
        self,
        in_channels=4,
        latent_size=32,
        patch_size=4,
        hidden_size=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        feature_layer=3,
    ):
        super().__init__()
        if latent_size % patch_size != 0:
            raise ValueError("latent_size must be divisible by patch_size.")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")

        self.in_channels = in_channels
        self.latent_size = latent_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_patches_side = latent_size // patch_size
        self.num_patches = self.num_patches_side**2
        self.feature_layer = feature_layer

        self.patch_embed = nn.Conv2d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size), requires_grad=False
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.final = FinalLayer(hidden_size, patch_size, in_channels)

        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight.view(module.weight.shape[0], -1))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.pos_embed.data.copy_(
            build_2d_sincos_position_embedding(self.num_patches_side, self.hidden_size)
        )

        for block in self.blocks:
            nn.init.zeros_(block.ada[-1].weight)
            nn.init.zeros_(block.ada[-1].bias)
        nn.init.zeros_(self.final.ada[-1].weight)
        nn.init.zeros_(self.final.ada[-1].bias)
        nn.init.zeros_(self.final.linear.weight)
        nn.init.zeros_(self.final.linear.bias)

    def forward(self, x, timesteps, return_features=False):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        cond = self.time_mlp(timestep_embedding(timesteps, self.hidden_size))

        features = None
        for layer_idx, block in enumerate(self.blocks):
            x = block(x, cond)
            if layer_idx == self.feature_layer:
                features = x

        out = self.final(x, cond)
        out = self.unpatchify(out)
        if return_features:
            return out, features if features is not None else x
        return out

    def unpatchify(self, x):
        bsz = x.shape[0]
        p = self.patch_size
        h = w = self.num_patches_side
        c = self.in_channels
        x = x.reshape(bsz, h, w, p, p, c)
        x = torch.einsum("bhwpqc->bchpwq", x)
        return x.reshape(bsz, c, h * p, w * p)


def build_2d_sincos_position_embedding(grid_size, hidden_size):
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing="ij")
    grid = torch.stack(grid, dim=0).reshape(2, 1, grid_size, grid_size)
    emb_h = build_1d_sincos_position_embedding(hidden_size // 2, grid[0].reshape(-1))
    emb_w = build_1d_sincos_position_embedding(hidden_size // 2, grid[1].reshape(-1))
    return torch.cat([emb_h, emb_w], dim=1).unsqueeze(0)


def build_1d_sincos_position_embedding(dim, positions):
    omega = torch.arange(dim // 2, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (dim / 2)))
    out = torch.einsum("m,d->md", positions, omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)
