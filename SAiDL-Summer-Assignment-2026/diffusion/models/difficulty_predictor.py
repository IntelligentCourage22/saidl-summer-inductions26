import math

import torch
import torch.nn as nn


class DifficultyPredictor(nn.Module):
    def __init__(self, feature_dim=384, token_grid=8, out_size=32):
        super().__init__()
        self.token_grid = token_grid
        self.out_size = out_size
        self.net = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, 256, kernel_size=2, stride=2),
            nn.GroupNorm(32, 256),
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.GroupNorm(32, 128),
            nn.GELU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GroupNorm(16, 64),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, features):
        if features.dim() == 3:
            bsz, tokens, channels = features.shape
            grid = int(math.sqrt(tokens))
            features = features.transpose(1, 2).reshape(bsz, channels, grid, grid)
        out = self.net(features)
        if out.shape[-1] != self.out_size:
            out = torch.nn.functional.interpolate(
                out, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False
            )
        return out
