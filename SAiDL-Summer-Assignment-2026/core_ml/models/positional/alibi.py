import torch
import torch.nn as nn


class AlibiBias(nn.Module):
    """Attention with Linear Biases for causal self-attention."""

    def __init__(self, n_heads):
        super().__init__()
        slopes = self._get_slopes(n_heads)
        self.register_buffer("slopes", torch.tensor(slopes).view(1, n_heads, 1, 1))

    def forward(self, seq_len, device, dtype):
        positions = torch.arange(seq_len, device=device)
        distance = positions.view(1, -1) - positions.view(-1, 1)
        distance = distance.clamp(max=0).abs().to(dtype)
        return -self.slopes.to(device=device, dtype=dtype) * distance.view(
            1, 1, seq_len, seq_len
        )

    @staticmethod
    def _get_slopes(n_heads):
        def slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(torch.log2(torch.tensor(n)).item() - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if torch.log2(torch.tensor(n_heads)).item().is_integer():
            return slopes_power_of_2(n_heads)

        closest_power = 2 ** int(torch.floor(torch.log2(torch.tensor(n_heads))).item())
        slopes = slopes_power_of_2(closest_power)
        extra = AlibiBias._get_slopes(2 * closest_power)
        return slopes + extra[0::2][: n_heads - closest_power]
