import torch
import torch.nn as nn


class RelativePositionBias(nn.Module):
    """Learned relative position bias clipped to the configured context length."""

    def __init__(self, n_heads, max_seq_len=4096):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.bias = nn.Embedding(2 * max_seq_len - 1, n_heads)

    def forward(self, seq_len, device):
        positions = torch.arange(seq_len, device=device)
        rel_pos = positions.view(1, -1) - positions.view(-1, 1)
        rel_pos = rel_pos.clamp(
            min=-(self.max_seq_len - 1), max=self.max_seq_len - 1
        )
        rel_pos = rel_pos + self.max_seq_len - 1
        bias = self.bias(rel_pos)
        return bias.permute(2, 0, 1).unsqueeze(0)
