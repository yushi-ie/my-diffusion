import torch
import torch.nn as nn


class FiLM(nn.Module):
    """Generate per-channel scale/bias from a conditioning vector."""

    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.to_gamma = nn.Linear(cond_dim, channels)
        self.to_beta = nn.Linear(cond_dim, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma = self.to_gamma(cond).unsqueeze(-1).unsqueeze(-1)
        beta = self.to_beta(cond).unsqueeze(-1).unsqueeze(-1)
        return x * (gamma + 1.0) + beta


class FiLMBlock(nn.Module):
    """
    Two-stage Conv + GN + SiLU + FiLM block to strengthen conditioning.
    - Stage1: in_ch -> out_ch
    - Stage2: out_ch -> out_ch
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        cond_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.act1 = nn.SiLU()
        self.film1 = FiLM(out_ch, cond_dim)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, 1, padding)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act2 = nn.SiLU()
        self.film2 = FiLM(out_ch, cond_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act1(h)
        h = self.film1(h, cond)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act2(h)
        h = self.film2(h, cond)
        return h
