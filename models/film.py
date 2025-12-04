import torch
import torch.nn as nn


class FiLM(nn.Module):
    """条件ベクトルから γ と β を生成して特徴マップを変調する FiLM 層"""

    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.to_gamma = nn.Linear(cond_dim, channels)
        self.to_beta = nn.Linear(cond_dim, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # cond: (B, cond_dim) を受け取り、チャネル毎の γ, β を生成
        gamma = self.to_gamma(cond).unsqueeze(-1).unsqueeze(-1)
        beta = self.to_beta(cond).unsqueeze(-1).unsqueeze(-1)
        return x * (gamma + 1.0) + beta


class FiLMBlock(nn.Module):
    """Conv + GN + SiLU + FiLM をひとまとめにした基本ブロック"""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.norm = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        self.film = FiLM(out_ch, cond_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = self.norm(h)
        h = self.act(h)
        h = self.film(h, cond)
        return h
