import math
from typing import List, Optional

import torch
import torch.nn as nn

from models.film import FiLMBlock
from models.conditioning_encoder import ConditioningEncoder


def sinusoidal_embedding(x: torch.Tensor, dim: int = 256) -> torch.Tensor:
    """
    連続値 (noise level = sqrt(alpha_bar)) をサイン波埋め込みに変換
    - x: (B,) scalar noise level
    - 出力: (B, dim)
    """
    half = dim // 2
    freq = torch.exp(
        torch.arange(half, device=x.device, dtype=x.dtype) * (-math.log(10000.0) / max(half - 1, 1))
    )
    angles = x.unsqueeze(1) * freq.unsqueeze(0)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class SelfAttention2d(nn.Module):
    """U-Net のボトルネックで使う簡易 Self-Attention (オプション)"""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # (B, C, H*W) に変換して注意機構
        flat = x.view(b, c, h * w).transpose(1, 2)  # (B, HW, C)
        out, _ = self.attn(flat, flat, flat)
        out = out.transpose(1, 2).view(b, c, h, w)
        return self.norm(out + x)


class DownBlock(nn.Module):
    """Conv → GN → SiLU → FiLM → Downsample"""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.block = FiLMBlock(in_ch, out_ch, cond_dim)
        self.down = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # skip 用の特徴は Downsample 前の活性 (空間サイズを保つ)
        h = self.block(x, cond)
        skip = h
        h = self.down(h)
        return skip, h


class UpBlock(nn.Module):
    """ConvTranspose → GN → SiLU → FiLM → skip connection 結合"""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.norm = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        self.film = nn.Linear(cond_dim, out_ch * 2)  # 一括で γ, β を生成

    def forward(self, x: torch.Tensor, skip: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.up(x)
        h = self.norm(h)
        h = self.act(h)
        gamma_beta = self.film(cond)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        h = h * (gamma + 1.0) + beta
        # skip connection を結合
        h = torch.cat([h, skip], dim=1)
        return h


class UNet2D(nn.Module):
    """
    WaveTransfer を 2D 画像向けにした U-Net
    - 入力: x_t (B,3,H,W), cond_img (B,3,Hc,Wc), noise_level (B,)
    - 出力: 予測ノイズ ε̂ (B,3,H,W)
    """

    def __init__(
        self,
        base_channels: int = 64,
        cond_dim: int = 256,
        num_resolutions: int = 4,
        use_attention: bool = True,
    ):
        super().__init__()
        self.cond_encoder = ConditioningEncoder(cond_dim=cond_dim, hidden=base_channels // 2)

        # noise level 埋め込みを cond_dim へ射影
        self.noise_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # 入力プロジェクション
        self.in_conv = nn.Conv2d(3, base_channels, 3, padding=1)

        channels: List[int] = [base_channels * (2 ** i) for i in range(num_resolutions)]
        self.downs = nn.ModuleList()
        for i in range(num_resolutions):
            in_ch = channels[i - 1] if i > 0 else base_channels
            out_ch = channels[i]
            self.downs.append(DownBlock(in_ch, out_ch, cond_dim))

        bottleneck_ch = channels[-1]
        self.mid_block1 = FiLMBlock(bottleneck_ch, bottleneck_ch, cond_dim)
        self.mid_attn = SelfAttention2d(bottleneck_ch) if use_attention else nn.Identity()
        self.mid_block2 = FiLMBlock(bottleneck_ch, bottleneck_ch, cond_dim)

        self.ups = nn.ModuleList()
        in_ch_running = channels[-1]  # ボトルネックのチャネル数から開始
        for i in reversed(range(num_resolutions)):
            out_ch = channels[i - 1] if i > 0 else base_channels
            skip_ch = channels[i]
            self.ups.append(UpBlock(in_ch_running, out_ch, cond_dim))
            # この UpBlock の出力チャネル数 = out_ch + skip_ch (cat)
            in_ch_running = out_ch + skip_ch

        self.out_norm = nn.GroupNorm(8, base_channels * 2)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(base_channels * 2, 3, 3, padding=1)

    def forward(self, x_t: torch.Tensor, cond_img: torch.Tensor, noise_level: torch.Tensor) -> torch.Tensor:
        # 条件ベクトル作成: cond_img 埋め込み + noise 埋め込み
        cond_vec = self.cond_encoder(cond_img)  # (B, cond_dim)
        noise_emb = sinusoidal_embedding(noise_level, cond_vec.shape[1])
        noise_emb = self.noise_mlp(noise_emb)
        cond = cond_vec + noise_emb

        skips: List[torch.Tensor] = []
        h = self.in_conv(x_t)

        # Down path
        for down in self.downs:
            skip, h = down(h, cond)
            skips.append(skip)

        # Bottleneck
        h = self.mid_block1(h, cond)
        h = self.mid_attn(h)
        h = self.mid_block2(h, cond)

        # Up path (skip を逆順で使用)
        for up, skip in zip(self.ups, reversed(skips)):
            h = up(h, skip, cond)

        h = self.out_norm(h)
        h = self.out_act(h)
        eps_hat = self.out_conv(h)
        return eps_hat
