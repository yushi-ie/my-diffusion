import torch
import torch.nn as nn


class ConditioningEncoder(nn.Module):
    """
    条件画像を CNN で埋め込みベクトルへ変換するエンコーダ
    - 入力サイズは任意 (B,3,Hc,Wc)
    - Conv で空間を縮小 → Global Average Pooling → MLP
    """

    def __init__(self, cond_dim: int = 256, hidden: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden, 5, stride=2, padding=2),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, hidden * 2),
            nn.SiLU(),
            nn.Conv2d(hidden * 2, hidden * 4, 3, stride=2, padding=1),
            nn.GroupNorm(8, hidden * 4),
            nn.SiLU(),
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden * 4, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, cond_img: torch.Tensor) -> torch.Tensor:
        # CNN で特徴抽出
        h = self.encoder(cond_img)
        # Global Average Pooling (B, C, H, W) -> (B, C)
        h = h.mean(dim=[2, 3])
        cond_vec = self.proj(h)
        return cond_vec
