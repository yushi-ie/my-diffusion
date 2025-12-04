class WaveTransferUNet(nn.Module):
    def __init__(self, cond_dim=128, base=32):
        super().__init__()

        # Condition: mel(128, 66) → 1D Upsample
        self.cond_up = nn.Sequential(
            nn.ConvTranspose1d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 4, 2, 1),
            nn.ReLU(),
        )

        # Downsample noisy audio
        self.down1 = nn.Conv2d(1, base, 4, 2, 1)
        self.down2 = nn.Conv2d(base, base*2, 4, 2, 1)
        self.down3 = nn.Conv2d(base*2, base*4, 4, 2, 1)

        # FiLM (use cond vector)
        self.film1 = FiLM(base, cond_dim)
        self.film2 = FiLM(base*2, cond_dim)
        self.film3 = FiLM(base*4, cond_dim)

        # Up path
        self.up1 = nn.ConvTranspose2d(base*4, base*2, 4, 2, 1)
        self.up2 = nn.ConvTranspose2d(base*2*2, base, 4, 2, 1)
        self.up3 = nn.ConvTranspose2d(base*2, 1, 4, 2, 1)

    def forward(self, x_t, mel, cond_vec):
        # 1) Condition upsample
        cond_feat = self.cond_up(mel)   # (B,32,T)

        # 2) Down path
        d1 = self.film1(self.down1(x_t), cond_vec)
        d2 = self.film2(self.down2(d1), cond_vec)
        d3 = self.film3(self.down3(d2), cond_vec)

        # 3) Up path with skip-connection
        u1 = self.up1(d3)
        u1 = torch.cat([u1, d2], dim=1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d1], dim=1)

        out = self.up3(u2)
        return out  # predicted noise
