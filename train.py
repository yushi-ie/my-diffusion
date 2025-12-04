import argparse
import os

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from models.unet2d import UNet2D
from utils.config import load_config, merge_config
from utils.dataloader import PairedImageDataset
from utils.diffusion import Diffusion, noise_prediction_loss


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 日本語コメント: データセットと DataLoader を準備
    cond_dirs = args.cond_dirs or ([args.cond_dir] if args.cond_dir else None)
    dataset = PairedImageDataset(
        args.data_dir,
        cond_dirs=cond_dirs,
        image_size=args.image_size,
        cond_sampling=args.cond_sampling,
        augment=args.augment,
        grayscale=args.grayscale,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    model = UNet2D(base_channels=args.base_channels, cond_dim=args.cond_dim, num_resolutions=args.depth).to(device)
    diffusion = Diffusion(T=args.timesteps, device=device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    os.makedirs(args.out_dir, exist_ok=True)
    global_step = 0
    model.train()
    loss_history = []
    last_saved_step = 0

    for epoch in range(args.epochs):
        for x0, cond_img in loader:
            x0 = x0.to(device)
            cond_img = cond_img.to(device)
            loss = noise_prediction_loss(model, diffusion, x0, cond_img)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            global_step += 1
            loss_history.append((global_step, loss.item()))
            if global_step % args.log_interval == 0:
                print(f"[epoch {epoch} step {global_step}] loss: {loss.item():.4f}")

            if global_step % args.save_interval == 0:
                ckpt_path = os.path.join(args.out_dir, f"unet2d_step{global_step}.pt")
                torch.save({"model": model.state_dict()}, ckpt_path)
                print(f"checkpoint saved: {ckpt_path}")
                last_saved_step = global_step

    # ループを通じて一度も save_interval に達しなかった場合、最終ステップで保存する
    if last_saved_step < global_step:
        ckpt_path = os.path.join(args.out_dir, f"unet2d_step{global_step}.pt")
        torch.save({"model": model.state_dict()}, ckpt_path)
        print(f"checkpoint saved (final): {ckpt_path}")

    # 学習終了後に損失を可視化して保存
    if loss_history:
        steps, losses = zip(*loss_history)
        plt.figure(figsize=(8, 4))
        plt.plot(steps, losses, label="train loss")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title("Training Loss History")
        plt.legend()
        plt.tight_layout()
        fig_path = os.path.join(args.out_dir, "loss_history.png")
        plt.savefig(fig_path)
        plt.close()
        # 数値も保存しておく
        txt_path = os.path.join(args.out_dir, "loss_history.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for s, l in loss_history:
                f.write(f"{s}\t{l}\n")
        print(f"loss history saved: {fig_path}, {txt_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="2D WaveTransfer 学習スクリプト")
    parser.add_argument("--config", type=str, default="config/default_config.json", help="設定ファイル(JSON)")
    parser.add_argument("--data_dir", type=str, default=None, help="教師画像ディレクトリ")
    parser.add_argument("--cond_dir", type=str, default=None, help="条件画像ディレクトリ (1 つだけ使う場合)")
    parser.add_argument(
        "--cond_dirs",
        type=str,
        nargs="*",
        default=None,
        help="複数の条件画像ディレクトリ (音色/スタイルが複数ある場合)",
    )
    parser.add_argument(
        "--cond_sampling",
        type=str,
        default=None,
        choices=["random", "first"],
        help="条件ドメインの選び方: random で各ステップランダム / first で先頭のみ",
    )
    parser.add_argument("--augment", action="store_true", help="データ拡張を有効化")
    parser.add_argument("--grayscale", action="store_true", help="グレー画像を扱う (3chに複製して学習/推論)")
    parser.add_argument("--out_dir", type=str, default=None, help="チェックポイント保存先")
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--base_channels", type=int, default=None)
    parser.add_argument("--cond_dim", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None, help="U-Net の深さ (3〜4 推奨)")
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--save_interval", type=int, default=None)
    args = parser.parse_args()

    # config を読み込み、CLI 未指定の項目を補完
    default_cfg = {}
    if args.config and os.path.exists(args.config):
        default_cfg = load_config(args.config)
    args = merge_config(args, default_cfg, section="train")

    # それでも None の項目にデフォルトを設定
    args.data_dir = args.data_dir or "data/train"
    args.out_dir = args.out_dir or "output/checkpoints"
    args.image_size = args.image_size or 256
    args.batch_size = args.batch_size or 8
    args.epochs = args.epochs or 10
    args.lr = args.lr or 2e-4
    args.timesteps = args.timesteps or 400
    args.base_channels = args.base_channels or 64
    args.cond_dim = args.cond_dim or 256
    args.depth = args.depth or 4
    args.log_interval = args.log_interval or 50
    args.save_interval = args.save_interval or 500
    args.cond_sampling = args.cond_sampling or "random"
    args.augment = bool(args.augment)
    args.grayscale = bool(args.grayscale)
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
