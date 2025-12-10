import argparse
import os
from pathlib import Path

import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

from models.unet2d import UNet2D
from utils.diffusion import Diffusion
from utils.config import load_config, merge_config


def load_image(path: str, size: int, grayscale: bool = False):
    tf = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3) if grayscale else transforms.Lambda(lambda x: x),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    img = Image.open(path).convert("L" if grayscale else "RGB")
    return tf(img)


def parse_args():
    parser = argparse.ArgumentParser(description="2D WaveTransfer 推論スクリプト")
    parser.add_argument("--config", type=str, default="config/default_config.json", help="設定ファイル(JSON)")
    parser.add_argument("--checkpoint", type=str, default=None, help="学習済みモデルのパス")
    parser.add_argument("--cond_image", type=str, default=None, help="条件画像パス（ターゲット音色/スタイル）")
    parser.add_argument("--source_image", type=str, default=None, help="変換したい元画像（指定時はここにノイズを足して開始）")
    parser.add_argument("--start_step", type=int, default=None, help="元画像にノイズを足すステップ数（0〜T-1）")
    parser.add_argument("--pure_noise", action="store_true", help="元画像なしで純ノイズから生成する場合に指定")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--base_channels", type=int, default=None)
    parser.add_argument("--cond_dim", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--grayscale", action="store_true", help="グレー画像を扱う（3chに複製）")
    args = parser.parse_args()

    default_cfg = {}
    if args.config and os.path.exists(args.config):
        default_cfg = load_config(args.config)
    args = merge_config(args, default_cfg, section="infer")

    # 必須パラメータが None の場合はエラー
    if args.checkpoint is None:
        raise ValueError("checkpoint が指定されていません (--checkpoint または config)")
    if args.cond_image is None:
        raise ValueError("cond_image が指定されていません (--cond_image または config)")
    if not args.pure_noise and args.source_image is None:
        raise ValueError("元画像を指定してください (--source_image または config) 。純ノイズ生成をしたい場合は --pure_noise を付けてください。")

    args.out_dir = args.out_dir or "output/samples"
    args.num_samples = args.num_samples or 4
    args.image_size = args.image_size or 256
    args.timesteps = args.timesteps or 400
    args.base_channels = args.base_channels or 64
    args.cond_dim = args.cond_dim or 256
    args.depth = args.depth or 4
    args.grayscale = bool(args.grayscale)
    return args


@torch.no_grad()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet2D(base_channels=args.base_channels, cond_dim=args.cond_dim, num_resolutions=args.depth).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    diffusion = Diffusion(T=args.timesteps, device=device)

    # 条件画像をロード（ターゲット）
    cond_img = load_image(args.cond_image, args.image_size, grayscale=args.grayscale).unsqueeze(0).to(device)
    cond_img = cond_img.repeat(args.num_samples, 1, 1, 1)

    # 元画像を使う場合: 指定ステップ分ノイズを足した x_t を開始点にする
    start_x = None
    start_t = None
    if not args.pure_noise:
        src = load_image(args.source_image, args.image_size, grayscale=args.grayscale).unsqueeze(0).to(device)
        src = src.repeat(args.num_samples, 1, 1, 1)
        start_t = args.start_step if args.start_step is not None else args.timesteps // 2
        start_t = max(0, min(args.timesteps - 1, int(start_t)))
        t_tensor = torch.full((args.num_samples,), start_t, device=device, dtype=torch.long)
        start_x, _ = diffusion.q_sample(src, t_tensor)

    samples = diffusion.sample(
        model,
        cond_img,
        shape=(3, args.image_size, args.image_size),
        start_x=start_x,
        start_t=start_t,
    )
    os.makedirs(args.out_dir, exist_ok=True)
    for i, img in enumerate(samples):
        save_path = Path(args.out_dir) / f"sample_{i}.png"
        # [-1,1] -> [0,1] へ戻す
        save_image((img.clamp(-1, 1) + 1) * 0.5, save_path)
        print(f"saved: {save_path}")


if __name__ == "__main__":
    main(parse_args())
