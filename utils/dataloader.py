import random
from pathlib import Path
from typing import List, Optional

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PairedImageDataset(Dataset):
    """
    入力画像と条件画像をペアで返すデータセット
    - data_dir: 教師画像 (x0)
    - cond_dirs: 条件画像ディレクトリのリスト (複数音色/スタイルを想定)
    - augment: データ拡張を有効化 (左右反転・カラーじったー)
    - 同名ファイルでペアリング (例: data_dir/piano/foo.png と cond_dir/guitar/foo.png)
    """

    def __init__(
        self,
        data_dir: str,
        cond_dirs: Optional[List[str]] = None,
        image_size: int = 128,
        cond_sampling: str = "random",
        augment: bool = False,
        grayscale: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.cond_dirs = [Path(d) for d in cond_dirs] if cond_dirs is not None else [self.data_dir]
        self.cond_sampling = cond_sampling
        self.grayscale = grayscale

        def collect(root: Path):
            return {p.name: p for p in root.rglob("*") if p.suffix.lower() in [".jpg", ".png"]}

        self.data_images = collect(self.data_dir)
        self.cond_images_list = [collect(d) for d in self.cond_dirs]

        if len(self.data_images) == 0:
            raise RuntimeError(f"画像が見つかりません: {self.data_dir}")

        # 全 cond ディレクトリと共通するファイル名でペアリング
        common_names = set(self.data_images.keys())
        for cond_map, cond_dir in zip(self.cond_images_list, self.cond_dirs):
            common_names &= set(cond_map.keys())
            if len(cond_map) == 0:
                raise RuntimeError(f"画像が見つかりません: {cond_dir}")
        if len(common_names) == 0:
            raise RuntimeError("data_dir と cond_dirs のファイル名に共通集合がありません")

        self.names = sorted(list(common_names))

        # データ拡張: オフならリサイズのみ
        aug_list = []
        if grayscale:
            # グレー画像を3chへ複製してモデルの入力形状 (3ch) に合わせる
            aug_list.append(transforms.Grayscale(num_output_channels=3))
        if augment:
            aug_list += [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            ]
        aug_list += [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
        self.tf = transforms.Compose(aug_list)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img_path = self.data_images[name]

        # cond_sampling = "random" なら条件ドメインをランダム選択
        cond_idx = random.randrange(len(self.cond_dirs)) if self.cond_sampling == "random" else 0
        cond_path = self.cond_images_list[cond_idx][name]

        img = Image.open(img_path).convert("L" if self.grayscale else "RGB")
        cond = Image.open(cond_path).convert("L" if self.grayscale else "RGB")
        return self.tf(img), self.tf(cond)
