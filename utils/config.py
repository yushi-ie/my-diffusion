import json
from argparse import Namespace
from copy import deepcopy


def load_config(path: str) -> dict:
    """JSON の設定ファイルを読み込むユーティリティ"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_config(args: Namespace, default_cfg: dict, section: str) -> Namespace:
    """
    argparse で受け取った引数に対して、config ファイル側の値をマージする。
    - CLI で指定した値 (None 以外) を優先
    - None の項目を config の値で補完
    """
    cfg = deepcopy(default_cfg.get(section, {}))
    for k, v in vars(args).items():
        if v is None and k in cfg:
            setattr(args, k, cfg[k])
    return args
