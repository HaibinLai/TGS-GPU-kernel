#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Tuple, Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import datasets, transforms

from torchvision.models import (
    resnet18, resnet34, resnet50,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(
        description="ResNet inference on ImageFolder or flat image directory."
    )
    p.add_argument("--data-dir", type=str, required=True,
                   help="目录：可为 ImageFolder（有子目录=类别）或扁平目录（仅图片）。")
    p.add_argument("--model", type=str, default="resnet18",
                   choices=["resnet18", "resnet34", "resnet50"])
    p.add_argument("--weights", type=str, default="default",
                   choices=["default", "none"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"])
    p.add_argument("--amp", action="store_true")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--out", type=str, default="preds.csv")
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--recursive", action="store_true",
                   help="扁平目录模式下递归搜索子目录中的图片（默认只看当前层）。")
    return p.parse_args()


def get_device_choice(arg_device: str) -> torch.device:
    if arg_device == "cpu":
        return torch.device("cpu")
    if arg_device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[WARN] CUDA 不可用，回退到 CPU。", file=sys.stderr)
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(model_name: str, weights_flag: str):
    use_weights = None
    categories = None

    if model_name == "resnet18":
        if weights_flag == "default":
            use_weights = ResNet18_Weights.DEFAULT
            categories = use_weights.meta.get("categories", None)
        model = resnet18(weights=use_weights)
    elif model_name == "resnet34":
        if weights_flag == "default":
            use_weights = ResNet34_Weights.DEFAULT
            categories = use_weights.meta.get("categories", None)
        model = resnet34(weights=use_weights)
    elif model_name == "resnet50":
        if weights_flag == "default":
            use_weights = ResNet50_Weights.DEFAULT
            categories = use_weights.meta.get("categories", None)
        model = resnet50(weights=use_weights)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.eval()
    return model, use_weights, categories


def build_infer_transform(use_weights):
    if use_weights is not None:
        return use_weights.transforms()
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])


class FlatDirDataset(Dataset):
    """不需要子目录标签；仅返回 (tensor, 0) 和路径。"""
    def __init__(self, root: str, transform=None, recursive=False):
        self.root = Path(root)
        self.transform = transform
        self.paths: List[Path] = []
        if recursive:
            for p in self.root.rglob("*"):
                if p.suffix.lower() in IMG_EXTS and p.is_file():
                    self.paths.append(p)
        else:
            for p in self.root.glob("*"):
                if p.suffix.lower() in IMG_EXTS and p.is_file():
                    self.paths.append(p)
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No images found in {root}.")
        self.paths.sort()

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # label 填 0（占位）
        return img, 0

    @property
    def samples(self) -> Sequence[Tuple[str, int]]:
        return [(str(p), 0) for p in self.paths]

    @property
    def classes(self) -> List[str]:
        return []


def try_imagefolder(data_dir: str, tfm):
    # 若 data_dir 下存在至少一个子目录，则按 ImageFolder 处理
    entries = [p for p in Path(data_dir).iterdir() if p.is_dir()]
    if len(entries) == 0:
        return None
    return datasets.ImageFolder(data_dir, transform=tfm)


def make_loader(data_dir: str, tfm, batch_size: int, workers: int,
                pin_memory: bool, recursive: bool):
    ds = try_imagefolder(data_dir, tfm)
    if ds is None:
        print("[INFO] 未发现子目录，启用扁平目录推理模式（FlatDirDataset）。")
        ds = FlatDirDataset(data_dir, transform=tfm, recursive=recursive)
    else:
        print("[INFO] 发现子目录，按 ImageFolder 处理。")

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=pin_memory, drop_last=False
    )
    return ds, loader


def safe_tqdm(total=None, disable=False):
    try:
        from tqdm import tqdm
        return tqdm(total=total, disable=disable)
    except Exception:
        class Dummy:
            def __init__(self, total=None, disable=False): pass
            def update(self, n=1): pass
            def close(self): pass
        return Dummy(total=total, disable=disable)


@torch.no_grad()
def run_inference(model, device, loader, topk: int, use_amp: bool,
                  class_names: List[str], out_csv: str, img_paths: List[str],
                  show_progress: bool):
    model.to(device)
    if device.type == "cuda" and use_amp:
        autocast_ctx = torch.cuda.amp.autocast
    else:
        autocast_ctx = torch.cpu.amp.autocast

    pbar = safe_tqdm(total=len(loader), disable=show_progress)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["path", "top1_idx", "top1_name", "top1_prob"]
        for k in range(1, topk + 1):
            header += [f"top{k}_idx", f"top{k}_name", f"top{k}_prob"]
        writer.writerow(header)

        offset = 0
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            with autocast_ctx():
                logits = model(images)
                probs = torch.softmax(logits, dim=1)

            topk_prob, topk_idx = torch.topk(probs, k=topk, dim=1)
            top1_idx = topk_idx[:, 0]
            top1_prob = topk_prob[:, 0]

            bsz = images.size(0)
            batch_paths = img_paths[offset:offset + bsz]
            offset += bsz

            for i in range(bsz):
                row = []
                path_i = batch_paths[i] if i < len(batch_paths) else ""
                row.append(path_i)

                idx1 = int(top1_idx[i].item())
                prob1 = float(top1_prob[i].item())
                name1 = class_names[idx1] if (class_names and 0 <= idx1 < len(class_names)) else str(idx1)
                row += [idx1, name1, f"{prob1:.6f}"]

                for k in range(topk):
                    idxk = int(topk_idx[i, k].item())
                    probk = float(topk_prob[i, k].item())
                    namek = class_names[idxk] if (class_names and 0 <= idxk < len(class_names)) else str(idxk)
                    row += [idxk, namek, f"{probk:.6f}"]

                writer.writerow(row)

            pbar.update(1)

    pbar.close()


def main():
    args = parse_args()

    device = get_device_choice(args.device)
    print(f"[INFO] Device: {device}")

    model, use_weights, categories = build_model(args.model, args.weights)
    print(f"[INFO] Model: {args.model}, Weights: {args.weights}")

    infer_transform = build_infer_transform(use_weights)

    dataset, loader = make_loader(
        args.data_dir, infer_transform,
        batch_size=args.batch_size,
        workers=args.workers,
        pin_memory=args.pin_memory,
        recursive=args.recursive
    )

    class_names = categories if categories is not None else getattr(dataset, "classes", [])
    if not class_names:
        print("[INFO] 未提供类别名（扁平目录或无权重类名），CSV 将仅写索引。")

    img_paths = [p for (p, _) in dataset.samples]

    run_inference(
        model=model,
        device=device,
        loader=loader,
        topk=max(1, args.topk),
        use_amp=bool(args.amp),
        class_names=class_names,
        out_csv=args.out,
        img_paths=img_paths,
        show_progress=args.no_progress
    )

    print(f"[INFO] Done. CSV saved to: {args.out}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    main()
