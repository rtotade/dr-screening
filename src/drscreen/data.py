"""Datasets, splits and augmentation.

Canonical manifest format (CSV): image_path, grade, patient_id, eye
  - grade: ICDR 0-4
  - patient_id: used for grouped splits so both eyes of one person never land
    in train and validation at the same time (a common source of inflated results)
Use scripts/prepare_dataset.py to convert APTOS / EyePACS / IDRiD / Messidor-2.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms as T

from .preprocess import load_rgb, preprocess

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def read_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"image_path", "grade"} - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    if "patient_id" not in df.columns:
        df["patient_id"] = df["image_path"]
    base = Path(path).parent
    df["image_path"] = [str(Path(p) if Path(p).is_absolute() else (base / p).resolve()) for p in df["image_path"]]
    df["grade"] = df["grade"].astype(int)
    return df


def grouped_split(df: pd.DataFrame, val_frac: float = 0.2, seed: int = 42):
    n_splits = max(2, round(1 / val_frac))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr_idx, va_idx = next(sgkf.split(df, df["grade"], groups=df["patient_id"]))
    return df.iloc[tr_idx].reset_index(drop=True), df.iloc[va_idx].reset_index(drop=True)


def build_transforms(size: int, train: bool):
    if train:
        return T.Compose([
            T.ToPILImage(),
            T.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(180),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.ToPILImage(),
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class FundusDataset(Dataset):
    """Loads, preprocesses (with on-disk cache) and augments fundus images."""

    def __init__(self, df: pd.DataFrame, size: int, train: bool, ben: bool = True,
                 cache_dir: str | Path | None = None):
        self.df = df.reset_index(drop=True)
        self.size, self.ben = size, ben
        self.tf = build_transforms(size, train)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.df)

    def _load(self, path: str) -> np.ndarray:
        if self.cache_dir:
            h = hashlib.sha1(f"{path}|{self.size}|{self.ben}".encode()).hexdigest()
            key = self.cache_dir / f"{h}.npy"
            if key.exists():
                return np.load(key)
        img = preprocess(load_rgb(path), size=self.size, ben=self.ben)
        if self.cache_dir:
            np.save(key, img)
        return img

    def __getitem__(self, i):
        row = self.df.iloc[i]
        x = self.tf(self._load(row["image_path"]))
        return x, torch.tensor(int(row["grade"]), dtype=torch.long)


def balanced_sampler(labels, power: float = 0.5) -> WeightedRandomSampler:
    """Soft class balancing: weight ~ 1 / freq**power (power=0.5 = sqrt balancing)."""
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=labels.max() + 1).astype(float)
    w = 1.0 / np.power(np.maximum(counts, 1), power)
    return WeightedRandomSampler(torch.as_tensor(w[labels], dtype=torch.double), len(labels), replacement=True)
