"""Fundus image preprocessing.

Pipeline: decode -> crop dark borders -> centre on the fundus circle -> resize ->
optional Ben Graham local-contrast normalisation -> circular mask.
All functions take/return RGB uint8 numpy arrays.
"""
from __future__ import annotations

import cv2
import numpy as np


def load_rgb(path_or_bytes) -> np.ndarray:
    """Load an image from a path or raw bytes as RGB uint8."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        buf = np.frombuffer(path_or_bytes, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(str(path_or_bytes), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def fundus_mask(img: np.ndarray, thresh: int = 10) -> np.ndarray:
    """Boolean mask of pixels that belong to the illuminated fundus."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray > thresh


def crop_to_fundus(img: np.ndarray, thresh: int = 10) -> np.ndarray:
    """Crop away the black border around the fundus."""
    mask = fundus_mask(img, thresh)
    if mask.sum() < 0.01 * mask.size:  # nearly black image: leave as is
        return img
    ys, xs = np.where(mask)
    cropped = img[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    # pad to square so the circle is not distorted when resized
    h, w = cropped.shape[:2]
    side = max(h, w)
    out = np.zeros((side, side, 3), dtype=img.dtype)
    y0, x0 = (side - h) // 2, (side - w) // 2
    out[y0: y0 + h, x0: x0 + w] = cropped
    return out


def ben_graham(img: np.ndarray, sigma_frac: float = 1 / 30, weight: float = 4.0) -> np.ndarray:
    """Local average colour subtraction (Ben Graham, Kaggle DR 2015).

    Highlights lesions (microaneurysms, haemorrhages, exudates) and evens out
    illumination differences between cameras.
    """
    sigma = max(1.0, img.shape[0] * sigma_frac)
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, weight, blur, -weight, 128)


def circle_mask(img: np.ndarray, scale: float = 0.95) -> np.ndarray:
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), int(min(h, w) / 2 * scale), 1, -1)
    return img * mask[..., None] + 128 * (1 - mask[..., None]).astype(img.dtype)


def preprocess(img: np.ndarray, size: int = 512, ben: bool = True) -> np.ndarray:
    """Full preprocessing used for both training and inference."""
    img = crop_to_fundus(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    if ben:
        img = ben_graham(img)
        img = circle_mask(img)
    return img.astype(np.uint8)
