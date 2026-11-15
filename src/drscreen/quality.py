"""Image quality gate.

Screening programmes fail quietly when blurred or dark photos are graded as
"No DR". This gate rejects ungradable images so the technician can retake them
on the spot (and ungradable patients are referred, never silently passed).

Thresholds are starting points: calibrate them on images from your own camera
with an ophthalmologist labelling gradable / ungradable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .preprocess import crop_to_fundus, fundus_mask


@dataclass
class QualityReport:
    gradable: bool
    sharpness: float
    brightness: float
    fundus_fraction: float
    min_side_px: int
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QualityThresholds:
    min_sharpness: float = 15.0       # variance of Laplacian on 512px green channel
    min_brightness: float = 25.0      # mean intensity inside fundus (0-255)
    max_brightness: float = 230.0
    min_fundus_fraction: float = 0.35  # share of frame that is illuminated fundus
    min_side_px: int = 300


def assess(img: np.ndarray, t: QualityThresholds | None = None) -> QualityReport:
    t = t or QualityThresholds()
    reasons: list[str] = []
    h, w = img.shape[:2]
    min_side = min(h, w)
    if min_side < t.min_side_px:
        reasons.append(f"Resolution too low ({min_side}px < {t.min_side_px}px)")

    mask = fundus_mask(img)
    fundus_fraction = float(mask.mean())
    if fundus_fraction < t.min_fundus_fraction:
        reasons.append("Fundus not centred or too small in frame")

    cropped = cv2.resize(crop_to_fundus(img), (512, 512), interpolation=cv2.INTER_AREA)
    green = cropped[..., 1]
    inner = fundus_mask(cropped)
    brightness = float(green[inner].mean()) if inner.any() else 0.0
    if brightness < t.min_brightness:
        reasons.append("Image too dark (check pupil dilation / flash)")
    if brightness > t.max_brightness:
        reasons.append("Image over-exposed")

    lap = cv2.Laplacian(green, cv2.CV_64F)
    sharpness = float(lap[inner].var()) if inner.any() else 0.0
    if sharpness < t.min_sharpness:
        reasons.append("Image blurred (refocus and retake)")

    return QualityReport(
        gradable=not reasons,
        sharpness=round(sharpness, 2),
        brightness=round(brightness, 2),
        fundus_fraction=round(fundus_fraction, 3),
        min_side_px=int(min_side),
        reasons=reasons,
    )
