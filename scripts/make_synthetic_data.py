"""Generate synthetic fundus-like images for smoke-testing the pipeline end to end.

These are NOT medical data. Lesion-like dots (red "microaneurysms", yellow
"exudates", a vessel-like "neovascular" tuft) scale with grade so a model can
learn something and every stage of the pipeline can be exercised offline.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def fundus(rng, grade: int, size: int = 400) -> np.ndarray:
    img = np.zeros((size, size, 3), np.uint8)
    c, r = size // 2, int(size * 0.46)
    base = np.array([rng.integers(150, 200), rng.integers(60, 90), rng.integers(20, 40)])
    cv2.circle(img, (c, c), r, base.tolist(), -1)
    yy, xx = np.mgrid[:size, :size]
    vign = 1 - 0.35 * (((yy - c) ** 2 + (xx - c) ** 2) / r ** 2)
    img = np.clip(img * vign[..., None], 0, 255).astype(np.uint8)
    disc = (c + rng.integers(40, 80) * rng.choice([-1, 1]), c + rng.integers(-20, 20))
    cv2.circle(img, disc, size // 18, (240, 210, 150), -1)
    for _ in range(8):  # vessels
        ang = rng.uniform(0, 2 * np.pi)
        end = (int(disc[0] + np.cos(ang) * r), int(disc[1] + np.sin(ang) * r))
        cv2.line(img, disc, end, (120, 20, 20), int(rng.integers(2, 4)))

    def spots(n, color, rad):
        for _ in range(n):
            a, d = rng.uniform(0, 2 * np.pi), rng.uniform(0, r * 0.8)
            cv2.circle(img, (int(c + np.cos(a) * d), int(c + np.sin(a) * d)), int(rng.integers(*rad)), color, -1)

    spots([0, 4, 12, 25, 30][grade], (90, 0, 0), (2, 5))           # microaneurysms / haemorrhages
    spots([0, 0, 6, 14, 18][grade], (235, 220, 90), (3, 6))         # hard exudates
    if grade == 4:                                                  # neovascular tuft
        for _ in range(25):
            p = (int(disc[0] + rng.integers(-30, 30)), int(disc[1] + rng.integers(-30, 30)))
            q = (p[0] + int(rng.integers(-15, 15)), p[1] + int(rng.integers(-15, 15)))
            cv2.line(img, p, q, (160, 0, 30), 1)
    mask = np.zeros((size, size), np.uint8); cv2.circle(mask, (c, c), r, 1, -1)
    img = img * mask[..., None]
    noise = rng.normal(0, 4, img.shape)
    return np.clip(img + noise * mask[..., None], 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--patients", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out); (out / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for pid in range(a.patients):
        g = int(rng.choice(5, p=[0.4, 0.15, 0.2, 0.12, 0.13]))
        for eye in ("left", "right"):
            ge = int(np.clip(g + rng.integers(-1, 1), 0, 4))
            name = f"P{pid:04d}_{eye}.png"
            cv2.imwrite(str(out / "images" / name), cv2.cvtColor(fundus(rng, ge), cv2.COLOR_RGB2BGR))
            rows.append(dict(image_path=f"images/{name}", grade=ge, patient_id=f"P{pid:04d}", eye=eye))
    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)
    print(f"Wrote {len(rows)} images to {out}")


if __name__ == "__main__":
    main()
