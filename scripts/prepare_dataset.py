"""Convert public DR datasets into the canonical manifest (image_path, grade, patient_id, eye).

Supported sources (download them yourself; check each licence before commercial use):
  aptos    APTOS 2019 Blindness Detection (Kaggle): train.csv [id_code, diagnosis], train_images/*.png
  eyepacs  EyePACS / Kaggle DR 2015: trainLabels.csv [image, level], images named <patient>_<left|right>.jpeg
  idrid    IDRiD Disease Grading: CSV [Image name, Retinopathy grade], images/*.jpg
  messidor2 Messidor-2 with Krause et al. adjudicated grades: CSV [image_id, adjudicated_dr_grade]
  folder   Your own data: one sub-folder per grade 0..4 (e.g. data/clinic/2/img001.jpg)

Example:
  python scripts/prepare_dataset.py aptos --labels /data/aptos/train.csv --images /data/aptos/train_images --out data/manifest.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def find(images: Path, stem: str) -> str | None:
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".JPG", ".PNG"):
        p = images / f"{stem}{ext}"
        if p.exists():
            return str(p.resolve())
    p = images / stem
    return str(p.resolve()) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["aptos", "eyepacs", "idrid", "messidor2", "folder"])
    ap.add_argument("--labels", help="label CSV (not needed for 'folder')")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    images = Path(a.images)
    rows = []

    if a.source == "folder":
        for g in range(5):
            for p in sorted((images / str(g)).glob("*")):
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
                    rows.append(dict(image_path=str(p.resolve()), grade=g, patient_id=p.stem.split("_")[0], eye=""))
    else:
        df = pd.read_csv(a.labels)
        if a.source == "aptos":
            it = ((r.id_code, r.diagnosis, r.id_code, "") for r in df.itertuples())
        elif a.source == "eyepacs":
            it = ((r.image, r.level, r.image.rsplit("_", 1)[0], r.image.rsplit("_", 1)[-1]) for r in df.itertuples())
        elif a.source == "idrid":
            df.columns = [c.strip() for c in df.columns]
            it = ((r["Image name"], r["Retinopathy grade"], r["Image name"], "") for _, r in df.iterrows())
        else:  # messidor2
            df = df.dropna(subset=["adjudicated_dr_grade"])
            it = ((Path(r.image_id).stem, int(r.adjudicated_dr_grade), Path(r.image_id).stem[:-1], "") for r in df.itertuples())
        for stem, grade, pid, eye in it:
            path = find(images, str(stem))
            if path:
                rows.append(dict(image_path=path, grade=int(grade), patient_id=str(pid), eye=eye))

    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("No images matched. Check --images / --labels paths.")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"Wrote {len(out)} rows to {a.out}")
    print(out["grade"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
