"""Evaluate a checkpoint on a held-out manifest (e.g. Messidor-2, IDRiD, or your
own prospectively collected Indian clinic data) and write a markdown report.

    python -m drscreen.evaluate --ckpt runs/default/best.pt --manifest data/test.csv --out runs/default/test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from . import GRADES
from .data import FundusDataset, read_manifest
from .metrics import full_report
from .model import load_checkpoint
from .train import predict_probs


def fmt_ci(ci):
    return f"({ci[0]:.3f}–{ci[1]:.3f})"


def to_markdown(r: dict, name: str) -> str:
    lines = [
        f"# Evaluation report: {name}", "",
        f"Images: {r['n_images']}  ·  Referable prevalence: {r['prevalence_referable']:.1%}", "",
        "## Referable DR (moderate NPDR or worse)", "",
        "| Metric | Value | 95% CI |", "| --- | --- | --- |",
        f"| Sensitivity | {r['referable_sensitivity']:.3f} | {fmt_ci(r['referable_sensitivity_ci95'])} |",
        f"| Specificity | {r['referable_specificity']:.3f} | {fmt_ci(r['referable_specificity_ci95'])} |",
    ]
    if "referable_auc" in r:
        lines.append(f"| AUC | {r['referable_auc']:.3f} | {fmt_ci(r['referable_auc_ci95'])} |")
    lines += [
        f"| PPV | {r['referable_ppv']:.3f} | |", f"| NPV | {r['referable_npv']:.3f} | |",
        f"| Threshold | {r['referable_threshold']:.3f} | |", "",
        "## 5-grade ICDR", "", f"Quadratic weighted kappa: **{r['qwk']:.3f}**  ·  Accuracy: {r['accuracy']:.3f}", "",
        "Confusion matrix (rows = truth, columns = prediction):", "",
        "| | " + " | ".join(GRADES) + " |", "| --- " * (len(GRADES) + 1) + "|",
    ]
    for g, row in zip(GRADES, r["confusion_matrix"]):
        lines.append(f"| **{g}** | " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=None, help="override the checkpoint's referral threshold")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_checkpoint(args.ckpt, device)
    d = ckpt["config"]["data"]
    df = read_manifest(args.manifest)
    dl = DataLoader(FundusDataset(df, d["image_size"], False, d["ben_graham"]), 32, shuffle=False)
    y, p = predict_probs(model, dl, device)
    thr = args.threshold if args.threshold is not None else ckpt.get("referable_threshold", 0.5)
    r = full_report(y, p, thr)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(r, indent=2))
    (out / "report.md").write_text(to_markdown(r, Path(args.manifest).name))
    print(to_markdown(r, Path(args.manifest).name))


if __name__ == "__main__":
    main()
