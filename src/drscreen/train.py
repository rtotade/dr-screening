"""Train a DR grading model.

    python -m drscreen.train --config configs/default.yaml

Writes to screening.output_dir: best.pt (model + config + referral threshold),
history.csv, val_metrics.json.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from . import REFERABLE_MIN_GRADE
from .data import FundusDataset, balanced_sampler, grouped_split, read_manifest
from .metrics import full_report, qwk, threshold_for_sensitivity
from .model import DRNet, save_checkpoint


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    ps, ys = [], []
    for x, y in loader:
        ps.append(model(x.to(device)).softmax(-1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def train(cfg: dict) -> dict:
    d, m, t, s = cfg["data"], cfg["model"], cfg["train"], cfg["screening"]
    seed_everything(t["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(s["output_dir"]); out.mkdir(parents=True, exist_ok=True)

    df = read_manifest(d["manifest"])
    tr_df, va_df = grouped_split(df, d["val_frac"], t["seed"])
    tr_df.to_csv(out / "train_split.csv", index=False); va_df.to_csv(out / "val_split.csv", index=False)
    print(f"train={len(tr_df)} val={len(va_df)} device={device}")
    print("train grade counts:", tr_df["grade"].value_counts().sort_index().to_dict())

    tr_ds = FundusDataset(tr_df, d["image_size"], True, d["ben_graham"], d.get("cache_dir"))
    va_ds = FundusDataset(va_df, d["image_size"], False, d["ben_graham"], d.get("cache_dir"))
    sampler = balanced_sampler(tr_df["grade"].values, d.get("balance_power", 0.5))
    nw = d.get("num_workers", 0)
    tr_dl = DataLoader(tr_ds, t["batch_size"], sampler=sampler, num_workers=nw, drop_last=len(tr_ds) > t["batch_size"])
    va_dl = DataLoader(va_ds, t["batch_size"] * 2, shuffle=False, num_workers=nw)

    model = DRNet(m["backbone"], m["pretrained"], m.get("dropout", 0.3)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=t["lr"], total_steps=t["epochs"] * len(tr_dl), pct_start=0.15)
    crit = nn.CrossEntropyLoss(label_smoothing=t.get("label_smoothing", 0.0))
    use_amp = t.get("amp", True) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best, best_epoch, history = -1.0, -1, []
    for epoch in range(1, t["epochs"] + 1):
        model.train(); t0 = time.time(); losses = []
        for x, y in tr_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=use_amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            losses.append(loss.item())
        y_va, p_va = predict_probs(model, va_dl, device)
        k = qwk(y_va, p_va.argmax(1))
        row = dict(epoch=epoch, train_loss=float(np.mean(losses)), val_qwk=k, secs=round(time.time() - t0, 1))
        history.append(row); print(row)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        if k > best:
            best, best_epoch = k, epoch
            y_ref = (y_va >= REFERABLE_MIN_GRADE).astype(int)
            p_ref = p_va[:, REFERABLE_MIN_GRADE:].sum(1)
            thr = threshold_for_sensitivity(y_ref, p_ref, s["target_sensitivity"]) if y_ref.any() else 0.5
            save_checkpoint(out / "best.pt", model, cfg, {"referable_threshold": thr, "epoch": epoch, "val_qwk": k})
            report = full_report(y_va, p_va, thr, n_boot=300)
            (out / "val_metrics.json").write_text(json.dumps(report, indent=2))
        elif epoch - best_epoch >= t.get("early_stop_patience", 999):
            print(f"Early stopping at epoch {epoch} (best {best:.4f} @ {best_epoch})")
            break
    print(f"Best val QWK {best:.4f} at epoch {best_epoch}. Checkpoint: {out / 'best.pt'}")
    return json.loads((out / "val_metrics.json").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="overrides like train.epochs=3")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    for kv in args.set:
        key, val = kv.split("=", 1); node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = yaml.safe_load(val)
    train(cfg)


if __name__ == "__main__":
    main()
