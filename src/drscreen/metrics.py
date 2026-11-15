"""Clinical and competition metrics.

The number that matters for a screening programme is sensitivity for referable
DR (missed referrals = preventable blindness), at an acceptable specificity
(false referrals overload the ophthalmologist). Report both with 95% CIs.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, roc_auc_score

from . import NUM_CLASSES, REFERABLE_MIN_GRADE


def qwk(y_true, y_pred) -> float:
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def sens_spec(y_bin, p, threshold):
    y_bin = np.asarray(y_bin).astype(bool)
    pred = np.asarray(p) >= threshold
    tp = (pred & y_bin).sum(); fn = (~pred & y_bin).sum()
    tn = (~pred & ~y_bin).sum(); fp = (pred & ~y_bin).sum()
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    ppv = tp / (tp + fp) if tp + fp else float("nan")
    npv = tn / (tn + fn) if tn + fn else float("nan")
    return dict(sensitivity=float(sens), specificity=float(spec), ppv=float(ppv), npv=float(npv),
                tp=int(tp), fn=int(fn), tn=int(tn), fp=int(fp))


def threshold_for_sensitivity(y_bin, p, target: float = 0.90) -> float:
    """Threshold with sensitivity >= target that maximises sensitivity + specificity.

    Ties go to the higher sensitivity (a missed referral costs more than a false one).
    """
    y_bin = np.asarray(y_bin).astype(bool)
    p = np.asarray(p)
    if not y_bin.any():
        return 0.5
    best_thr, best_key = float(p[y_bin].min()), (-1.0, -1.0)
    for thr in np.unique(p[y_bin]):
        s = sens_spec(y_bin, p, thr)
        if s["sensitivity"] + 1e-12 < target:
            continue
        spec = 0.0 if s["specificity"] != s["specificity"] else s["specificity"]
        key = (s["sensitivity"] + spec, s["sensitivity"])
        if key > best_key:
            best_key, best_thr = key, float(thr)
    return best_thr


def bootstrap_ci(fn, *arrays, n: int = 1000, seed: int = 0, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    m = len(arrays[0]); vals = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        try:
            v = fn(*[np.asarray(a)[idx] for a in arrays])
        except ValueError:
            continue
        if v == v:  # not nan
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2))))


def full_report(y_true, probs, threshold: float, n_boot: int = 1000) -> dict:
    y_true = np.asarray(y_true); probs = np.asarray(probs)
    y_pred = probs.argmax(1)
    p_ref = probs[:, REFERABLE_MIN_GRADE:].sum(1)
    y_ref = (y_true >= REFERABLE_MIN_GRADE).astype(int)
    out = {
        "n_images": int(len(y_true)),
        "prevalence_referable": float(y_ref.mean()),
        "qwk": qwk(y_true, y_pred),
        "accuracy": float((y_true == y_pred).mean()),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES))).tolist(),
        "referable_threshold": float(threshold),
        **{f"referable_{k}": v for k, v in sens_spec(y_ref, p_ref, threshold).items()},
    }
    if 0 < y_ref.sum() < len(y_ref):
        out["referable_auc"] = float(roc_auc_score(y_ref, p_ref))
        out["referable_auc_ci95"] = bootstrap_ci(roc_auc_score, y_ref, p_ref, n=n_boot)
    out["referable_sensitivity_ci95"] = bootstrap_ci(
        lambda y, p: sens_spec(y, p, threshold)["sensitivity"], y_ref, p_ref, n=n_boot)
    out["referable_specificity_ci95"] = bootstrap_ci(
        lambda y, p: sens_spec(y, p, threshold)["specificity"], y_ref, p_ref, n=n_boot)
    return out
