"""Inference: quality gate -> grade -> referral decision -> Grad-CAM heatmap.

Patient-level decision follows standard screening practice: the patient is
graded by the worse eye, and any ungradable eye triggers a referral.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from . import GRADES, REFERABLE_MIN_GRADE
from .data import build_transforms
from .model import load_checkpoint
from .preprocess import crop_to_fundus, load_rgb, preprocess
from .quality import QualityThresholds, assess


@dataclass
class EyeResult:
    eye: str
    gradable: bool
    quality: dict
    grade: int | None = None
    grade_name: str | None = None
    probabilities: dict | None = None
    referable_probability: float | None = None
    referable: bool | None = None
    heatmap_png_b64: str | None = None


def recommendation(grade: int | None, gradable: bool) -> str:
    if not gradable:
        return "Ungradable image: retake photo; if still ungradable, refer to ophthalmologist."
    return {
        0: "No DR detected. Rescreen in 12 months; continue glycaemic and BP control.",
        1: "Mild NPDR. Rescreen in 6–12 months; optimise glycaemic and BP control.",
        2: "Moderate NPDR: REFER to ophthalmologist within 3 months.",
        3: "Severe NPDR: URGENT referral to ophthalmologist within 4 weeks.",
        4: "Proliferative DR: URGENT referral to ophthalmologist within 1–2 weeks.",
    }[grade]


class Screener:
    def __init__(self, ckpt_path: str, device: str | None = None, quality: QualityThresholds | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, ckpt = load_checkpoint(ckpt_path, self.device)
        self.cfg = ckpt["config"]
        self.threshold = float(ckpt.get("referable_threshold", 0.5))
        self.size = self.cfg["data"]["image_size"]
        self.ben = self.cfg["data"]["ben_graham"]
        self.tf = build_transforms(self.size, train=False)
        self.qt = quality or QualityThresholds()
        self.model_info = {"backbone": self.cfg["model"]["backbone"], "epoch": ckpt.get("epoch"),
                           "val_qwk": ckpt.get("val_qwk"), "referable_threshold": self.threshold}

    def _gradcam(self, x: torch.Tensor, cls: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        feats = self.model.features(x)
        feats.retain_grad()
        logits = self.model.head(self.model.pool(feats))
        logits[0, cls].backward()
        w = feats.grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * feats).sum(1))[0].detach().cpu().numpy()
        cam = cam / (cam.max() + 1e-8)
        return cv2.resize(cam, (self.size, self.size))

    def screen_eye(self, image, eye: str = "unknown", with_heatmap: bool = True) -> EyeResult:
        rgb = load_rgb(image) if isinstance(image, (bytes, bytearray, str)) else image
        q = assess(rgb, self.qt)
        res = EyeResult(eye=eye, gradable=q.gradable, quality=q.to_dict())
        if not q.gradable:
            return res
        proc = preprocess(rgb, self.size, self.ben)
        x = self.tf(proc).unsqueeze(0).to(self.device)
        if with_heatmap:
            with torch.enable_grad():
                logits = self.model(x)
        else:
            with torch.no_grad():
                logits = self.model(x)
        probs = logits.softmax(-1)[0].detach().cpu().numpy()
        p_ref = float(probs[REFERABLE_MIN_GRADE:].sum())
        grade = int(probs.argmax())
        referable = p_ref >= self.threshold
        # Keep grade consistent with the (sensitivity-tuned) referral decision
        if referable and grade < REFERABLE_MIN_GRADE:
            grade = REFERABLE_MIN_GRADE + int(probs[REFERABLE_MIN_GRADE:].argmax())
        res.grade, res.grade_name = grade, GRADES[grade]
        res.probabilities = {g: round(float(p), 4) for g, p in zip(GRADES, probs)}
        res.referable_probability, res.referable = round(p_ref, 4), bool(referable)
        if with_heatmap:
            cam = self._gradcam(x, grade)
            base = cv2.resize(crop_to_fundus(rgb), (self.size, self.size))
            heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
            overlay = (0.6 * base + 0.4 * heat).astype(np.uint8)
            ok, png = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            res.heatmap_png_b64 = base64.b64encode(png.tobytes()).decode() if ok else None
        return res

    def screen_patient(self, eyes: dict) -> dict:
        """eyes: {"left": image, "right": image} (either may be missing)."""
        results = [self.screen_eye(img, eye) for eye, img in eyes.items() if img is not None]
        if not results:
            raise ValueError("No images supplied")
        any_ungradable = any(not r.gradable for r in results)
        graded = [r for r in results if r.gradable]
        worst = max((r.grade for r in graded), default=None)
        referable = any_ungradable or any(r.referable for r in graded)
        if worst is not None and not any_ungradable:
            rec = recommendation(worst, True)
        elif worst is not None and worst >= REFERABLE_MIN_GRADE:
            rec = recommendation(worst, True) + " (One eye ungradable.)"
        else:
            rec = recommendation(None, False)
        return {
            "eyes": [r.__dict__ for r in results],
            "patient_grade": worst,
            "patient_grade_name": GRADES[worst] if worst is not None else None,
            "refer": bool(referable),
            "recommendation": rec,
            "model": self.model_info,
            "disclaimer": "Decision-support output for research use. Not a diagnosis; must be confirmed by a qualified eye-care professional.",
        }
