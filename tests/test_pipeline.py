import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT)]

from drscreen.metrics import full_report, sens_spec, threshold_for_sensitivity  # noqa: E402
from drscreen.preprocess import crop_to_fundus, preprocess  # noqa: E402
from drscreen.quality import assess  # noqa: E402
from make_synthetic_data import fundus  # noqa: E402


@pytest.fixture
def img():
    return fundus(np.random.default_rng(0), 3, 400)


def test_preprocess_shape(img):
    padded = np.zeros((500, 700, 3), np.uint8); padded[50:450, 150:550] = img
    assert crop_to_fundus(padded).shape[0] == crop_to_fundus(padded).shape[1]
    out = preprocess(padded, 256)
    assert out.shape == (256, 256, 3) and out.dtype == np.uint8


def test_quality_accepts_good_and_rejects_bad(img):
    assert assess(img).gradable
    assert not assess(np.zeros((400, 400, 3), np.uint8)).gradable          # black
    import cv2
    assert not assess(cv2.GaussianBlur(img, (0, 0), 12)).gradable         # blurred
    assert not assess(img[:200, :200]).gradable                            # too small


def test_threshold_hits_target_sensitivity():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500); p = np.clip(y * 0.4 + rng.random(500) * 0.6, 0, 1)
    thr = threshold_for_sensitivity(y, p, 0.9)
    assert sens_spec(y, p, thr)["sensitivity"] >= 0.9


def test_full_report_keys():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 5, 200); probs = rng.dirichlet(np.ones(5), 200)
    r = full_report(y, probs, 0.5, n_boot=50)
    for k in ("qwk", "referable_sensitivity", "referable_specificity", "referable_auc", "confusion_matrix"):
        assert k in r


def test_api_end_to_end(tmp_path, monkeypatch):
    """Train-free: build a tiny random model checkpoint and exercise the API."""
    import cv2
    import torch
    from drscreen.model import DRNet, save_checkpoint
    cfg = {"data": {"image_size": 128, "ben_graham": True}, "model": {"backbone": "resnet18", "dropout": 0.0}}
    ck = tmp_path / "m.pt"
    torch.manual_seed(0)
    save_checkpoint(ck, DRNet("resnet18", pretrained=False), cfg, {"referable_threshold": 0.5})
    monkeypatch.setenv("DR_MODEL", str(ck)); monkeypatch.setenv("DR_DB", str(tmp_path / "s.db"))
    import importlib

    import api.main as m
    importlib.reload(m)
    from fastapi.testclient import TestClient
    c = TestClient(m.app)
    assert c.get("/health").json()["status"] == "ok"
    good = cv2.imencode(".png", fundus(np.random.default_rng(3), 2, 400))[1].tobytes()
    bad = cv2.imencode(".png", np.zeros((400, 400, 3), np.uint8))[1].tobytes()
    r = c.post("/api/screenings", data={"patient_ref": "T-1", "age": "55"},
               files={"right_eye": ("r.png", good, "image/png"), "left_eye": ("l.png", bad, "image/png")})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["refer"] is True  # ungradable left eye must force referral
    right = [e for e in j["eyes"] if e["eye"] == "right"][0]
    assert right["gradable"] and right["heatmap_png_b64"]
    assert c.get(f"/api/screenings/{j['id']}/report").status_code == 200
    assert len(c.get("/api/screenings").json()) == 1
    assert c.post("/api/screenings", data={"patient_ref": "T-2"}).status_code == 400
