"""Screening service: REST API + technician web app.

    DR_MODEL=runs/default/best.pt uvicorn api.main:app --host 0.0.0.0 --port 8000

Endpoints
  GET  /health                       service + model status
  POST /api/screenings               multipart: patient_ref, age, sex, diabetes_years, left_eye, right_eye
  GET  /api/screenings               recent screenings (no images)
  GET  /api/screenings/{id}          one screening (with heatmaps)
  GET  /api/screenings/{id}/report   printable referral slip (HTML)
  GET  /                             technician web app

Privacy: only a pseudonymous patient reference is stored (no name / Aadhaar /
phone). Raw images are not stored unless DR_STORE_IMAGES=1.
"""
from __future__ import annotations

import html
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from drscreen import __version__
from drscreen.inference import Screener

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = os.environ.get("DR_MODEL", str(ROOT / "runs/default/best.pt"))
DB_PATH = os.environ.get("DR_DB", str(ROOT / "data/screenings.db"))
STORE_IMAGES = os.environ.get("DR_STORE_IMAGES", "0") == "1"
MAX_BYTES = 15 * 1024 * 1024

app = FastAPI(title="DR Screening", version=__version__)
app.mount("/static", StaticFiles(directory=ROOT / "app/static"), name="static")
_screener: Screener | None = None


def screener() -> Screener:
    global _screener
    if _screener is None:
        if not Path(MODEL_PATH).exists():
            raise HTTPException(503, f"Model not found at {MODEL_PATH}. Train one or set DR_MODEL.")
        _screener = Screener(MODEL_PATH)
    return _screener


def db() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS screenings(
        id TEXT PRIMARY KEY, created_at TEXT, patient_ref TEXT, age INTEGER, sex TEXT,
        diabetes_years REAL, site TEXT, refer INTEGER, patient_grade INTEGER, result_json TEXT)""")
    return con


@app.get("/health")
def health():
    ok = Path(MODEL_PATH).exists()
    info = screener().model_info if ok else None
    return {"status": "ok" if ok else "no_model", "version": __version__, "model": info}


@app.post("/api/screenings")
async def create_screening(
    patient_ref: str = Form(..., max_length=64),
    age: int | None = Form(None),
    sex: str | None = Form(None),
    diabetes_years: float | None = Form(None),
    site: str | None = Form(None),
    left_eye: UploadFile | None = File(None),
    right_eye: UploadFile | None = File(None),
):
    eyes = {}
    for name, up in (("left", left_eye), ("right", right_eye)):
        if up is not None and up.filename:
            data = await up.read()
            if len(data) > MAX_BYTES:
                raise HTTPException(413, f"{name} image too large")
            eyes[name] = data
    if not eyes:
        raise HTTPException(400, "Upload at least one eye image")
    try:
        result = screener().screen_patient(eyes)
    except ValueError as e:
        raise HTTPException(400, str(e))
    sid = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result.update(id=sid, created_at=now, patient_ref=patient_ref, age=age, sex=sex,
                  diabetes_years=diabetes_years, site=site)
    if STORE_IMAGES:
        d = Path(DB_PATH).parent / "images" / sid; d.mkdir(parents=True, exist_ok=True)
        for name, data in eyes.items():
            (d / f"{name}.jpg").write_bytes(data)
    with db() as con:
        con.execute("INSERT INTO screenings VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (sid, now, patient_ref, age, sex, diabetes_years, site, int(result["refer"]),
                     result["patient_grade"], json.dumps(result)))
    return result


@app.get("/api/screenings")
def list_screenings(limit: int = 50):
    with db() as con:
        rows = con.execute("SELECT id, created_at, patient_ref, site, refer, patient_grade FROM screenings "
                           "ORDER BY created_at DESC LIMIT ?", (min(limit, 500),)).fetchall()
    keys = ["id", "created_at", "patient_ref", "site", "refer", "patient_grade"]
    return [dict(zip(keys, r)) for r in rows]


def _get(sid: str) -> dict:
    with db() as con:
        row = con.execute("SELECT result_json FROM screenings WHERE id=?", (sid,)).fetchone()
    if not row:
        raise HTTPException(404, "Screening not found")
    return json.loads(row[0])


@app.get("/api/screenings/{sid}")
def get_screening(sid: str):
    return _get(sid)


@app.get("/api/screenings/{sid}/report", response_class=HTMLResponse)
def report(sid: str):
    r = _get(sid); e = html.escape
    eye_rows = ""
    for eye in r["eyes"]:
        status = e(eye["grade_name"]) if eye["gradable"] else "Ungradable: " + e("; ".join(eye["quality"]["reasons"]))
        img = f'<img src="data:image/png;base64,{eye["heatmap_png_b64"]}">' if eye.get("heatmap_png_b64") else ""
        eye_rows += f"<tr><td>{e(eye['eye'].title())}</td><td>{status}</td><td>{img}</td></tr>"
    banner = "REFER TO OPHTHALMOLOGIST" if r["refer"] else "NO REFERRAL NEEDED"
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>DR screening {e(sid)}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:760px;margin:24px auto;color:#1a2433}}
.b{{padding:12px 16px;border-radius:8px;font-weight:700;font-size:20px;color:#fff;background:{'#b42318' if r['refer'] else '#067647'}}}
table{{width:100%;border-collapse:collapse;margin:16px 0}}td,th{{border:1px solid #d0d5dd;padding:8px;vertical-align:top;text-align:left}}
img{{width:220px;border-radius:6px}}small{{color:#667085}}@media print{{button{{display:none}}}}</style></head><body>
<h2>Diabetic Retinopathy Screening Report</h2>
<p>Screening ID <b>{e(sid)}</b> · {e(r['created_at'])} · Site {e(r.get('site') or '-')}<br>
Patient ref <b>{e(r['patient_ref'])}</b> · Age {e(str(r.get('age') or '-'))} · Sex {e(r.get('sex') or '-')} · Diabetes duration {e(str(r.get('diabetes_years') or '-'))} yrs</p>
<div class="b">{banner}</div>
<p><b>Recommendation:</b> {e(r['recommendation'])}</p>
<table><tr><th>Eye</th><th>AI grade</th><th>Attention map</th></tr>{eye_rows}</table>
<p>Reviewed by (eye-care professional): ____________________ &nbsp; Date: ____________</p>
<small>{e(r['disclaimer'])} Model: {e(r['model']['backbone'])}.</small>
<p><button onclick="window.print()">Print</button></p></body></html>"""


@app.get("/")
def index():
    return FileResponse(ROOT / "app/static/index.html")
