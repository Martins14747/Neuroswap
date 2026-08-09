# Neuroswap — Live Face Swap Web UI

A cyberpunk-styled web interface for **real-time webcam face tracking + face swapping** powered by InsightFace's `buffalo_l` pipeline and `inswapper_128.onnx`. Frames stream from your browser to a FastAPI backend over WebSockets at ~15 FPS with live bounding-box / landmark overlays drawn on a HUD-style canvas layer.

> Demo mode (no real models required) → simulated face detection + HUD overlay so you can still iterate on UI/UX offline.

---

## ✨ Features

- **Real face detection / landmarking / recognition** — uses local InsightFace model files and auto-downloads them when needed; no large model binaries are committed to GitHub.
- **Webcam → WebSocket → Processed frame round-trip** — frames sent as base64 JPEG, swapped frame + face metadata returned as JSON.
- **Intensity slider (alpha blend)** — user-controlled swap strength via `cv2.addWeighted`.
- **Drag-and-drop target upload** — drag a headshot onto the dropzone. Auto-generates padded 160×160 face-thumbnails for the history strip.
- **Recent target strip (LRU, max 8)** — click any thumbnail to change the active swap target.
- **Live HUD overlay** — draws neon bboxes, 5-point landmarks, det score, swap quality %, FPS, and latency on the frontend canvas.
- **Session timeline** — append event to a running log (connect / swap target / engage / disengage).
- **Graceful fallback model classes** — if the environment can't download models (sandboxes, offline, firewalls, no GitHub SSL egress) the app boots in "DEMO / PARTIAL" mode with an info banner instead of crashing.
- **Zero-friction model install** — `main.py` auto-installs `buffalo_l.zip` into either `INSIGHTFACE_ROOT` env or `~/.insightface/models/` on first boot.

---

## 🧠 Architecture

```
┌─────────── Web browser ─────────────┐
│  index.html (Cyberpunk HUD)         │
│    ├─ getUserMedia() → canvas       │
│    ├─ WebSocket('/ws')              │◀┐
│    ├─ Intensity slider              │ │ JSON: frame b64 + intensity
│    └─ HUD overlay canvas            │ │
└─────────────────────────────────────┘ │
              ▲                         │
              │ WS JSON                 │
              ▼                         │
┌──────── FastAPI (main.py) ────────────┘
│  /           → index.html            │
│  /upload-target                      │ → thumbnail + LRU strip +
│  /select-target/{id}                 │   register face_embedding
│  /targets  /target-status            │
│  /ws                                │←  receive frame
│    FaceAnalysis.det_10g → bboxes     │
│    Inswapper128.get(face, target)    │   swap
│    cv2.addWeighted(intensity)        │
│    → processed_frame (b64) + faces[] │
└─────────────────────────────────────┘
```

### Backend modules

| File | Purpose |
|------|---------|
| [main.py](file:///home/me/Documents/SOAIR/face-swap-web/main.py) | FastAPI app, WebSocket frame loop, model lifecycle, uploads, target history |
| [templates/index.html](file:///home/me/Documents/SOAIR/face-swap-web/templates/index.html) | Single-file frontend (HTML/CSS/JS) — HUD, canvases, sliders, DnD |
| [test_backend.py](file:///home/me/Documents/SOAIR/face-swap-web/test_backend.py) | 58 end-to-end tests — REST, WS, robustness, bad messages, LRU cap |
| [requirements.txt](file:///home/me/Documents/SOAIR/face-swap-web/requirements.txt) | Pinned dependency versions (freeze) |

---

## 🚀 Quick Start

### 1. Install

```bash
git clone <your-fork> && cd face-swap-web
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Bundle models

If you already have model files you can skip download entirely.

**Option A — local model cache (preferred):**

If you already have a local InsightFace model archive or extracted model directory, place it in the standard InsightFace locations (for example `~/.insightface/models/` or `$INSIGHTFACE_ROOT/models/`). On first boot, `main.py` will use those files directly and avoid any GitHub-sized assets.

**Option B — download (requires GitHub HTTPS access):**

```bash
python3 -c "
import insightface
from insightface.app import FaceAnalysis
fa = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
fa.prepare(ctx_id=0, det_size=(640,640))
swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, providers=['CPUExecutionProvider'])
print('OK')
"
```

**Option C — offline/DEMO mode:**

Skip all downloads — `main.py` boots with `_FallbackAnalyzer` and `_FallbackSwapper` so the UI + uploads still work for UI testing. The status pill shows `AI CORE (PARTIAL)` or `AI CORE (DEMO)` with a cyan info banner.

### 3. Run

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --log-level info
```

Open **http://127.0.0.1:8000**

Wait for the "AI CORE" pill to turn green (first boot extracts the model zip — ~30s). Then:

1. Click / drop a portrait image onto **INPUT TARGET** (face must be clearly visible).
2. When **TARGET** pill says `LOADED` and a thumbnail appears, click **ENGAGE SWAP**.
3. Allow webcam access → the HUD canvas should render your face with a neon bbox + the target face blended on top.
4. Drag the **INTENSITY** slider to control blending strength in real time.

### 4. Verify

```bash
./venv/bin/python test_backend.py          # full suite
./venv/bin/python test_backend.py --host 127.0.0.1 --port 8000  # target live server
```

Expected: **56–58 / 58 tests pass**. (Two static tests fail only if the sample `Keanu Reeves.jpeg` isn't present — that's normal, it's intentionally gitignored so user privacy isn't violated.)

---

## 📁 Repo Layout

```
face-swap-web/
├── main.py                     # FastAPI + InsightFace integration + WebSocket loop
├── backend.py                  # Legacy module (kept for reference)
├── test_backend.py             # pytest-style end-to-end suite (no pytest needed)
├── requirements.txt            # Pinned install deps
├── .gitignore
├── templates/
│   └── index.html              # Single-file HUD frontend
├── uploads/                    # Persisted user uploads (gitignored except .gitkeep)
└── .insightface_models/        # Auto-extract cache (gitignored)
```

---

## 🔌 API

### REST

| Method | Path | Body | Description |
|--------|------|------|-------------|
| `GET`    | `/`                 | — | Render HUD |
| `GET`    | `/target-status`    | — | `{models_loaded, models_loading, models_fallback, models_error, targets, current_target_id, has_target}` |
| `GET`    | `/targets`          | — | Same as above but only targets + current |
| `POST`   | `/upload-target`    | `multipart/form-data: file=<image>` | Accept a target photo. Returns 200 + `target_id`, `thumb`, `detection_score`, or 400 `No face detected`. |
| `POST`   | `/select-target/{id}` | — | Switch active target from recent list |
| `GET`    | `/uploads/<file>`   | — | Static persisted uploads |
| `DELETE` | `/targets/<id>`     | — | Delete a recent target |

### WebSocket `ws://<host>/ws`

**Client → server:**
```jsonc
{"type":"frame","image":"data:image/jpeg;base64,..."}  // webcam frame
{"type":"set_intensity","value":0.65}                   // blend strength 0–1
{"type":"ping","id":42}
```

**Server → client:**
```jsonc
{"type":"welcome","intensity":1.0,"has_target":true,
 "models_loaded":true,"models_fallback":true,"models_error":"...","targets":[...]}

{"type":"processed_frame",
 "image":"data:image/jpeg;base64,...",
 "faces":[{"bbox":[x1,y1,x2,y2], "score":0.94,
           "landmarks":[[x,y]*5], "swapped":true}],
 "intensity":0.65, "latency_ms":132, "fps":14.2,
 "swap_quality_pct":92}

{"type":"error","error":"no_target","message":"No target face selected."}
```

---

## 🪪 Model Licensing

This repository ships `buffalo_l.zip` **for development convenience only**. The .onnx weights inside are released by the InsightFace project under their original license — see the [InsightFace GitHub](https://github.com/deepinsight/insightface) for details. If redistribution of the `inswapper_128.onnx` or `buffalo_l` weights conflicts with your policies, delete `buffalo_l.zip` from your fork — the app will still work in DEMO mode (see step 2C above) and prompt users to download models on first run themselves.

---

## 🧪 Known limitations (informational, not bugs)

- **`inswapper_128.onnx` (~525 MB)** is NOT bundled with the repo (too large for GitHub). It's fetched from GitHub releases by default; if your network blocks it you'll get PARTIAL mode (detector real, swapper = placeholder neon overlay). Place a local copy at:
  ```
  ~/.insightface/models/inswapper_128.onnx
  ```
  or
  ```
  $PROJECT_ROOT/.insightface_models/models/inswapper_128.onnx
  ```
  and `main.py` will pick it up on next boot with `download=False`.
- CPU only — GPU is faster, but `providers=['CPUExecutionProvider']` keeps install friction minimal for a dev setup.
- Sandbox / restricted egress: when `insightface` can't reach GitHub for downloads, the partial/demo path is used intentionally.

---

## 🖼 Screenshots

*(replace with your own — HUD shows neon bboxes, cyan slider, target thumbnail strip, session timeline on right)*

---

## 🤝 Contributing

1. `git checkout -b feature/<name>`
2. Make your changes
3. Run `./venv/bin/python test_backend.py` → **56+ / 58**
4. Commit + push + open PR

---

## Deploy to production

See **[DEPLOYMENT.md](DEPLOYMENT.md)** for Render (Docker, recommended) and Vercel (experimental) setup.
