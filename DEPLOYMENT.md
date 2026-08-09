# Deployment Guide

NeuroSwap is a **FastAPI + InsightFace + ffmpeg** app with **WebSockets**, video export, and ML models. Plan deployments accordingly.

| Platform | Recommended? | Notes |
|----------|--------------|-------|
| **Render (Docker)** | **Yes — primary** | WebSockets, ffmpeg, long-running process, persistent disk |
| **Vercel (serverless)** | **Experimental only** | 250 MB limit, 10–60 s timeouts, no ffmpeg, WebSockets unreliable |

---

## Render (recommended)

### Option A — Blueprint (one click)

1. Push this repo to GitHub.
2. In [Render Dashboard](https://dashboard.render.com/) → **New** → **Blueprint**.
3. Connect the repo — Render reads `render.yaml`.
4. After deploy, open the service URL. Wait ~1–2 minutes for models to load (`/health` → `"models_loaded": true`).
5. Copy the generated `NEUROSWAP_API_KEY` from Render env vars and enter it in the UI (**KEY** button) if auth is enabled.

### Option B — Manual Docker web service

1. **New** → **Web Service** → connect repo.
2. **Runtime:** Docker  
3. **Health check path:** `/health`  
4. **Environment variables** (see `.env.example`):

   | Variable | Value |
   |----------|--------|
   | `NEUROSWAP_DATA_DIR` | `/app/data` |
   | `NEUROSWAP_SERVE_UPLOADS` | `1` |
   | `NEUROSWAP_API_KEY` | (generate a secret) |
   | `INSIGHTFACE_ROOT` | `/app/data/.insightface_models` |

5. **Disk (paid plans):** mount `2 GB` at `/app/data` so uploads, exports, and model cache survive redeploys.

### Models on Render

- `buffalo_l` is extracted from `buffalo_l.zip` on first boot if present in the repo, or downloaded when network allows.
- `inswapper_128.onnx` (~525 MB) is **not** bundled — either:
  - Let the app download it on first boot (slow), or
  - Upload to persistent disk at `/app/data/.insightface_models/models/inswapper_128.onnx` via Render shell.

### Render free tier

- No persistent disk — uploads/exports are lost on redeploy.
- Service spins down after inactivity (cold start ~30–60 s).
- Use **Standard** plan + disk for production.

### Local Docker test

```bash
docker build -t neuroswap .
docker run --rm -p 8000:8000 -e NEUROSWAP_DATA_DIR=/app/data neuroswap
curl http://127.0.0.1:8000/health
```

---

## Vercel

Vercel is **not** a good fit for the full stack (ML + ffmpeg + WebSockets). Two supported patterns:

### Pattern 1 — Experimental serverless (demo / health only)

The repo includes `vercel.json` + `api/index.py` (Mangum ASGI adapter).

```bash
npm i -g vercel   # or use Vercel Git integration
vercel
```

**Limitations:**

- Deployment size exceeds Vercel limits once InsightFace/OpenCV are installed.
- Cold starts time out before models load.
- `/ws` WebSocket face swap will not work reliably.
- Video export (ffmpeg) is unavailable.

Use this only to verify routing or serve `/health` and `/docs` in a trimmed-down fork.

### Pattern 2 — Vercel frontend + Render backend (production split)

1. Deploy the **full app on Render** (see above).
2. Point your custom domain or use the Render URL directly for the UI.
3. If you host a static frontend on Vercel later, set on **Render**:

   ```
   CORS_ORIGINS=https://your-project.vercel.app
   ```

4. Configure the frontend to call the Render API/WebSocket origin (same host avoids CORS).

**Do not** proxy WebSockets through Vercel rewrites — connect the browser directly to the Render `wss://` URL.

---

## Environment variables

See [`.env.example`](.env.example) for the full list.

| Variable | Purpose |
|----------|---------|
| `PORT` | HTTP port (Render sets automatically) |
| `NEUROSWAP_API_KEY` | Optional auth for REST + WebSocket |
| `NEUROSWAP_DATA_DIR` | Persistent data root |
| `INSIGHTFACE_ROOT` | InsightFace model cache |
| `CORS_ORIGINS` | Comma-separated allowed origins |
| `NEUROSWAP_SERVE_UPLOADS` | `1` to expose `/uploads/*` |
| `MAX_VIDEO_UPLOAD_BYTES` | Upload size cap (default 1 GB) |

---

## Health checks

```bash
curl https://your-service.onrender.com/health
```

Expected when ready:

```json
{
  "status": "ok",
  "models_loaded": true,
  "models_loading": false,
  "models_fallback": false,
  "ffmpeg_available": true
}
```

---

## Post-deploy checklist

- [ ] `/health` returns `models_loaded: true` (or acceptable fallback)
- [ ] `/` loads the UI
- [ ] Upload a target face → `TARGET` status OK
- [ ] WebSocket engage works (camera / clip)
- [ ] Video export: upload clip → Scan → Render → Download MP4
- [ ] Set `NEUROSWAP_API_KEY` in production and store it in the browser via the KEY control

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `/ws` returns 404 | Ensure `websockets` or `wsproto` is installed (included in `requirements.txt`) |
| `ffmpeg_available: false` | Render Docker image includes ffmpeg; native Python runtime does not |
| Models never load | Check logs; upload `inswapper_128.onnx` to persistent disk |
| 401 on API calls | Set matching `NEUROSWAP_API_KEY` in env and browser localStorage |
| Uploads lost after deploy | Add Render persistent disk at `/app/data` |
