import os as _os
import sys as _sys
import site as _site
from pathlib import Path as _Path

BASE_DIR = _Path(__file__).parent
_DATA_ROOT = _Path(_os.environ.get("NEUROSWAP_DATA_DIR", str(BASE_DIR)))
if _os.environ.get("NEUROSWAP_DATA_DIR"):
    _if_root = _DATA_ROOT / ".insightface_models"
    _os.environ.setdefault("INSIGHTFACE_ROOT", str(_if_root))
    try:
        _if_root.mkdir(parents=True, exist_ok=True)
        (_DATA_ROOT / "uploads").mkdir(parents=True, exist_ok=True)
        (_DATA_ROOT / "videos").mkdir(parents=True, exist_ok=True)
        (_DATA_ROOT / "outputs").mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

# ── Kali / venv cross-pollination bootstrap ─────────────────────────────────
# The venv (./venv/) has opencv / fastapi / insightface from requirements.txt,
# but WebSocket transports (websockets / real wsproto) live in the system
# packages at /usr/lib/python3/dist-packages (installed via apt).  Venv
# isolation hides them from uvicorn, which means /ws returns HTTP 404 unless
# the vendored wsproto_stub.shim runs — but the stub can't satisfy uvicorn's
# wsproto_impl.py state-machine API (it only exposes raw RFC 6455 framing).
# Fix: append ALL site-packages paths (system + user) BEFORE any imports so
# uvicorn finds the real websockets / wsproto packages regardless of which
# interpreter is used. This matches what test_backend.py already does.
_EXTRA_PATHS = []
try:
    _EXTRA_PATHS.extend(_site.getsitepackages())
except Exception:
    pass
try:
    _usr_dist = "/usr/lib/python{}.{}".format(_sys.version_info.major,
                                              _sys.version_info.minor)
    if _usr_dist + "/dist-packages" not in _EXTRA_PATHS:
        _EXTRA_PATHS.append(_usr_dist + "/dist-packages")
    if "/usr/lib/python3/dist-packages" not in _EXTRA_PATHS:
        _EXTRA_PATHS.append("/usr/lib/python3/dist-packages")
except Exception:
    pass
for _p in _EXTRA_PATHS:
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.append(_p)

# ── neuroswap offline/restricted-network bootstrap ──────────────────────────
# Project operates in TLS-intercepted environments where PyPI access (direct or
# via mirrors) fails. uvicorn requires either the `wsproto` or the `websockets`
# package to perform HTTP->WS protocol upgrades. Without either, every /ws
# request returns HTTP 404 (see AutoWebSocketsProtocol = None in uvicorn's
# websockets/auto.py). We ship a self-contained, pure-stdlib wsproto-compatible
# shim in `wsproto_stub.py` (RFC 6455 client+server framing, 64 MB cap, PING
# auto-PONG, base64/hashlib/struct only) and expose it as the real `wsproto`
# module before any uvicorn imports run. This is a no-op if the REAL wsproto
# (>= v1) is already installed (e.g. user ran `pip install wsproto` in a
# network-enabled environment later).
_SHIM_INSTALLED = False
def _install_wsproto_shim_if_needed() -> bool:
    global _SHIM_INSTALLED

    def _real_wsproto_available() -> bool:
        try:
            import importlib as _il
            if "wsproto" in _sys.modules:
                _w = _sys.modules["wsproto"]
            else:
                _w = _il.import_module("wsproto")
            if hasattr(_w, "ConnectionType") or (
                hasattr(_w, "Connection") and hasattr(_w, "WSConnectionType")
            ):
                return True
        except Exception:
            return False
        return False

    if _real_wsproto_available():
        return False

    try:
        import importlib.util as _ilu
        shim_path = str(BASE_DIR / "wsproto_stub.py")
        if not _os.path.isfile(shim_path):
            return False
        _spec = _ilu.spec_from_file_location("wsproto", shim_path)
        if _spec is None or _spec.loader is None:
            return False
        _mod = _ilu.module_from_spec(_spec)
        _sys.modules["wsproto"] = _mod
        _spec.loader.exec_module(_mod)
        _SHIM_INSTALLED = True
        return True
    except Exception as _e:
        print(f"[WARN] neuroswap wsproto shim failed to install: {_e}", file=_sys.stderr)
        return False

_SHIM_USED = _install_wsproto_shim_if_needed()

_default_root = _os.environ.get("INSIGHTFACE_ROOT")
if not _default_root:
    _candidates = [
        str(_DATA_ROOT / ".insightface_models"),
        str(BASE_DIR / ".insightface_models"),
        str(_Path(_os.path.expanduser("~/.insightface"))),
    ]
    for _root in _candidates:
        _buffalo_dir = _Path(_root) / "models" / "buffalo_l"
        if _buffalo_dir.exists() and any(p.suffix.lower() == ".onnx" for p in _buffalo_dir.iterdir()):
            _os.environ["INSIGHTFACE_ROOT"] = _root
            break

from fastapi import FastAPI, WebSocket, Request, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, Response, FileResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

import neuroswap_storage as _storage
import base64
import json
import os
import uuid
import time
import threading
import mimetypes
import zipfile
import shutil
shutil_which = shutil.which
import hashlib
import asyncio
import re
import anyio
from pathlib import Path
from collections import OrderedDict


class _FallbackFace:
    def __init__(self, idx=0, img_h=400, img_w=400):
        cx = img_w // 2 + (idx - 0.5) * 80
        cy = img_h // 2
        bw = int(min(img_w, img_h) * 0.45)
        bh = int(bw * 1.15)
        x1 = max(0, cx - bw // 2)
        y1 = max(0, cy - bh // 2)
        x2 = min(img_w, cx + bw // 2)
        y2 = min(img_h, cy + bh // 2)
        self.bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
        self.det_score = 0.93
        self.kps = np.array([
            [cx - bw * 0.18, cy - bh * 0.10],
            [cx + bw * 0.18, cy - bh * 0.10],
            [cx, cy + bh * 0.02],
            [cx - bw * 0.14, cy + bh * 0.22],
            [cx + bw * 0.14, cy + bh * 0.22]
        ], dtype=np.float32)


class _FallbackAnalyzer:
    def prepare(self, *a, **kw):
        pass

    def get(self, img):
        if img is None:
            return []
        H, W = img.shape[:2]
        if H < 100 or W < 100:
            return []
        return [_FallbackFace(0, H, W)]


class _FallbackSwapper:
    def get(self, frame, face, target_face, paste_back=True):
        out = frame.copy()
        try:
            b = face.bbox.astype(int)
            x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
            H, W = out.shape[:2]
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(W, x2), min(H, y2)
            if x2c > x1c and y2c > y1c:
                overlay = out[y1c:y2c, x1c:x2c].copy()
                border_color = (128, 0, 255)
                cv2.rectangle(overlay, (0, 0), (overlay.shape[1] - 1, overlay.shape[0] - 1),
                              border_color, 2)
                for i in range(0, overlay.shape[1], 6):
                    cv2.line(overlay, (i, 0), (i, overlay.shape[0]), (255, 43, 214), 1)
                for j in range(0, overlay.shape[0], 6):
                    cv2.line(overlay, (0, j), (overlay.shape[1], j), (0, 240, 255), 1)
                out[y1c:y2c, x1c:x2c] = cv2.addWeighted(
                    overlay, 0.35, out[y1c:y2c, x1c:x2c], 0.65, 0)
                cv2.rectangle(out, (x1c, y1c), (x2c, y2c), border_color, 2)
        except Exception:
            pass
        return out


def _install_local_buffalo_zip():
    zip_path = BASE_DIR / "buffalo_l.zip"
    project_models_root = BASE_DIR / ".insightface_models"
    candidates = []
    home_models = Path(os.path.expanduser("~/.insightface/models"))
    candidates.append((home_models, False))
    candidates.append((project_models_root, True))
    for models_root, is_project_local in candidates:
        target_dir = models_root / "buffalo_l"
        if target_dir.exists() and any(target_dir.iterdir()):
            if is_project_local:
                os.environ["INSIGHTFACE_ROOT"] = str(models_root.parent)
            return True
    if not zip_path.exists():
        return False
    for models_root, is_project_local in candidates:
        try:
            models_root.mkdir(parents=True, exist_ok=True)
            target_dir = models_root / "buffalo_l"
            extract_tmp = models_root / "_buffalo_l_extract_tmp"
            if extract_tmp.exists():
                shutil.rmtree(extract_tmp, ignore_errors=True)
            with zipfile.ZipFile(str(zip_path), "r") as zf:
                zf.extractall(str(extract_tmp))
            inner = extract_tmp / "buffalo_l"
            src = inner if inner.exists() else extract_tmp
            if target_dir.exists():
                shutil.rmtree(target_dir, ignore_errors=True)
            shutil.move(str(src), str(target_dir))
            if extract_tmp.exists() and extract_tmp != target_dir:
                shutil.rmtree(extract_tmp, ignore_errors=True)
            ok = target_dir.exists() and any(target_dir.iterdir())
            if ok and is_project_local:
                os.environ["INSIGHTFACE_ROOT"] = str(models_root.parent)
            if ok:
                return True
        except PermissionError:
            continue
        except Exception as e:
            print(f"[WARN] Could not install local buffalo_l.zip to {models_root}: {e}")
            continue
    return False

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

# ── WebSocket transport startup diagnostic ───────────────────────────────────
def _neuroswap_report_ws_transport() -> str:
    """Run after app construction, BEFORE serving. Forces uvicorn auto-detection
    to re-evaluate with the shim installed, and prints a visible banner.

    Returns: short status string for tests.
    """
    global _SHIM_INSTALLED, _SHIM_USED
    import sys as _s
    import warnings as _w
    ws_proto = None
    ws_src = "UNKNOWN"
    try:
        # Force uvicorn.auto to RE-EVALUATE (the top-level module-level try/except
        # ran at import time — may have been evaluated before our shim got in).
        import importlib as _il
        try:
            from uvicorn.protocols.websockets import auto as _auto_mod
            _il.reload(_auto_mod)
            ws_proto = getattr(_auto_mod, "AutoWebSocketsProtocol", None)
        except Exception:
            ws_proto = None
        if ws_proto is not None:
            # Inspect __module__ / name to determine real implementation
            try:
                cls_name = ws_proto.__name__
                mod_name = ws_proto.__module__
                if "websockets_sansio" in mod_name:
                    ws_src = "websockets"
                elif "wsproto_impl" in mod_name:
                    try:
                        import wsproto as _wp
                        if getattr(_wp, "__version__", "").startswith("1.3.0.stub-neuroswap"):
                            ws_src = "STDLIB-SHIM"  # ours
                        else:
                            ws_src = "wsproto"
                    except Exception:
                        ws_src = "wsproto"
                else:
                    ws_src = f"{mod_name}.{cls_name}"
            except Exception:
                ws_src = f"{type(ws_proto).__name__}"
        else:
            ws_src = "NONE!!!"
    except Exception:
        pass

    banner = (
        "╔══════════════════════════════════════════════════════════════════╗\n"
        f"║  NEUROSWAP WEBSOCKET TRANSPORT: {ws_src:<29} ║\n"
        "║   (missing 'wsproto' or 'websockets' → /ws returns 404!)       ║\n"
        "╚══════════════════════════════════════════════════════════════════╝"
    )
    if ws_src == "NONE!!!" or ws_proto is None:
        _w.warn(
            "NO WEBSOCKET TRANSPORT AVAILABLE. WebSocket /ws endpoint will return HTTP 404. "
            "The vendored wsproto_stub.py shim should have auto-loaded; check stderr above. "
            "If needed, run: pip install wsproto",
            RuntimeWarning,
        )
        print(banner.replace("STDLIB-SHIM", "NONE - CRITICAL - /ws BROKEN").replace("WEBSOCKET TRANSPORT:", "WEBSOCKET TRANSPORT: ❌ "), file=_s.stderr, flush=True)
        return "NONE"
    # When shim is used, print a colourful positive banner so the restricted-deploy user knows it's working
    if ws_src == "STDLIB-SHIM":
        ok_banner = (
            "\n"
            "\x1b[38;5;81m╔══════════════════════════════════════════════════════════════════╗\x1b[0m\n"
            "\x1b[38;5;81m║\x1b[0m \x1b[1;97mNEUROSWAP WEBSOCKET: STDLIB-SHIM (offline-safe)\x1b[0m  \x1b[38;5;207mONLINE ✓\x1b[0m     \x1b[38;5;81m║\x1b[0m\n"
            "\x1b[38;5;81m║\x1b[0m  Pure-python RFC 6455 implementation, no pip install needed     \x1b[38;5;81m║\x1b[0m\n"
            "\x1b[38;5;81m╚══════════════════════════════════════════════════════════════════╝\x1b[0m"
        )
        print(ok_banner, file=_s.stderr, flush=True)
    else:
        ok_banner = (
            "\n"
            "\x1b[38;5;81m╔══════════════════════════════════════════════════════════════════╗\x1b[0m\n"
            f"\x1b[38;5;81m║\x1b[0m \x1b[1;97mNEUROSWAP WEBSOCKET:\x1b[0m {ws_src:<26} \x1b[38;5;46mONLINE ✓\x1b[0m      \x1b[38;5;81m║\x1b[0m\n"
            "\x1b[38;5;81m╚══════════════════════════════════════════════════════════════════╝\x1b[0m"
        )
        print(ok_banner, file=_s.stderr, flush=True)
    return ws_src or "OK"

_WS_TRANSPORT = _neuroswap_report_ws_transport()

_API_KEY = os.environ.get("NEUROSWAP_API_KEY") or None

def _require_api_key(conn):
    if not _API_KEY:
        return
    headers = getattr(conn, "headers", {}) or {}
    hdr = headers.get("x-api-key") or ""
    auth = headers.get("authorization") or ""
    qp = getattr(conn, "query_params", {}) or {}
    qk = ""
    try:
        qk = qp.get("api_key") or qp.get("x_api_key") or ""
    except Exception:
        qk = ""
    ok = hdr == _API_KEY or auth == f"Bearer {_API_KEY}" or qk == _API_KEY
    if not ok:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/openapi.json", include_in_schema=False)
async def openapi_json(request: Request):
    _require_api_key(request)
    return JSONResponse(app.openapi())

_CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = _DATA_ROOT / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR = _DATA_ROOT / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = _DATA_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_TARGETS_INDEX_FILENAME = "targets_index.json"
_targets_index_lock = threading.Lock()
_targets_index_cache = None


def _targets_index_path() -> _Path:
    return (UPLOAD_DIR / _TARGETS_INDEX_FILENAME).resolve()


def _load_targets_index() -> dict:
    global _targets_index_cache
    with _targets_index_lock:
        if isinstance(_targets_index_cache, dict):
            return _targets_index_cache
        idx = {}
        if _storage.is_s3_enabled():
            try:
                resp = _storage.get_stream("uploads", _TARGETS_INDEX_FILENAME)
                body = resp.get("Body")
                raw = body.read() if body is not None else b""
                try:
                    if body is not None:
                        body.close()
                except Exception:
                    pass
                j = json.loads((raw or b"{}").decode("utf-8", "replace") or "{}")
                idx = dict((j.get("targets") or {}) if isinstance(j, dict) else {})
            except Exception:
                idx = {}
        else:
            p = _targets_index_path()
            if p.exists():
                try:
                    j = json.loads(p.read_text(encoding="utf-8") or "{}")
                    idx = dict((j.get("targets") or {}) if isinstance(j, dict) else {})
                except Exception:
                    idx = {}
        _targets_index_cache = idx
        return _targets_index_cache


def _save_targets_index(idx: dict) -> None:
    global _targets_index_cache
    if not isinstance(idx, dict):
        raise RuntimeError("Invalid targets index")
    payload = json.dumps({"version": 1, "targets": idx}, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with _targets_index_lock:
        _targets_index_cache = idx
        if _storage.is_s3_enabled():
            _storage.put_bytes("uploads", _TARGETS_INDEX_FILENAME, payload, "application/json")
            return
        p = _targets_index_path()
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_bytes(payload)
        _os.replace(str(tmp), str(p))
TEMPLATES_DIR = BASE_DIR / "templates"
INDEX_HTML_CACHE = None

if os.environ.get("NEUROSWAP_SERVE_UPLOADS", "").strip().lower() in ("1", "true", "yes", "on"):
    app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

@app.get("/static-root/wsproto_stub.py", include_in_schema=False)
async def wsproto_stub_static():
    stub_path = BASE_DIR / "wsproto_stub.py"
    if not stub_path.exists():
        raise HTTPException(status_code=404, detail="wsproto_stub.py not found")
    data = stub_path.read_bytes()
    return Response(content=data, media_type="text/plain; charset=utf-8")

_FAVICON_B64_CACHE = None
_FAVICON_ETAG = None
_FAVICON_FALLBACK_BYTES = None


def _build_favicon_fallback() -> bytes:
    import struct
    import zlib

    SZ = 32
    raw = bytearray()
    CYAN = (0, 240, 255)
    MAGENTA = (255, 43, 214)
    DARK = (8, 12, 24)
    for y in range(SZ):
        raw.append(0)
        for x in range(SZ):
            dx = x - SZ / 2
            dy = y - SZ / 2
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < SZ / 2 - 1:
                if abs(dist - SZ / 4) < 1.5:
                    c = MAGENTA
                elif abs(dx) < 1.5 and abs(dy) < SZ / 3:
                    c = CYAN
                elif abs(dy) < 1.5 and abs(dx) < SZ / 3:
                    c = CYAN
                else:
                    c = DARK
            else:
                c = (0, 0, 0)
            raw.extend(c)

    def chunk(tag, data):
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", SZ, SZ, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes(raw), 9)
    png = sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")
    return png


def _build_favicon_bytes() -> bytes:
    global _FAVICON_B64_CACHE, _FAVICON_ETAG, _FAVICON_FALLBACK_BYTES
    if _FAVICON_B64_CACHE is not None:
        return _FAVICON_B64_CACHE
    try:
        SZ = 64
        img = np.zeros((SZ, SZ, 4), dtype=np.uint8)
        CYAN = (0, 240, 255, 255)
        MAGENTA = (255, 43, 214, 255)
        YELLOW = (255, 230, 77, 255)
        DARK = (8, 12, 24, 255)
        BORDER_W = 3

        img[:, :] = DARK

        pts_outer = np.array([
            [SZ // 2, BORDER_W],
            [SZ - BORDER_W - 1, SZ * 2 // 7],
            [SZ - BORDER_W - 1, SZ * 5 // 7],
            [SZ // 2, SZ - BORDER_W - 1],
            [BORDER_W, SZ * 5 // 7],
            [BORDER_W, SZ * 2 // 7],
        ], dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(img, [pts_outer], (14, 22, 42, 255))
        cv2.polylines(img, [pts_outer], True, CYAN, BORDER_W, lineType=cv2.LINE_AA)

        inner = SZ // 2 - 2
        pts_inner = np.array([
            [SZ // 2, BORDER_W + 6],
            [SZ - BORDER_W - 7, SZ * 2 // 7],
            [SZ - BORDER_W - 7, SZ * 5 // 7],
            [SZ // 2, SZ - BORDER_W - 7],
            [BORDER_W + 6, SZ * 5 // 7],
            [BORDER_W + 6, SZ * 2 // 7],
        ], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts_inner], True, MAGENTA, 1, lineType=cv2.LINE_AA)

        cx, cy = SZ // 2, SZ // 2
        eye_dx, eye_dy = 11, 4
        eye_r = 5
        cv2.circle(img, (cx - eye_dx, cy - eye_dy), eye_r, YELLOW, -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (cx + eye_dx, cy - eye_dy), eye_r, YELLOW, -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (cx - eye_dx + 1, cy - eye_dy - 1), 2, (255, 255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (cx + eye_dx + 1, cy - eye_dy - 1), 2, (255, 255, 255, 255), -1, lineType=cv2.LINE_AA)

        scan_y = int(SZ * 0.28)
        for dy in (0, SZ - 2 * BORDER_W - 2):
            y = BORDER_W + 1 + dy
            cv2.line(img, (BORDER_W + 2, y), (SZ - BORDER_W - 3, y), (255, 43, 214, 90), 1)
        for y in range(scan_y, SZ - BORDER_W - 2, 6):
            cv2.line(img, (BORDER_W + 4, y), (SZ - BORDER_W - 5, y), (0, 240, 255, 28), 1)

        cv2.line(img, (cx, SZ // 2 + 2), (cx - 6, SZ // 2 + 14), CYAN, 2, lineType=cv2.LINE_AA)
        cv2.line(img, (cx, SZ // 2 + 2), (cx + 6, SZ // 2 + 14), CYAN, 2, lineType=cv2.LINE_AA)
        cv2.line(img, (cx - 6, SZ // 2 + 14), (cx + 6, SZ // 2 + 14), MAGENTA, 2, lineType=cv2.LINE_AA)

        png_params = [int(getattr(cv2, "IMWRITE_PNG_COMPRESSION", 16)), 5]
        ok, buf = cv2.imencode(".png", img, png_params)
        if not ok:
            raise RuntimeError("cv2 imencode failed")
        png = buf.tobytes()
    except Exception:
        if _FAVICON_FALLBACK_BYTES is None:
            _FAVICON_FALLBACK_BYTES = _build_favicon_fallback()
        png = _FAVICON_FALLBACK_BYTES

    _FAVICON_B64_CACHE = png
    _FAVICON_ETAG = '"' + hashlib.md5(png).hexdigest() + '"'
    return png


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    global _FAVICON_ETAG, _FAVICON_FALLBACK_BYTES
    try:
        data = _build_favicon_bytes()
    except Exception:
        if _FAVICON_FALLBACK_BYTES is None:
            _FAVICON_FALLBACK_BYTES = _build_favicon_fallback()
        data = _FAVICON_FALLBACK_BYTES
        _FAVICON_ETAG = '"' + hashlib.md5(data).hexdigest() + '"'
    etag = _FAVICON_ETAG if _FAVICON_ETAG is not None else '"neuroswap-favicon"'
    headers = {
        "Cache-Control": "public, max-age=604800, immutable",
        "ETag": etag,
    }
    return Response(content=data, media_type="image/png", headers=headers)


_DOCS_HTML_CACHE = None


def _build_docs_html() -> str:
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Neuroswap API · Reference</title>
<style>
  :root{
    --bg0:#05070f;
    --bg1:#0a0f20;
    --bg2:#0f1630;
    --cyan:#00f0ff;
    --magenta:#ff2bd6;
    --yellow:#ffe64d;
    --green:#3dffa5;
    --red:#ff4757;
    --text:#d0e1ff;
    --dim:#5f7aa8;
    --line:rgba(0,240,255,0.18);
  }
  *{box-sizing:border-box;margin:0;padding:0}
  html,body{background:var(--bg0);color:var(--text);font-family:'Segoe UI',system-ui,-apple-system,sans-serif;min-height:100vh}
  body{
    background:
      radial-gradient(1200px 600px at 80% -10%, rgba(255,43,214,0.08), transparent 60%),
      radial-gradient(900px 500px at -10% 30%, rgba(0,240,255,0.08), transparent 60%),
      linear-gradient(180deg, var(--bg0), var(--bg1));
  }
  header{
    padding:24px 32px;
    border-bottom:1px solid var(--line);
    backdrop-filter: blur(8px);
    position:sticky;top:0;z-index:10;
    background:rgba(5,7,15,0.7);
  }
  .brand{
    display:flex;align-items:center;gap:14px;
  }
  .logo{
    width:38px;height:38px;border-radius:10px;
    background:linear-gradient(135deg,var(--cyan),var(--magenta));
    box-shadow:0 0 18px rgba(0,240,255,0.4), inset 0 0 10px rgba(255,255,255,0.2);
    position:relative;
  }
  .logo::after{
    content:'';position:absolute;inset:6px;border-radius:6px;
    background:var(--bg0);
    display:flex;
  }
  .brand h1{
    font-size:20px;letter-spacing:4px;font-weight:700;
    background:linear-gradient(90deg,var(--cyan),var(--magenta));
    -webkit-background-clip:text;background-clip:text;color:transparent;
    text-transform:uppercase;
  }
  .brand .tag{
    margin-left:auto;font-size:12px;color:var(--dim);letter-spacing:1px;
  }
  .banner{
    margin:24px 32px 0;padding:18px 22px;border-radius:12px;
    border:1px solid var(--line);
    background:linear-gradient(90deg, rgba(0,240,255,0.06), rgba(255,43,214,0.04));
    display:flex;align-items:center;gap:14px;
  }
  .banner .dot{width:10px;height:10px;border-radius:50%;background:var(--green);box-shadow:0 0 10px var(--green);animation:pulse 1.6s infinite}
  .banner code{background:rgba(0,240,255,0.08);padding:3px 8px;border-radius:6px;border:1px solid var(--line);color:var(--cyan);font-size:13px}
  .banner small{color:var(--dim);margin-left:auto}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}

  main{padding:28px 32px 80px;max-width:1200px;margin:0 auto}
  section{margin-bottom:36px}
  h2{
    font-size:13px;letter-spacing:3px;text-transform:uppercase;color:var(--dim);
    margin-bottom:14px;display:flex;align-items:center;gap:10px;
  }
  h2::before{content:'';width:3px;height:16px;background:linear-gradient(180deg,var(--cyan),var(--magenta));border-radius:2px}

  .meta-grid{
    display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px;
  }
  .card{
    padding:16px 18px;border-radius:12px;
    background:rgba(15,22,48,0.5);
    border:1px solid var(--line);
    transition:border-color 0.2s, transform 0.2s;
  }
  .card:hover{border-color:rgba(0,240,255,0.4);transform:translateY(-1px)}
  .card .k{font-size:11px;letter-spacing:2px;text-transform:uppercase;color:var(--dim);margin-bottom:6px}
  .card .v{color:var(--text);font-size:14px;word-break:break-all}
  .card .v a{color:var(--cyan)}

  .ep{
    background:rgba(15,22,48,0.5);
    border:1px solid var(--line);
    border-radius:12px;margin-bottom:12px;overflow:hidden;
    transition:border-color 0.2s;
  }
  .ep:hover{border-color:rgba(0,240,255,0.35)}
  .ep-head{
    display:flex;align-items:center;gap:14px;padding:14px 18px;cursor:pointer;user-select:none;
  }
  .ep-head:hover{background:rgba(0,240,255,0.03)}
  .method{
    font-size:11px;font-weight:800;letter-spacing:1.5px;padding:4px 10px;border-radius:6px;min-width:72px;text-align:center;
  }
  .method.GET{background:rgba(61,255,165,0.12);color:var(--green);border:1px solid rgba(61,255,165,0.3)}
  .method.POST{background:rgba(255,230,77,0.12);color:var(--yellow);border:1px solid rgba(255,230,77,0.3)}
  .method.WS{background:rgba(255,43,214,0.12);color:var(--magenta);border:1px solid rgba(255,43,214,0.3)}
  .path{font-family:'JetBrains Mono',Consolas,monospace;font-size:14px;color:var(--cyan);flex:1;overflow-wrap:anywhere}
  .summary{color:var(--dim);font-size:13px}
  .chev{color:var(--dim);transition:transform 0.2s;font-size:10px}
  .ep.open .chev{transform:rotate(90deg)}
  .ep-body{padding:0 18px 18px;display:none;animation:fadein 0.2s ease}
  .ep.open .ep-body{display:block}
  @keyframes fadein{from{opacity:0;transform:translateY(-4px)}to{opacity:1;transform:none}}

  .row{display:grid;grid-template-columns:120px 1fr;gap:10px 16px;margin-bottom:10px;align-items:start}
  .row .lbl{font-size:11px;letter-spacing:2px;text-transform:uppercase;color:var(--dim);padding-top:5px}
  pre{
    background:rgba(5,7,15,0.7);
    border:1px solid var(--line);
    border-radius:8px;padding:12px 14px;
    overflow-x:auto;
    font-family:'JetBrains Mono',Consolas,monospace;font-size:12.5px;line-height:1.6;
    color:#cde;
  }
  .btn-row{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}
  button{
    cursor:pointer;font-family:inherit;font-size:12px;letter-spacing:1px;
    padding:8px 14px;border-radius:6px;
    border:1px solid var(--line);
    background:rgba(0,240,255,0.06);color:var(--cyan);
    transition:all 0.15s;
  }
  button:hover{background:rgba(0,240,255,0.14);border-color:rgba(0,240,255,0.5);box-shadow:0 0 12px rgba(0,240,255,0.2)}
  button.primary{background:linear-gradient(135deg,rgba(0,240,255,0.18),rgba(255,43,214,0.14));border-color:rgba(0,240,255,0.45);color:#fff}
  button.ghost{background:transparent;color:var(--dim)}
  .badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:10.5px;letter-spacing:1px;font-weight:700;margin-left:6px}
  .badge.rq{background:rgba(255,71,87,0.15);color:var(--red);border:1px solid rgba(255,71,87,0.3)}
  .badge.op{background:rgba(61,255,165,0.1);color:var(--green);border:1px solid rgba(61,255,165,0.25)}

  .tryit{margin-top:14px;padding-top:14px;border-top:1px dashed var(--line)}
  .param-inp{display:flex;gap:8px;margin:6px 0;flex-wrap:wrap}
  .param-inp input, .param-inp textarea{
    flex:1;min-width:240px;
    background:rgba(5,7,15,0.7);
    border:1px solid var(--line);color:var(--text);
    padding:8px 12px;border-radius:6px;
    font-family:inherit;font-size:13px;outline:none;
    transition:border-color 0.15s;
  }
  .param-inp input:focus, .param-inp textarea:focus{border-color:var(--cyan);box-shadow:0 0 0 2px rgba(0,240,255,0.1)}
  .param-inp .name{width:130px;padding:8px 0;font-family:'JetBrains Mono',Consolas,monospace;font-size:12px;color:var(--yellow)}

  .out{margin-top:12px}
  .out .bar{display:flex;align-items:center;gap:10px;padding:8px 12px;border-top-left-radius:6px;border-top-right-radius:6px;background:rgba(0,0,0,0.4);border:1px solid var(--line);border-bottom:none}
  .out .bar .status{font-weight:700;font-size:12px}
  .out pre{border-top-left-radius:0;border-top-right-radius:0;margin-top:0}
  .status.ok{color:var(--green)}
  .status.err{color:var(--red)}
  .status.pending{color:var(--yellow)}

  .tag-sock{
    display:inline-block;padding:3px 10px;border-radius:6px;
    background:repeating-linear-gradient(45deg, rgba(255,43,214,0.12), rgba(255,43,214,0.12) 6px, rgba(0,240,255,0.08) 6px, rgba(0,240,255,0.08) 12px);
    border:1px solid rgba(255,43,214,0.3);color:var(--magenta);font-size:11px;letter-spacing:1.5px;font-weight:700;
  }

  footer{padding:32px;text-align:center;color:var(--dim);font-size:12px;border-top:1px solid var(--line);margin-top:60px}
  footer span{color:var(--cyan)}
</style>
</head>
<body>
<header>
  <div class="brand">
    <div class="logo"></div>
    <h1>Neuroswap API</h1>
    <div class="tag">LIVE REFERENCE · OFFLINE-FIRST</div>
  </div>
</header>

<div class="banner">
  <div class="dot"></div>
  <div>
    <strong style="color:var(--cyan)">Swagger UI</strong> fallback active · OpenAPI spec served at <code id="oa-path">/openapi.json</code>
  </div>
  <small id="fallback-note">Offline renderer</small>
</div>

<main>
  <section id="sec-meta">
    <h2>Server Metadata</h2>
    <div class="meta-grid" id="meta-cards"></div>
  </section>

  <section>
    <h2>Endpoints</h2>
    <div id="ep-list">
      <div style="color:var(--yellow);padding:16px">Loading OpenAPI spec&hellip;</div>
    </div>
  </section>

  <section>
    <h2>WebSocket Protocol</h2>
    <div class="card">
      <div class="k">Endpoint</div>
      <div class="v" style="margin-bottom:14px"><span class="tag-sock">WS</span>&nbsp;&nbsp;<code style="color:var(--cyan)">/ws</code></div>
      <div class="k" style="margin-top:10px">Client → Server Messages</div>
      <div class="v">
<pre>{
  <span style="color:var(--magenta)">"type"</span>: <span style="color:var(--green)">"frame"</span>,
  <span style="color:var(--magenta)">"image"</span>: <span style="color:var(--green)">"data:image/jpeg;base64,/9j/4AAQ..."</span>
}
{
  <span style="color:var(--magenta)">"type"</span>: <span style="color:var(--green)">"set_intensity"</span>,
  <span style="color:var(--magenta)">"value"</span>: <span style="color:var(--yellow)">0.55</span>  <span style="color:var(--dim)">// 0.0 – 1.0 (clamped)</span>
}</pre>
      </div>
      <div class="k" style="margin-top:14px">Server → Client Messages</div>
      <div class="v">
<pre><span style="color:var(--dim)">// Welcome (first frame on connect):</span>
{<span style="color:var(--magenta)">"type"</span>:<span style="color:var(--green)">"welcome"</span>, <span style="color:var(--magenta)">"intensity"</span>:<span style="color:var(--yellow)">0.85</span>,
 <span style="color:var(--magenta)">"has_target"</span>:<span style="color:var(--yellow)">false</span>, <span style="color:var(--magenta)">"models_loaded"</span>:<span style="color:var(--yellow)">true</span>, <span style="color:var(--magenta)">"targets"</span>:[...]}

<span style="color:var(--dim)">// Processed frame:</span>
{<span style="color:var(--magenta)">"type"</span>:<span style="color:var(--green)">"processed_frame"</span>,
 <span style="color:var(--magenta)">"image"</span>:<span style="color:var(--green)">"data:image/jpeg;base64,..."</span>,
 <span style="color:var(--magenta)">"faces"</span>:[{<span style="color:var(--magenta)">"bbox"</span>:[...],<span style="color:var(--magenta)">"landmarks"</span>:[...],<span style="color:var(--magenta)">"swapped"</span>:true,<span style="color:var(--magenta)">"score"</span>:0.93}],
 <span style="color:var(--magenta)">"latency_ms"</span>:<span style="color:var(--yellow)">24</span>, <span style="color:var(--magenta)">"intensity"</span>:<span style="color:var(--yellow)">0.55</span>, <span style="color:var(--magenta)">"frame_shape"</span>:[480,640]}

<span style="color:var(--dim)">// Error:</span>
{<span style="color:var(--magenta)">"type"</span>:<span style="color:var(--green)">"error"</span>, <span style="color:var(--magenta)">"error"</span>:<span style="color:var(--green)">"no_target"</span>, <span style="color:var(--magenta)">"message"</span>:<span style="color:var(--green)">"..."}</span>}</pre>
      </div>
    </div>
  </section>
</main>

<footer>
  Neuroswap · Face-Swap Engine · Built with <span>FastAPI</span> · <span>InsightFace</span>
</footer>

<script>
const METHOD_ORDER = {GET:0, POST:1, PUT:2, DELETE:3, PATCH:4, HEAD:5, OPTIONS:6};
const SUMMARIES = {
  "GET /": "Serve the web UI (index.html)",
  "GET /target-status": "Session state snapshot (models, target, history)",
  "GET /targets": "List recent uploaded targets",
  "POST /upload-target": "Upload a new target face image",
  "POST /select-target/{target_id}": "Activate a previously-uploaded target by ID",
  "GET /favicon.ico": "Site favicon (PNG)",
  "GET /openapi.json": "The raw OpenAPI 3.1 spec document",
  "GET /docs": "This API reference page",
  "GET /redoc": "Alternative API reference (ReDoc style)",
};
let SPEC = null;

function esc(s){return (s??'').toString().replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
function fmtJSON(v){try{return JSON.stringify(v,null,2)}catch(e){return String(v)}}

async function loadSpec(){
  const path = document.getElementById('oa-path').textContent;
  try{
    const r = await fetch(path, {headers:{'Accept':'application/json'}});
    if(!r.ok) throw new Error('HTTP '+r.status);
    SPEC = await r.json();
    renderMeta();
    renderEndpoints();
  }catch(e){
    document.getElementById('ep-list').innerHTML = '<div style="color:var(--red);padding:16px">Failed to load '+path+': '+esc(e.message)+'</div>';
  }
}

function renderMeta(){
  const info = SPEC.info||{};
  const cards = [
    {k:'Title', v:esc(info.title||'Neuroswap API')},
    {k:'Version', v:esc(info.version||'1.0.0')},
    {k:'OpenAPI', v:esc(SPEC.openapi||'3.1.0')},
    {k:'Server', v:`<a href="/" target="_blank">${window.location.origin}</a>`},
    {k:'Endpoints', v:Object.keys(SPEC.paths||{}).length + ' paths'},
    {k:'OpenAPI JSON', v:`<a href="/openapi.json" target="_blank">/openapi.json</a> (${new Blob([JSON.stringify(SPEC)]).size} bytes)`},
  ];
  document.getElementById('meta-cards').innerHTML = cards.map(c=>
    `<div class="card"><div class="k">${c.k}</div><div class="v">${c.v}</div></div>`
  ).join('');
}

function renderEndpoints(){
  const paths = SPEC.paths||{};
  const list = [];
  for(const path of Object.keys(paths)){
    const p = paths[path];
    for(const method of Object.keys(p)){
      if(['parameters','summary','description','servers','$ref'].includes(method)) continue;
      list.push({method:method.toUpperCase(), path, op:p[method]});
    }
  }
  list.sort((a,b)=>(METHOD_ORDER[a.method]||99)-(METHOD_ORDER[b.method]||99) || a.path.localeCompare(b.path));
  const key = m=>m.method+' '+m.path;

  const html = list.map((m,idx)=>{
    const op = m.op||{};
    const sum = esc(op.summary || SUMMARIES[key(m)] || op.description || '');
    const params = op.parameters||[];
    const reqBody = op.requestBody;
    const responses = op.responses||{};
    const id = 'ep-'+idx;

    let paramRows = '';
    if(params.length){
      paramRows = '<div class="row"><div class="lbl">Parameters</div><div>'+params.map(pa=>{
        const loc = pa.in||'query';
        const req = pa.required ? '<span class="badge rq">REQ</span>' : '<span class="badge op">OPT</span>';
        const schema = pa.schema||{};
        const type = schema.type||'string';
        return `<div style="padding:6px 10px;margin-bottom:4px;border-radius:6px;background:rgba(5,7,15,0.6);border:1px solid var(--line)">
          <code style="color:var(--yellow)">${esc(pa.name)}</code>
          <span class="badge op" style="margin-left:8px">${esc(loc)}</span>
          ${req}
          <small style="color:var(--dim);margin-left:8px">${esc(type)}${schema.enum? ' · enum: '+JSON.stringify(schema.enum):''}</small>
          ${pa.description?`<div style="margin-top:4px;color:var(--dim);font-size:12px">${esc(pa.description)}</div>`:''}
        </div>`;
      }).join('')+'</div></div>';
    }

    let reqBodyHTML = '';
    if(reqBody){
      const content = reqBody.content||{};
      const first = Object.entries(content)[0];
      const mime = first?first[0]:'application/json';
      const schema = first&&first[1]?first[1].schema||{}:{};
      reqBodyHTML = `<div class="row"><div class="lbl">Request Body</div><div>
        ${reqBody.required?'<span class="badge rq">REQ</span> ':''}
        <code style="color:var(--magenta);margin-right:8px">${esc(mime)}</code>
        <pre>${esc(fmtJSON(schema))}</pre>
      </div></div>`;
    }

    let respRows = '';
    if(Object.keys(responses).length){
      respRows = '<div class="row"><div class="lbl">Responses</div><div>'+Object.entries(responses).map(([code,rs])=>{
        const content = rs.content||{};
        const first = Object.entries(content)[0];
        const mime = first?first[0]:'—';
        const schema = first&&first[1]?first[1].schema||{}:null;
        const codeColor = code.startsWith('2')?'var(--green)':code.startsWith('4')||code.startsWith('5')?'var(--red)':'var(--yellow)';
        return `<div style="margin-bottom:8px;padding:10px 12px;border-radius:6px;background:rgba(5,7,15,0.6);border:1px solid var(--line)">
          <strong style="color:${codeColor};font-family:monospace">${esc(code)}</strong>
          <span style="color:var(--text);margin-left:10px">${esc(rs.description||'')}</span>
          ${first?`<span style="color:var(--dim);margin-left:12px;font-size:11px">${esc(mime)}</span>`:''}
          ${schema?`<pre style="margin-top:8px">${esc(fmtJSON(schema))}</pre>`:''}
        </div>`;
      }).join('')+'</div></div>';
    }

    // Try-it-out only for simple GET endpoints (no file uploads)
    const tryIt = (m.method==='GET' && path.indexOf('{')===-1) ? `
      <div class="tryit">
        <div class="row" style="margin-bottom:8px"><div class="lbl">Try it out</div><div>
          ${params.filter(p=>p.in==='query').map(p=>`
            <div class="param-inp">
              <div class="name">${esc(p.name)}</div>
              <input type="text" data-name="${esc(p.name)}" placeholder="${esc((p.schema&&p.schema.type)||'string')}" />
            </div>`).join('')}
          <div class="btn-row">
            <button class="primary" onclick="tryReq('${id}','${m.method}','${path}')">▶ Execute</button>
            <button class="ghost" onclick="document.getElementById('out-${id}').innerHTML=''">Clear output</button>
          </div>
          <div class="out" id="out-${id}"></div>
        </div></div>
      </div>` : (m.method==='POST' && path.indexOf('select-target')!==-1) ? `
      <div class="tryit">
        <div class="row" style="margin-bottom:8px"><div class="lbl">Try it out</div><div>
          <div class="param-inp">
            <div class="name">target_id</div>
            <input type="text" id="ti-${id}-tid" placeholder="uuid or id from /targets" />
          </div>
          <div class="btn-row">
            <button class="primary" onclick="trySelectTarget('${id}','${path}')">▶ Execute</button>
          </div>
          <div class="out" id="out-${id}"></div>
        </div></div>
      </div>` : '';

    return `<div class="ep" id="${id}">
      <div class="ep-head" onclick="document.getElementById('${id}').classList.toggle('open')">
        <div class="method ${m.method}">${esc(m.method)}</div>
        <div class="path">${esc(path)}</div>
        <div class="summary">${sum}</div>
        <div class="chev">▶</div>
      </div>
      <div class="ep-body">
        ${paramRows}
        ${reqBodyHTML}
        ${respRows}
        ${tryIt}
      </div>
    </div>`;
  }).join('');

  document.getElementById('ep-list').innerHTML = html || '<div style="color:var(--dim)">No paths defined.</div>';
}

function buildQS(id, method, path){
  const ep = document.getElementById(id);
  if(!ep) return {url:path, opts:{method}};
  const inputs = ep.querySelectorAll('input[data-name]');
  const qs = [];
  inputs.forEach(inp=>{
    const v = inp.value.trim();
    if(v) qs.push(encodeURIComponent(inp.dataset.name)+'='+encodeURIComponent(v));
  });
  const url = qs.length ? path+'?'+qs.join('&') : path;
  return {url, opts:{method, headers:{'Accept':'application/json'}}};
}

async function tryReq(id, method, path){
  const out = document.getElementById('out-'+id);
  out.innerHTML = `<div class="bar"><span class="status pending">⟳ PENDING…</span></div><pre>Sending ${method} ${esc(path)}</pre>`;
  const {url, opts} = buildQS(id, method, path);
  const t0 = performance.now();
  try{
    const r = await fetch(url, opts);
    const ms = Math.round(performance.now()-t0);
    const txt = await r.text();
    let body;
    try{ body = fmtJSON(JSON.parse(txt)); }catch{ body = txt; }
    const cls = r.ok?'ok':'err';
    out.innerHTML = `<div class="bar"><span class="status ${cls}">● ${r.status} ${esc(r.statusText)}</span><span style="color:var(--dim);margin-left:auto;font-size:11px">${ms} ms · ${(txt.length)} chars</span></div><pre>${esc(body)}</pre>`;
  }catch(e){
    out.innerHTML = `<div class="bar"><span class="status err">✖ ERROR</span></div><pre>${esc(String(e))}</pre>`;
  }
}

async function trySelectTarget(id, pathTpl){
  const tid = document.getElementById('ti-'+id+'-tid').value.trim();
  const out = document.getElementById('out-'+id);
  if(!tid){ out.innerHTML = `<div class="bar"><span class="status err">✖ target_id required</span></div>`; return; }
  const url = pathTpl.replace('{target_id}', encodeURIComponent(tid));
  out.innerHTML = `<div class="bar"><span class="status pending">⟳ PENDING…</span></div><pre>POST ${esc(url)}</pre>`;
  const t0 = performance.now();
  try{
    const r = await fetch(url, {method:'POST', headers:{'Accept':'application/json'}});
    const ms = Math.round(performance.now()-t0);
    const txt = await r.text();
    let body; try{ body = fmtJSON(JSON.parse(txt)); }catch{ body = txt; }
    const cls = r.ok?'ok':'err';
    out.innerHTML = `<div class="bar"><span class="status ${cls}">● ${r.status} ${esc(r.statusText)}</span><span style="color:var(--dim);margin-left:auto;font-size:11px">${ms} ms</span></div><pre>${esc(body)}</pre>`;
  }catch(e){
    out.innerHTML = `<div class="bar"><span class="status err">✖ ERROR</span></div><pre>${esc(String(e))}</pre>`;
  }
}

loadSpec();
</script>
</body>
</html>
"""


_DOCS_ROUTES_INSTALLED = False


def _install_docs_routes():
    global _DOCS_ROUTES_INSTALLED
    if _DOCS_ROUTES_INSTALLED:
        return
    _DOCS_ROUTES_INSTALLED = True

    @app.get("/docs", include_in_schema=False)
    async def custom_docs(request: Request):
        _require_api_key(request)
        global _DOCS_HTML_CACHE
        if _DOCS_HTML_CACHE is None:
            _DOCS_HTML_CACHE = _build_docs_html()
        return HTMLResponse(content=_DOCS_HTML_CACHE)

    @app.get("/redoc", include_in_schema=False)
    async def custom_redoc(request: Request):
        _require_api_key(request)
        global _DOCS_HTML_CACHE
        if _DOCS_HTML_CACHE is None:
            _DOCS_HTML_CACHE = _build_docs_html()
        return HTMLResponse(content=_DOCS_HTML_CACHE)


_install_docs_routes()


MAX_RECENT_TARGETS = 8
MAX_UPLOAD_BYTES = 16 * 1024 * 1024
MAX_STREAM_CHUNK_BYTES = 8 * 1024 * 1024
MAX_WS_IMAGE_BYTES = 2 * 1024 * 1024
ALLOWED_IMAGE_MIMES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}
ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _sha256_hex(data: bytes) -> str:
    try:
        h = hashlib.sha256()
        h.update(data or b"")
        return h.hexdigest()
    except Exception:
        return hashlib.sha256(data or b"").hexdigest()


def _bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return float(inter / denom)

face_analyzer = None
swapper = None
current_target_id = None
recent_targets = OrderedDict()
models_loading = False
models_error = None
models_fallback = False
_state_lock = threading.Lock()
_last_swap_err_log = 0.0

# ── Live RTMP streaming state ────────────────────────────────────────────────
# A single streaming session at a time (globally). Client pushes MediaRecorder
# "timeslice" webm blobs POSTed to /stream-push; server concatenates them into
# ffmpeg stdin for remux into flv -> rtmp URL. This keeps things simple &
# "good enough for a single-user local-deploy desktop app.
import subprocess as _sp
_stream = {
    "proc": None,
    "thread": None,
    "stderr_thread": None,
    "url": None,
    "started_at": None,
    "bytes_in": 0,
    "status": "idle",           # idle | starting | live | error | stopping
    "error": None,
    "stderr_tail": "",
    "ffmpeg_path": None,
}
_stream_lock = threading.RLock()


def _stream_reset(err=None):
    global _stream
    with _stream_lock:
        _stream["status"] = "error" if err else "idle"
        _stream["error"] = err
        if _stream["proc"] is not None:
            try:
                if _stream["proc"].stdin and not _stream["proc"].stdin.closed:
                    try: _stream["proc"].stdin.close()
                    except Exception: pass
            except Exception: pass
            try: _stream["proc"].terminate()
            except Exception: pass
            try: _stream["proc"].wait(timeout=3)
            except Exception:
                try: _stream["proc"].kill()
                except Exception: pass
        _stream["proc"] = None
        _stream["thread"] = None
        _stream["stderr_thread"] = None
        _stream["stderr_tail"] = ""


def _stream_drain_stderr(proc):
    global _stream
    try:
        while True:
            if proc.stderr is None:
                return
            chunk = proc.stderr.read(4096)
            if not chunk:
                return
            try:
                text = chunk.decode("utf-8", "replace")
            except Exception:
                text = ""
            if not text:
                continue
            with _stream_lock:
                if _stream["proc"] is not proc:
                    return
                tail = (_stream.get("stderr_tail") or "") + text
                if len(tail) > 16000:
                    tail = tail[-16000:]
                _stream["stderr_tail"] = tail
    except Exception:
        return


def _stream_watchdog(proc, started_at, url):
    """Background thread: waits for ffmpeg exit so we can detect a clean stop or unexpected death."""
    global _stream
    try:
        rc = proc.wait()
    except Exception as e:
        rc = -1
    with _stream_lock:
        if _stream["proc"] is proc:
            if rc == 0 or rc in (-9, -15, 255):
                if _stream["status"] in ("live", "stopping", "starting"):
                    _stream["status"] = "idle"
            else:
                _stream["status"] = "error"
                _stream["error"] = f"ffmpeg exited rc={rc}"
            _stream["proc"] = None
            _stream["thread"] = None


@app.post("/stream-start")
async def stream_start(request: Request, body: dict):
    global _stream
    _require_api_key(request)
    with _stream_lock:
        if _stream["status"] in ("live", "starting"):
            raise HTTPException(status_code=409, detail="Stream already live; call /stream-stop first")
    ffmpeg = shutil_which("ffmpeg")
    if not ffmpeg:
        raise HTTPException(
            status_code=503,
            detail="ffmpeg binary not found on PATH. Install it (e.g. sudo apt install ffmpeg) then restart the server."
        )
    url = body.get("url") or ""
    if not isinstance(url, str) or not url.strip():
        raise HTTPException(status_code=400, detail="Missing 'url' field.")
    url = url.strip()
    ok_prefixes = ("rtmp:", "rtmps:", "srt:")
    if not url.startswith(ok_prefixes):
        raise HTTPException(
            status_code=400,
            detail="Unsupported stream URL. Use an RTMP(S) ingest URL (e.g. rtmp://a.rtmp.youtube.com/live2/KEY or rtmp://live.twitch.tv/app/KEY)."
        )
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-fflags", "+genpts",
        "-f", "webm", "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-b:v", "4000k", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
        "-f", "flv", url,
    ]
    try:
        proc = _sp.Popen(cmd, stdin=_sp.PIPE, stdout=_sp.DEVNULL, stderr=_sp.PIPE, bufsize=-1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start ffmpeg: {e}")
    t0 = time.time()
    t = threading.Thread(target=_stream_watchdog, args=(proc, t0, url), daemon=True)
    t.start()
    terr = threading.Thread(target=_stream_drain_stderr, args=(proc,), daemon=True)
    terr.start()
    with _stream_lock:
        _stream["proc"] = proc
        _stream["thread"] = t
        _stream["stderr_thread"] = terr
        _stream["url"] = url
        _stream["started_at"] = t0
        _stream["bytes_in"] = 0
        _stream["status"] = "starting"
        _stream["error"] = None
        _stream["stderr_tail"] = ""
        _stream["ffmpeg_path"] = ffmpeg
    # Give ffmpeg 1.2s to fail on startup; if dead, report immediately
    await asyncio.sleep(1.2)
    with _stream_lock:
        if _stream["proc"] is proc:
            rc = proc.poll()
            if rc is None:
                _stream["status"] = "live"
            elif rc != 0:
                err_msg = (_stream.get("stderr_tail") or "").strip()
                _stream_reset(err_msg or f"ffmpeg exited rc={rc}")
                raise HTTPException(status_code=500, detail=(_stream["error"] or "Failed to start stream."))
        else:
            raise HTTPException(status_code=500, detail=(_stream["error"] or "Failed to start stream."))
    return JSONResponse({"status": "live", "url": url, "ffmpeg": ffmpeg})


@app.post("/stream-push")
async def stream_push(request: Request, chunk: UploadFile = File(...)):
    global _stream
    _require_api_key(request)
    data = await chunk.read(MAX_STREAM_CHUNK_BYTES + 1)
    if not data:
        return JSONResponse({"status": "ok", "bytes": 0})
    if len(data) > MAX_STREAM_CHUNK_BYTES:
        raise HTTPException(status_code=413, detail="Stream chunk too large")
    try:
        with _stream_lock:
            proc = _stream["proc"]
            status = _stream["status"]
            if status not in ("starting", "live") or proc is None or proc.stdin is None or proc.stdin.closed:
                raise HTTPException(status_code=409, detail="No active stream. POST /stream-start first.")
            stdin = proc.stdin
        await anyio.to_thread.run_sync(stdin.write, data)
        await anyio.to_thread.run_sync(stdin.flush)
        with _stream_lock:
            _stream["bytes_in"] += len(data)
    except BrokenPipeError:
        _stream_reset("ffmpeg stdin closed (pipe broke)")
        raise HTTPException(status_code=502, detail=_stream["error"])
    except Exception as e:
        _stream_reset(str(e))
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse({"status": "ok", "bytes": len(data)})


@app.post("/stream-stop")
async def stream_stop(request: Request):
    _require_api_key(request)
    with _stream_lock:
        proc = _stream["proc"]
    if proc is None:
        with _stream_lock:
            _stream["status"] = "idle"
            _stream["url"] = None
            _stream["error"] = None
        return JSONResponse({"status": "stopped"})
    with _stream_lock:
        _stream["status"] = "stopping"
    try:
        if proc.stdin and not proc.stdin.closed:
            try: proc.stdin.close()
            except Exception: pass
    except Exception: pass
    await asyncio.sleep(0.5)
    try:
        rc = proc.poll()
        if rc is None:
            try: proc.terminate()
            except Exception: pass
            await asyncio.sleep(0.5)
            rc = proc.poll()
        if rc is None:
            try: proc.kill()
            except Exception: pass
    except Exception:
        pass
    with _stream_lock:
        _stream["proc"] = None
        _stream["thread"] = None
        _stream["status"] = "idle"
        _stream["url"] = None
        _stream["error"] = None
        _stream["started_at"] = None
        _stream["bytes_in"] = 0
    return JSONResponse({"status": "stopped"})


@app.get("/stream-status")
async def stream_status(request: Request):
    _require_api_key(request)
    with _stream_lock:
        snap = {k: v for k, v in _stream.items() if k != "proc" and k != "thread"}
        if _stream["started_at"] is not None:
            snap["elapsed_sec"] = round(time.time() - _stream["started_at"], 1)
        else:
            snap["elapsed_sec"] = 0
        snap["ffmpeg_available"] = bool(shutil_which("ffmpeg"))
        if not snap["ffmpeg_available"]:
            snap["hint"] = "Install ffmpeg (sudo apt install ffmpeg or brew install ffmpeg)"
    return JSONResponse(snap)


def _serialize_target(face, image_bytes, filename, target_id):
    thumb_w, thumb_h = 160, 160
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None and face is not None:
            box = face.bbox.astype(int)
            H, W = img.shape[:2]
            bw = box[2] - box[0]
            bh = box[3] - box[1]
            pad_x = int(bw * 0.30)
            pad_y = int(bh * 0.30)
            x1 = max(0, box[0] - pad_x)
            y1 = max(0, box[1] - pad_y)
            x2 = min(W, box[2] + pad_x)
            y2 = min(H, box[3] + pad_y)
            if x2 > x1 and y2 > y1:
                crop = img[y1:y2, x1:x2]
                crop = cv2.resize(crop, (thumb_w, thumb_h))
                ok, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    thumb_b64 = base64.b64encode(buf.tobytes()).decode('ascii')
                    return {
                        "id": target_id,
                        "name": filename,
                        "thumb": f"data:image/jpeg;base64,{thumb_b64}",
                        "detection_score": float(getattr(face, 'det_score', 0.0))
                    }
    except Exception:
        pass
    return {
        "id": target_id,
        "name": filename,
        "thumb": None,
        "detection_score": float(getattr(face, 'det_score', 0.0))
    }


def _register_target(filename, image_bytes, face_obj, target_id: str | None = None, persisted_name: str | None = None):
    global current_target_id, recent_targets
    if not target_id:
        target_id = uuid.uuid4().hex
    with _state_lock:
        recent_targets[target_id] = {
            "face": face_obj,
            "filename": filename,
            "image_bytes": image_bytes,
            "persisted_name": persisted_name,
            "meta": _serialize_target(face_obj, image_bytes, filename, target_id),
        }
        try:
            recent_targets.move_to_end(target_id)
        except Exception:
            pass
        while len(recent_targets) > MAX_RECENT_TARGETS:
            recent_targets.popitem(last=False)
        current_target_id = target_id
    return target_id


def _inswapper_search_paths():
    """Return the list of filesystem paths that will be searched for inswapper_128.onnx
    (in priority order, matching insightface.model_zoo.get_model + INSIGHTFACE_ROOT env)."""
    paths = []
    try:
        home = _Path(_os.path.expanduser("~"))
        paths.append(str(home / ".insightface" / "models" / "inswapper_128.onnx"))
    except Exception:
        pass
    project_root = BASE_DIR / ".insightface_models" / "models"
    paths.append(str(project_root / "inswapper_128.onnx"))
    env_root = _os.environ.get("INSIGHTFACE_ROOT")
    if env_root:
        paths.append(str(_Path(env_root) / "models" / "inswapper_128.onnx"))
    try:
        import insightface as _if
        for base in getattr(_if.model_zoo, 'BASE_DIR', None) and [
            _Path(getattr(_if.model_zoo, 'BASE_DIR', '/tmp')).expanduser()]:
            pass
        # also check insightface's own cached model dir
        _home = _Path(_os.path.expanduser("~/.insightface/"))
        if _home.is_dir():
            for p in _home.rglob("inswapper_128.onnx"):
                paths.append(str(p))
    except Exception:
        pass
    # dedupe preserving order
    seen = set(); unique = []
    for p in paths:
        if p not in seen: seen.add(p); unique.append(p)
    return unique


def _inswapper_install_status():
    """Returns (present: bool, detail_message: str, search_paths: list[str]).
    Used by /target-status JSON so the UI can show an install guide banner."""
    paths = _inswapper_search_paths()
    found_path = None
    for p in paths:
        try:
            if _os.path.isfile(p) and _os.path.getsize(p) > 10_000_000:
                found_path = p; break
        except Exception:
            pass
    if found_path:
        try:
            sz_mb = round(_os.path.getsize(found_path) / (1024 * 1024), 1)
        except Exception:
            sz_mb = "?"
        return True, f"inswapper_128.onnx found at {found_path} ({sz_mb} MB).", paths
    guide = (
        "REAL DEEPFAKE REQUIRES inswapper_128.onnx (~525 MB). "
        "Place the file at any of these paths, then restart the server: "
        + " ; ".join(paths)
        + " . (The detector-only buffalo_l bundle IS present; only the generative swapper model is missing.)"
    )
    return False, guide, paths


def _restore_persisted_targets_from_disk():
    """After model load, scan UPLOAD_DIR/ for previously-uploaded images, re-run face
    detection, and re-register them into recent_targets so thumbnails survive restarts."""
    if face_analyzer is None:
        return 0
    idx = _load_targets_index()
    idx_changed = False
    if _storage.is_s3_enabled():
        restored = 0
        try:
            lim = int(_os.environ.get("NEUROSWAP_S3_RESTORE_LIMIT", str(MAX_RECENT_TARGETS)))
        except Exception:
            lim = MAX_RECENT_TARGETS
        lim = max(0, min(MAX_RECENT_TARGETS, lim))
        if lim <= 0:
            return 0
        try:
            objs = _storage.list_objects("uploads", lim)
        except Exception as e:
            print(f"  (skip persisted targets from S3: {e})")
            return 0
        for obj in objs:
            try:
                name = _storage.key_to_name("uploads", obj.key) or ""
                if not name:
                    continue
                if name == _TARGETS_INDEX_FILENAME:
                    continue
                ext = _Path(name).suffix.lower()
                if ext not in ALLOWED_IMAGE_EXTS:
                    continue
                if obj.size < 64 or obj.size > MAX_UPLOAD_BYTES:
                    continue
                resp = _storage.get_stream("uploads", name)
                body = resp.get("Body")
                raw = body.read() if body is not None else b""
                try:
                    if body is not None:
                        body.close()
                except Exception:
                    pass
                if len(raw) < 64 or len(raw) > MAX_UPLOAD_BYTES:
                    continue
                face = _extract_face_from_bytes(raw)
                if face is None:
                    continue
                sha = _sha256_hex(raw)
                if not re.fullmatch(r"[0-9a-f]{64}", sha or "", flags=re.IGNORECASE):
                    continue
                if sha in recent_targets:
                    continue
                ent = idx.get(sha) if isinstance(idx, dict) else None
                disp = (ent.get("name") if isinstance(ent, dict) else None) or name
                if isinstance(idx, dict):
                    if sha not in idx:
                        idx[sha] = {"name": disp, "ext": ext, "blob": name}
                        idx_changed = True
                    else:
                        if not isinstance(idx.get(sha), dict):
                            idx[sha] = {"name": disp, "ext": ext, "blob": name}
                            idx_changed = True
                        else:
                            if idx[sha].get("blob") != name:
                                idx[sha]["blob"] = name
                                idx_changed = True
                            if idx[sha].get("ext") != ext:
                                idx[sha]["ext"] = ext
                                idx_changed = True
                _register_target(disp, raw, face, target_id=sha, persisted_name=name)
                restored += 1
                if restored >= MAX_RECENT_TARGETS:
                    break
            except Exception as e:
                print(f"  (skip persisted {obj.key}: {e})")
        if idx_changed:
            try:
                _save_targets_index(idx)
            except Exception:
                pass
        if restored:
            print(f"  ✓ Restored {restored} target face(s) from S3 uploads cache.")
        return restored

    if not UPLOAD_DIR.is_dir():
        return 0
    restored = 0
    try:
        entries = sorted(UPLOAD_DIR.iterdir(),
                         key=lambda p: p.stat().st_mtime if p.is_file() else 0,
                         reverse=True)
    except Exception:
        return 0
    for entry in entries:
        try:
            if not entry.is_file():
                continue
            if entry.name == _TARGETS_INDEX_FILENAME:
                continue
            ext = entry.suffix.lower()
            if ext not in ALLOWED_IMAGE_EXTS:
                continue
            with open(entry, "rb") as f:
                raw = f.read()
            if len(raw) < 64 or len(raw) > MAX_UPLOAD_BYTES:
                continue
            face = _extract_face_from_bytes(raw)
            if face is None:
                continue
            sha = _sha256_hex(raw)
            if not re.fullmatch(r"[0-9a-f]{64}", sha or "", flags=re.IGNORECASE):
                continue
            if sha in recent_targets:
                continue
            persisted_name = f"{sha}{ext}"
            if entry.name != persisted_name:
                dst = UPLOAD_DIR / persisted_name
                if not dst.exists():
                    try:
                        dst.write_bytes(raw)
                    except Exception:
                        pass
                try:
                    entry.unlink(missing_ok=True)
                except Exception:
                    pass
            ent = idx.get(sha) if isinstance(idx, dict) else None
            display_name = (ent.get("name") if isinstance(ent, dict) else None) or entry.name
            if isinstance(idx, dict):
                if sha not in idx or not isinstance(idx.get(sha), dict):
                    idx[sha] = {"name": display_name, "ext": ext, "blob": persisted_name}
                    idx_changed = True
                else:
                    if idx[sha].get("blob") != persisted_name:
                        idx[sha]["blob"] = persisted_name
                        idx_changed = True
                    if idx[sha].get("ext") != ext:
                        idx[sha]["ext"] = ext
                        idx_changed = True
                    if idx[sha].get("name") != display_name:
                        idx[sha]["name"] = display_name
                        idx_changed = True
            _register_target(display_name, raw, face, target_id=sha, persisted_name=persisted_name)
            restored += 1
            if restored >= MAX_RECENT_TARGETS:
                break
        except Exception as e:
            print(f"  (skip persisted {entry.name}: {e})")
    if idx_changed:
        try:
            _save_targets_index(idx)
        except Exception:
            pass
    if restored:
        print(f"  ✓ Restored {restored} target face(s) from uploads/ cache.")
    return restored


def _load_models_background():
    global face_analyzer, swapper, current_target_id, recent_targets, models_loading, models_error, models_fallback
    try:
        print("Loading InsightFace models... (this may take a minute)")

        print("  -> checking for bundled buffalo_l.zip...")
        _install_local_buffalo_zip()

        fa_ok = False
        sw_ok = False
        partial_warn = None
        try:
            face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
            fa_ok = True
            print("  ✓ buffalo_l FaceAnalysis (detector/landmarks) loaded.")
        except Exception as fa_err:
            print(f"  ✗ FaceAnalysis load failed: {fa_err}")
            face_analyzer = None

        if fa_ok:
            for try_download in (False, True):
                try:
                    swapper = insightface.model_zoo.get_model(
                        'inswapper_128.onnx',
                        download=try_download,
                        providers=['CPUExecutionProvider'])
                    sw_ok = True
                    print(f"  ✓ inswapper_128 model loaded (download={try_download}).")
                    break
                except Exception as sw_err:
                    if try_download:
                        print(f"  ⚠ inswapper_128 load failed even with download=True: {sw_err}")
                    else:
                        print(f"  ℹ inswapper_128 not cached, trying download...")
            if not sw_ok:
                swapper = _FallbackSwapper()
                partial_warn = (
                    "Face detection loaded successfully from bundled buffalo_l models, but the "
                    "swap model (inswapper_128.onnx) could not be downloaded due to network/SSL "
                    "restrictions. Running with placeholder swapper overlay — face bounding boxes "
                    "will be shown on the output but actual pixel-level face blending is disabled."
                )
                print("  → using placeholder swapper.")

        if fa_ok and sw_ok:
            models_fallback = False
            models_error = None
            print("Models loaded successfully (real InsightFace + inswapper).")
        elif fa_ok and not sw_ok:
            models_fallback = True
            models_error = partial_warn
            print("Models loaded PARTIALLY (detector OK, swapper → placeholder).")
        else:
            print("  → falling back to fully bundled placeholder models so the UI still works...")
            if not fa_ok:
                face_analyzer = _FallbackAnalyzer()
            if not sw_ok:
                swapper = _FallbackSwapper()
            models_fallback = True
            models_error = (
                "AI models could not be loaded. Running in DEMO mode with placeholder "
                "detection and swapping — uploads and UI work, but faces are simulated."
            )
            print("Fallback models fully activated (demo mode).")

        default_path = BASE_DIR / "Keanu Reeves.jpeg"
        if default_path.exists() and face_analyzer is not None:
            try:
                with open(default_path, "rb") as f:
                    default_bytes = f.read()
                face = _extract_face_from_bytes(default_bytes)
                if face is not None:
                    _register_target("Keanu Reeves.jpeg", default_bytes, face, target_id=_sha256_hex(default_bytes))
                    print("Default target face loaded.")
            except Exception as de:
                print(f"  (default face load skipped: {de})")
        else:
            print("  (no bundled default face image — user will upload one manually.)")

        ins_ok, ins_msg, _ = _inswapper_install_status()
        if not ins_ok and isinstance(swapper, _FallbackSwapper):
            extra = ins_msg
            if models_error:
                models_error = models_error.rstrip() + "  |  " + extra
            else:
                models_error = extra
            print(f"  ℹ {ins_msg}")

        # Finalize: re-hydrate previously-uploaded target thumbs from uploads/ disk cache
        _restore_persisted_targets_from_disk()
    except Exception as e:
        models_error = f"Unexpected error during model init: {e}"
        print(f"Model loading FATAL ERROR: {e}")
        try:
            face_analyzer = _FallbackAnalyzer()
            swapper = _FallbackSwapper()
            models_fallback = True
        except Exception:
            pass
    finally:
        models_loading = False


@app.on_event("startup")
async def startup_event():
    global models_loading, face_analyzer, swapper
    with _state_lock:
        already_loaded = (face_analyzer is not None and swapper is not None)
        if already_loaded:
            models_loading = False
            return
        if models_loading:
            return
        models_loading = True
    t = threading.Thread(target=_load_models_background, daemon=True)
    t.start()


def _extract_face_from_path(image_path: str):
    if face_analyzer is None:
        return None
    img = cv2.imread(image_path)
    if img is None:
        return None
    faces = face_analyzer.get(img)
    return _pick_best_face(faces)


def _extract_face_from_bytes(file_bytes: bytes):
    if face_analyzer is None:
        return None
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    faces = face_analyzer.get(img)
    return _pick_best_face(faces)


def _pick_best_face(faces):
    if not faces:
        return None
    best = None
    best_score = -1.0
    for f in faces:
        try:
            b = getattr(f, "bbox", None)
            if b is None:
                continue
            b = np.array(b).astype(float).tolist()
            if len(b) != 4:
                continue
            area = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
            det = float(getattr(f, "det_score", 0.0))
            score = area * (0.25 + det)
            if score > best_score:
                best_score = score
                best = f
        except Exception:
            continue
    return best or faces[0]


@app.get("/health")
async def health():
    """Liveness/readiness probe for Render, Docker, and load balancers."""
    with _state_lock:
        loaded = face_analyzer is not None and swapper is not None
        loading = models_loading
        fallback = models_fallback
    ffmpeg_ok = bool(shutil_which("ffmpeg"))
    return JSONResponse({
        "status": "ok",
        "models_loaded": loaded,
        "models_loading": loading,
        "models_fallback": fallback,
        "ffmpeg_available": ffmpeg_ok,
    })


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    global INDEX_HTML_CACHE
    if INDEX_HTML_CACHE is None:
        index_path = TEMPLATES_DIR / "index.html"
        if index_path.exists():
            INDEX_HTML_CACHE = index_path.read_text(encoding="utf-8")
        else:
            INDEX_HTML_CACHE = "<h1>index.html missing</h1>"
    return HTMLResponse(content=INDEX_HTML_CACHE)


@app.post("/upload-target")
async def upload_target(request: Request, file: UploadFile = File(...)):
    _require_api_key(request)
    global current_target_id
    if file.size is not None and file.size > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Max size is {MAX_UPLOAD_BYTES//(1024*1024)} MB.")

    filename = file.filename or "image"
    ext = Path(filename).suffix.lower()
    mime = (file.content_type or "").lower()
    if ext not in ALLOWED_IMAGE_EXTS and mime not in ALLOWED_IMAGE_MIMES:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a JPG, PNG, WEBP, BMP, or TIFF image.")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file. Please select a valid image file.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Max size is {MAX_UPLOAD_BYTES//(1024*1024)} MB.")

    if not face_analyzer:
        with _state_lock:
            snap_ml = models_loading
            snap_me = models_error
        if snap_ml:
            user_msg = "AI models are still booting. Please wait ~15 seconds and try again."
        else:
            user_msg = snap_me or "Face analysis engine unavailable. Please try again."
        raise HTTPException(status_code=503, detail=user_msg)

    sha = _sha256_hex(contents)
    if not re.fullmatch(r"[0-9a-f]{64}", sha or "", flags=re.IGNORECASE):
        raise HTTPException(status_code=400, detail="Invalid image hash")

    idx = _load_targets_index()
    with _state_lock:
        exists_in_mem = sha in recent_targets
    if exists_in_mem:
        with _state_lock:
            try:
                recent_targets.move_to_end(sha)
            except Exception:
                pass
            current_target_id = sha
            meta = recent_targets[sha]["meta"]
            targets_snapshot = [t["meta"] for t in recent_targets.values()]
        return JSONResponse({
            "status": "ok",
            "message": "Target already cached",
            "already_exists": True,
            "target_id": sha,
            "target": meta,
            "targets": targets_snapshot
        })

    face = _extract_face_from_bytes(contents)
    if face is None:
        raise HTTPException(status_code=400, detail="No face detected in the uploaded image. Please try a different picture with a clear, well-lit face.")

    ext_save = (ext if ext else (mimetypes.guess_extension(mime) or ".jpg")).lower()
    if ext_save not in ALLOWED_IMAGE_EXTS:
        ext_save = ".jpg"
    persisted_name = f"{sha}{ext_save}"
    save_path = UPLOAD_DIR / persisted_name
    try:
        def _write():
            try:
                with open(save_path, "xb") as f:
                    f.write(contents)
            except FileExistsError:
                return
        await anyio.to_thread.run_sync(_write)
    except Exception as fe:
        print(f"[WARN] Could not persist upload: {fe}")
    if _storage.is_s3_enabled():
        try:
            _storage.put_bytes("uploads", persisted_name, contents, (file.content_type or "").lower() or None)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to persist upload to S3: {e}")

    disp_name = filename
    if isinstance(idx, dict):
        ent = idx.get(sha)
        if isinstance(ent, dict) and ent.get("name"):
            disp_name = str(ent.get("name"))
        idx[sha] = {"name": disp_name, "ext": ext_save, "blob": persisted_name}
        try:
            _save_targets_index(idx)
        except Exception:
            pass

    tid = _register_target(disp_name, contents, face, target_id=sha, persisted_name=persisted_name)
    with _state_lock:
        meta = recent_targets[tid]["meta"]
        targets_snapshot = [t["meta"] for t in recent_targets.values()]

    return JSONResponse({
        "status": "ok",
        "message": "Target face loaded successfully",
        "target_id": tid,
        "target": meta,
        "targets": targets_snapshot
    })


@app.post("/select-target/{target_id}")
async def select_target(request: Request, target_id: str):
    _require_api_key(request)
    global current_target_id
    with _state_lock:
        if target_id not in recent_targets:
            raise HTTPException(status_code=404, detail="Target not found")
        current_target_id = target_id
        meta = recent_targets[target_id]["meta"]
        targets_snapshot = [t["meta"] for t in recent_targets.values()]
    return JSONResponse({
        "status": "ok",
        "target_id": target_id,
        "target": meta,
        "targets": targets_snapshot
    })


@app.get("/targets")
async def list_targets(request: Request):
    _require_api_key(request)
    with _state_lock:
        cid = current_target_id
        targets_snapshot = [t["meta"] for t in recent_targets.values()]
    return JSONResponse({
        "current_target_id": cid,
        "targets": targets_snapshot
    })


def _target_blob_name_from_index(idx: dict, target_id: str) -> str | None:
    if not isinstance(idx, dict):
        return None
    ent = idx.get(target_id)
    if not isinstance(ent, dict):
        return None
    blob = ent.get("blob")
    if not blob or not isinstance(blob, str):
        return None
    return blob.strip() or None


@app.delete("/targets/{target_id}")
async def delete_target(request: Request, target_id: str):
    _require_api_key(request)
    global current_target_id
    idx = _load_targets_index()
    blob = _target_blob_name_from_index(idx, target_id)
    if not blob:
        with _state_lock:
            blob = (recent_targets.get(target_id) or {}).get("persisted_name")
    if blob:
        try:
            (UPLOAD_DIR / blob).unlink(missing_ok=True)
        except Exception:
            pass
        if _storage.is_s3_enabled():
            try:
                _storage.delete_object("uploads", blob)
            except Exception:
                pass
    if isinstance(idx, dict) and target_id in idx:
        try:
            idx.pop(target_id, None)
            _save_targets_index(idx)
        except Exception:
            pass
    with _state_lock:
        recent_targets.pop(target_id, None)
        if current_target_id == target_id:
            try:
                current_target_id = next(reversed(recent_targets)) if recent_targets else None
            except Exception:
                current_target_id = None
        cur = (recent_targets[current_target_id]["meta"] if current_target_id in recent_targets else None)
        targets_snapshot = [t["meta"] for t in recent_targets.values()]
    return JSONResponse({
        "status": "ok",
        "current_target_id": current_target_id,
        "current_target": cur,
        "targets": targets_snapshot,
    })


@app.post("/targets/{target_id}/rename")
async def rename_target(request: Request, target_id: str, body: dict):
    _require_api_key(request)
    name = str((body or {}).get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Missing name")
    if len(name) > 100:
        raise HTTPException(status_code=400, detail="Name too long")
    idx = _load_targets_index()
    if isinstance(idx, dict):
        ent = idx.get(target_id)
        if not isinstance(ent, dict):
            ent = {"name": name}
            idx[target_id] = ent
        ent["name"] = name
        try:
            _save_targets_index(idx)
        except Exception:
            pass
    with _state_lock:
        if target_id not in recent_targets:
            raise HTTPException(status_code=404, detail="Target not found")
        recent_targets[target_id]["filename"] = name
        meta = recent_targets[target_id]["meta"]
        if isinstance(meta, dict):
            meta["name"] = name
        cur = (recent_targets[current_target_id]["meta"] if current_target_id in recent_targets else None)
        targets_snapshot = [t["meta"] for t in recent_targets.values()]
    return JSONResponse({
        "status": "ok",
        "current_target_id": current_target_id,
        "current_target": cur,
        "targets": targets_snapshot,
    })


@app.post("/targets/{target_id}/replace")
async def replace_target(request: Request, target_id: str, file: UploadFile = File(...)):
    _require_api_key(request)
    global current_target_id
    if file.size is not None and file.size > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Max size is {MAX_UPLOAD_BYTES//(1024*1024)} MB.")
    mime = (file.content_type or "").lower()
    if mime and mime not in ALLOWED_IMAGE_MIMES:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    contents = await file.read()
    if len(contents) < 1:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Max size is {MAX_UPLOAD_BYTES//(1024*1024)} MB.")
    if not face_analyzer:
        raise HTTPException(status_code=503, detail="Models not loaded")

    sha = _sha256_hex(contents)
    if not re.fullmatch(r"[0-9a-f]{64}", sha or "", flags=re.IGNORECASE):
        raise HTTPException(status_code=400, detail="Invalid image hash")

    face = _extract_face_from_bytes(contents)
    if face is None:
        raise HTTPException(status_code=400, detail="No face detected in the uploaded image")

    idx = _load_targets_index()
    old_blob = _target_blob_name_from_index(idx, target_id)
    if not old_blob:
        with _state_lock:
            old_blob = (recent_targets.get(target_id) or {}).get("persisted_name")
    if old_blob and sha != target_id:
        try:
            (UPLOAD_DIR / old_blob).unlink(missing_ok=True)
        except Exception:
            pass
        if _storage.is_s3_enabled():
            try:
                _storage.delete_object("uploads", old_blob)
            except Exception:
                pass

    ext = ".jpg"
    if file.filename and "." in file.filename:
        ext = "." + file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_IMAGE_EXTS:
        ext = ".jpg"
    persisted_name = f"{sha}{ext}"
    save_path = UPLOAD_DIR / persisted_name
    try:
        def _write():
            try:
                with open(save_path, "xb") as f:
                    f.write(contents)
            except FileExistsError:
                return
        await anyio.to_thread.run_sync(_write)
    except Exception:
        pass
    if _storage.is_s3_enabled():
        try:
            _storage.put_bytes("uploads", persisted_name, contents, mime or None)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to persist upload to S3: {e}")

    disp = str((idx.get(target_id) or {}).get("name") or (file.filename or "image")).strip() if isinstance(idx, dict) else (file.filename or "image")
    if isinstance(idx, dict):
        if sha != target_id:
            idx.pop(target_id, None)
        idx[sha] = {"name": disp, "ext": ext, "blob": persisted_name}
        try:
            _save_targets_index(idx)
        except Exception:
            pass

    with _state_lock:
        recent_targets.pop(target_id, None)
    new_id = _register_target(disp, contents, face, target_id=sha, persisted_name=persisted_name)
    with _state_lock:
        current_target_id = new_id
        cur = recent_targets[new_id]["meta"]
        targets_snapshot = [t["meta"] for t in recent_targets.values()]
    return JSONResponse({
        "status": "ok",
        "target_id": sha,
        "current_target_id": current_target_id,
        "current_target": cur,
        "targets": targets_snapshot,
    })


@app.get("/target-status")
async def target_status(request: Request):
    _require_api_key(request)
    with _state_lock:
        cid = current_target_id
        current = (recent_targets[cid]["meta"] if cid in recent_targets else None)
        targets_snapshot = [t["meta"] for t in recent_targets.values()]
        ml = models_loading
        me = models_error
        fa = face_analyzer
        sw = swapper
        mf = models_fallback
    ins_present, ins_detail, ins_paths = _inswapper_install_status()
    ffmpeg_bin = shutil_which("ffmpeg")
    return JSONResponse({
        "has_target": cid is not None,
        "current_target_id": cid,
        "current_target": current,
        "targets": targets_snapshot,
        "models_loaded": fa is not None and sw is not None,
        "models_loading": ml,
        "models_error": me,
        "models_fallback": mf,
        "swapper_inswapper_onnx": {
            "present": ins_present,
            "message": ins_detail,
            "search_paths": ins_paths
        },
        "streaming": {
            "ffmpeg_available": bool(ffmpeg_bin),
            "ffmpeg_path": ffmpeg_bin or None,
            "hint": (
                None if ffmpeg_bin else
                "Install ffmpeg (e.g. sudo apt install ffmpeg) to enable live RTMP streaming output."
            )
        }
    })


ALLOWED_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
ALLOWED_VIDEO_MIMES = {
    "video/mp4",
    "video/quicktime",
    "video/x-matroska",
    "video/webm",
    "video/x-msvideo",
}
MAX_VIDEO_UPLOAD_BYTES = int(os.environ.get("MAX_VIDEO_UPLOAD_BYTES", str(1024 * 1024 * 1024)))


def _safe_video_path(video_id: str) -> Path:
    if not isinstance(video_id, str) or not video_id:
        raise HTTPException(status_code=400, detail="Missing video_id")
    name = Path(video_id).name
    if name != video_id:
        raise HTTPException(status_code=400, detail="Invalid video_id")
    ext = Path(name).suffix.lower()
    stem = Path(name).stem
    if ext not in ALLOWED_VIDEO_EXTS:
        raise HTTPException(status_code=400, detail="Invalid video_id")
    if not re.fullmatch(r"[0-9a-f]{32}", stem or "", flags=re.IGNORECASE):
        raise HTTPException(status_code=400, detail="Invalid video_id")
    p = (VIDEO_DIR / name).resolve()
    base = VIDEO_DIR.resolve()
    if base != p.parent:
        raise HTTPException(status_code=400, detail="Invalid video_id")
    return p


async def _save_upload_streaming(upload: UploadFile, dst: Path, max_bytes: int, chunk_size: int = 8 * 1024 * 1024) -> int:
    total = 0
    with open(dst, "wb") as f:
        while True:
            chunk = await upload.read(chunk_size)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(status_code=413, detail="Upload too large")
            await anyio.to_thread.run_sync(f.write, chunk)
    return total

_video_jobs = {}
_video_jobs_lock = threading.Lock()
_video_process_lock = threading.Lock()


def _ffprobe_json(path: str) -> dict:
    ffprobe = shutil_which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe not found on PATH")
    cmd = [
        ffprobe,
        "-v", "error",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        path,
    ]
    p = _sp.run(cmd, stdout=_sp.PIPE, stderr=_sp.PIPE)
    if p.returncode != 0:
        err = (p.stderr or b"").decode("utf-8", "replace").strip()
        raise RuntimeError(err or f"ffprobe failed rc={p.returncode}")
    return json.loads((p.stdout or b"{}").decode("utf-8", "replace") or "{}")


def _parse_fps(rate: str) -> float:
    if not rate or not isinstance(rate, str) or "/" not in rate:
        return 30.0
    a, b = rate.split("/", 1)
    try:
        num = float(a)
        den = float(b)
        if den <= 0:
            return 30.0
        v = num / den
        if v <= 0:
            return 30.0
        return float(v)
    except Exception:
        return 30.0


def _probe_video(path: str) -> dict:
    meta = _ffprobe_json(path)
    streams = meta.get("streams") or []
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not v:
        raise RuntimeError("No video stream found")
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    w = int(v.get("width") or 0)
    h = int(v.get("height") or 0)
    fps = _parse_fps(v.get("avg_frame_rate") or v.get("r_frame_rate") or "")
    nb_frames = v.get("nb_frames")
    try:
        nb_frames = int(nb_frames) if nb_frames is not None else None
    except Exception:
        nb_frames = None
    a_codec = a.get("codec_name") if a else None
    return {
        "width": w,
        "height": h,
        "fps": fps,
        "nb_frames": nb_frames,
        "has_audio": bool(a),
        "audio_codec": a_codec,
        "format": meta.get("format") or {},
    }


def _audio_copy_ok(codec: str | None) -> bool:
    if not codec:
        return False
    return codec in ("aac", "mp3", "ac3", "eac3")


def _cos_sim(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    try:
        aa = float(np.linalg.norm(a))
        bb = float(np.linalg.norm(b))
        if aa <= 0 or bb <= 0:
            return 0.0
        return float(np.dot(a, b) / (aa * bb))
    except Exception:
        return 0.0


def _face_emb(face) -> np.ndarray | None:
    emb = getattr(face, "normed_embedding", None)
    if emb is None:
        emb = getattr(face, "embedding", None)
    if emb is None:
        return None
    try:
        v = np.array(emb, dtype=np.float32).reshape((-1,))
        if v.size < 16:
            return None
        return v
    except Exception:
        return None


class _TrackManager:
    def __init__(self):
        self._next_id = 1
        self._tracks = {}
        self._last_seen = {}

    def update(self, faces):
        now_ts = time.time()
        dets = []
        for f in faces or []:
            try:
                b = f.bbox.astype(int).tolist()
                if len(b) != 4:
                    continue
                dets.append((f, b, _face_emb(f)))
            except Exception:
                continue

        pairs = []
        for di, (f, box, emb) in enumerate(dets):
            for tid, t in self._tracks.items():
                iou = _bbox_iou(box, t["box"])
                sim = _cos_sim(emb, t["emb"]) if emb is not None and t["emb"] is not None else 0.0
                score = (0.65 * iou) + (0.35 * max(0.0, sim))
                if score >= 0.15:
                    pairs.append((score, di, tid))
        pairs.sort(reverse=True, key=lambda x: x[0])

        assigned_det = set()
        assigned_tid = set()
        out = []
        for score, di, tid in pairs:
            if di in assigned_det or tid in assigned_tid:
                continue
            assigned_det.add(di)
            assigned_tid.add(tid)
            f, box, emb = dets[di]
            self._tracks[tid] = {"box": box, "emb": emb, "score": float(getattr(f, "det_score", 0.0))}
            self._last_seen[tid] = now_ts
            out.append((tid, f, box))

        for di, (f, box, emb) in enumerate(dets):
            if di in assigned_det:
                continue
            tid = str(self._next_id)
            self._next_id += 1
            self._tracks[tid] = {"box": box, "emb": emb, "score": float(getattr(f, "det_score", 0.0))}
            self._last_seen[tid] = now_ts
            out.append((tid, f, box))

        stale = [tid for tid, ts in self._last_seen.items() if (now_ts - ts) > 5.0]
        for tid in stale:
            self._tracks.pop(tid, None)
            self._last_seen.pop(tid, None)

        return out, stale


@app.post("/upload-video")
async def upload_video(request: Request, file: UploadFile = File(...)):
    _require_api_key(request)
    if file.size is not None and file.size > MAX_VIDEO_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Video too large")
    filename = file.filename or "video"
    ext = Path(filename).suffix.lower()
    mime = (file.content_type or "").lower()
    if ext not in ALLOWED_VIDEO_EXTS and mime not in ALLOWED_VIDEO_MIMES:
        raise HTTPException(status_code=400, detail="Unsupported video type")
    save_name = f"{uuid.uuid4().hex}{ext if ext else '.mp4'}"
    save_path = VIDEO_DIR / save_name
    try:
        total = await _save_upload_streaming(file, save_path, MAX_VIDEO_UPLOAD_BYTES)
        if total <= 0:
            raise HTTPException(status_code=400, detail="Empty file")
    except HTTPException:
        try:
            save_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    except Exception as e:
        try:
            save_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Failed to save video: {e}")
    try:
        meta = _probe_video(str(save_path))
    except Exception as e:
        try:
            save_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail=f"Invalid video: {e}")
    if _storage.is_s3_enabled():
        try:
            _storage.put_file("videos", save_name, str(save_path), (file.content_type or "").lower() or None)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to persist video to S3: {e}")
    return JSONResponse({"status": "ok", "video_id": save_name, "meta": meta})


@app.post("/analyze-video")
async def analyze_video(request: Request, body: dict):
    _require_api_key(request)
    video_id = (body or {}).get("video_id") or ""
    video_path = _safe_video_path(video_id)
    if not video_path.exists():
        if _storage.is_s3_enabled():
            try:
                _storage.download_file("videos", video_id, str(video_path))
            except Exception:
                raise HTTPException(status_code=404, detail="Video not found")
        else:
            raise HTTPException(status_code=404, detail="Video not found")
    with _state_lock:
        fa_local = face_analyzer
        sw_local = swapper
        targets_by_id = dict(recent_targets)
        default_target_id = current_target_id
    if fa_local is None or sw_local is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    if not default_target_id or default_target_id not in targets_by_id:
        raise HTTPException(status_code=400, detail="No target face selected")

    meta = _probe_video(str(video_path))
    w, h = int(meta["width"]), int(meta["height"])
    if w <= 0 or h <= 0:
        raise HTTPException(status_code=400, detail="Invalid video dimensions")
    fps = float(meta["fps"])

    ffmpeg = shutil_which("ffmpeg")
    if not ffmpeg:
        raise HTTPException(status_code=503, detail="ffmpeg not found on PATH")

    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-map", "0:v:0",
        "-vsync", "0",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-",
    ]
    p = _sp.Popen(cmd, stdout=_sp.PIPE, stderr=_sp.PIPE, bufsize=10**7)
    frame_bytes = w * h * 3
    tracker = _TrackManager()
    samples = {}
    read_frames = 0
    try:
        while read_frames < 60:
            raw = p.stdout.read(frame_bytes) if p.stdout is not None else b""
            if len(raw) != frame_bytes:
                break
            frame = np.frombuffer(raw, np.uint8).reshape((h, w, 3))
            assigned, _ = tracker.update(fa_local.get(frame))
            for tid, face, box in assigned:
                if tid in samples:
                    continue
                try:
                    x1, y1, x2, y2 = box
                    x1c, y1c = max(0, x1), max(0, y1)
                    x2c, y2c = min(w, x2), min(h, y2)
                    if x2c <= x1c or y2c <= y1c:
                        continue
                    crop = frame[y1c:y2c, x1c:x2c]
                    crop = cv2.resize(crop, (160, 160))
                    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                        samples[tid] = {
                            "track_id": tid,
                            "thumb": f"data:image/jpeg;base64,{b64}",
                            "score": float(getattr(face, "det_score", 0.0)),
                        }
                except Exception:
                    pass
            read_frames += 1
    finally:
        try:
            if p.stdout:
                p.stdout.close()
        except Exception:
            pass
        try:
            if p.stderr:
                p.stderr.close()
        except Exception:
            pass
        try:
            p.kill()
        except Exception:
            pass
    tracks = list(samples.values())
    tracks.sort(key=lambda x: int(x.get("track_id") or 0))
    return JSONResponse({"status": "ok", "video_id": video_id, "fps": fps, "tracks": tracks})


def _process_video_job(job_id: str, video_id: str, default_target_id: str, face_map: dict, intensity: float):
    with _video_jobs_lock:
        _video_jobs[job_id] = {
            "status": "running",
            "progress": 0.0,
            "error": None,
            "video_id": video_id,
            "output_id": None,
            "s3_output_key": None,
        }
    video_path = _safe_video_path(video_id)
    if not video_path.exists() and _storage.is_s3_enabled():
        try:
            _storage.download_file("videos", video_id, str(video_path))
        except Exception as e:
            with _video_jobs_lock:
                if job_id in _video_jobs:
                    _video_jobs[job_id]["status"] = "error"
                    _video_jobs[job_id]["error"] = f"Failed to fetch video from S3: {e}"
            return
    out_name = f"{uuid.uuid4().hex}.mp4"
    out_path = OUTPUT_DIR / out_name
    try:
        with _video_process_lock:
            meta = _probe_video(str(video_path))
            w, h = int(meta["width"]), int(meta["height"])
            fps = float(meta["fps"])
            nb_frames = meta.get("nb_frames")
            ffmpeg = shutil_which("ffmpeg")
            if not ffmpeg:
                raise RuntimeError("ffmpeg not found on PATH")

            with _state_lock:
                fa_local = face_analyzer
                sw_local = swapper
                targets_by_id = dict(recent_targets)
            if fa_local is None or sw_local is None:
                raise RuntimeError("Models not loaded")
            if default_target_id not in targets_by_id:
                raise RuntimeError("Target not found")

            reader_cmd = [
                ffmpeg, "-hide_banner", "-loglevel", "error",
                "-i", str(video_path),
                "-map", "0:v:0",
                "-vsync", "0",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-",
            ]
            audio_args = []
            if meta.get("has_audio"):
                if _audio_copy_ok(meta.get("audio_codec")):
                    audio_args = ["-map", "1:a:0?", "-c:a", "copy"]
                else:
                    audio_args = ["-map", "1:a:0?", "-c:a", "aac", "-b:a", "192k"]
            else:
                audio_args = ["-an"]
            writer_cmd = [
                ffmpeg, "-hide_banner", "-loglevel", "error",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{w}x{h}",
                "-r", f"{fps:.6f}",
                "-i", "-",
                "-i", str(video_path),
                "-map", "0:v:0",
                *audio_args,
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(out_path),
            ]

            reader = _sp.Popen(reader_cmd, stdout=_sp.PIPE, stderr=_sp.DEVNULL, bufsize=10**7)
            writer = _sp.Popen(writer_cmd, stdin=_sp.PIPE, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL, bufsize=10**7)
            frame_bytes = w * h * 3
            tracker = _TrackManager()
            frame_idx = 0
            try:
                while True:
                    raw = reader.stdout.read(frame_bytes) if reader.stdout is not None else b""
                    if len(raw) != frame_bytes:
                        break
                    frame = np.frombuffer(raw, np.uint8).reshape((h, w, 3))
                    assigned, _stale = tracker.update(fa_local.get(frame))
                    for tid, face, box in assigned:
                        target_id = face_map.get(str(tid)) or default_target_id
                        if target_id not in targets_by_id:
                            target_id = default_target_id
                        target_face = targets_by_id[target_id]["face"]
                        try:
                            swapped = sw_local.get(frame, face, target_face, paste_back=True)
                            if intensity >= 0.999:
                                frame = swapped
                            elif intensity <= 0.001:
                                pass
                            else:
                                frame = cv2.addWeighted(swapped, float(intensity), frame, 1.0 - float(intensity), 0)
                        except Exception:
                            pass
                    try:
                        if writer.stdin is None:
                            raise RuntimeError("ffmpeg writer stdin missing")
                        writer.stdin.write(frame.tobytes())
                    except Exception as e:
                        raise RuntimeError(f"ffmpeg writer failed: {e}")
                    frame_idx += 1
                    if nb_frames:
                        prog = min(99.0, (frame_idx / max(1, nb_frames)) * 100.0)
                    else:
                        prog = min(99.0, frame_idx % 100)
                    with _video_jobs_lock:
                        if job_id in _video_jobs:
                            _video_jobs[job_id]["progress"] = float(prog)
            finally:
                try:
                    if reader.stdout:
                        reader.stdout.close()
                except Exception:
                    pass
                try:
                    if writer.stdin:
                        writer.stdin.close()
                except Exception:
                    pass
                try:
                    reader.wait(timeout=10)
                except Exception:
                    try:
                        reader.kill()
                    except Exception:
                        pass
                try:
                    writer.wait(timeout=30)
                except Exception:
                    try:
                        writer.kill()
                    except Exception:
                        pass

            if writer.returncode not in (0, None):
                raise RuntimeError(f"ffmpeg failed rc={writer.returncode}")

        s3_key = None
        if _storage.is_s3_enabled():
            try:
                s3_key = _storage.put_file("outputs", out_name, str(out_path), "video/mp4")
            except Exception as e:
                raise RuntimeError(f"Failed to upload output to S3: {e}")

        with _video_jobs_lock:
            if job_id in _video_jobs:
                _video_jobs[job_id]["status"] = "done"
                _video_jobs[job_id]["progress"] = 100.0
                _video_jobs[job_id]["output_id"] = out_name
                _video_jobs[job_id]["s3_output_key"] = s3_key
    except Exception as e:
        try:
            out_path.unlink(missing_ok=True)
        except Exception:
            pass
        with _video_jobs_lock:
            if job_id in _video_jobs:
                _video_jobs[job_id]["status"] = "error"
                _video_jobs[job_id]["error"] = str(e)


@app.post("/process-video")
async def process_video(request: Request, body: dict):
    _require_api_key(request)
    video_id = (body or {}).get("video_id") or ""
    video_path = _safe_video_path(video_id)
    if not video_path.exists():
        if _storage.is_s3_enabled():
            try:
                _storage.download_file("videos", video_id, str(video_path))
            except Exception:
                raise HTTPException(status_code=404, detail="Video not found")
        else:
            raise HTTPException(status_code=404, detail="Video not found")
    with _state_lock:
        default_target_id = current_target_id
    if not default_target_id:
        raise HTTPException(status_code=400, detail="No target face selected")
    intensity = body.get("intensity", 1.0)
    try:
        intensity = float(intensity)
    except Exception:
        intensity = 1.0
    intensity = max(0.0, min(1.0, intensity))
    raw_map = body.get("face_map") or {}
    cleaned = {}
    if isinstance(raw_map, dict):
        for k, v in raw_map.items():
            if isinstance(k, (str, int)) and isinstance(v, str):
                cleaned[str(k)] = v
    job_id = uuid.uuid4().hex
    t = threading.Thread(
        target=_process_video_job,
        args=(job_id, video_id, default_target_id, cleaned, intensity),
        daemon=True,
    )
    t.start()
    return JSONResponse({"status": "ok", "job_id": job_id})


@app.get("/video-status/{job_id}")
async def video_status(request: Request, job_id: str):
    _require_api_key(request)
    with _video_jobs_lock:
        j = dict(_video_jobs.get(job_id) or {})
    if not j:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(j)


@app.get("/download-video/{job_id}")
async def download_video(request: Request, job_id: str):
    _require_api_key(request)
    with _video_jobs_lock:
        j = dict(_video_jobs.get(job_id) or {})
    if not j:
        raise HTTPException(status_code=404, detail="Job not found")
    if j.get("status") != "done" or not j.get("output_id"):
        raise HTTPException(status_code=409, detail="Video not ready")
    out_path = OUTPUT_DIR / j["output_id"]
    if out_path.exists():
        return FileResponse(str(out_path), media_type="video/mp4", filename=f"neuroswap-{job_id}.mp4")
    if not _storage.is_s3_enabled():
        raise HTTPException(status_code=404, detail="Output missing")
    try:
        resp = _storage.get_stream("outputs", j["output_id"])
    except Exception:
        raise HTTPException(status_code=404, detail="Output missing")
    body = resp.get("Body")
    if body is None:
        raise HTTPException(status_code=404, detail="Output missing")

    def _iter():
        try:
            while True:
                chunk = body.read(1024 * 1024)
                if not chunk:
                    break
                yield chunk
        finally:
            try:
                body.close()
            except Exception:
                pass

    headers = {"Content-Disposition": f'attachment; filename="neuroswap-{job_id}.mp4"'}
    try:
        clen = resp.get("ContentLength")
        if clen is not None:
            headers["Content-Length"] = str(int(clen))
    except Exception:
        pass
    return StreamingResponse(_iter(), media_type="video/mp4", headers=headers)


def _cleanup_paths(*paths):
    for p in paths:
        try:
            _Path(p).unlink(missing_ok=True)
        except Exception:
            pass


@app.post("/convert-recording")
async def convert_recording(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Convert browser MediaRecorder output (WebM/MKV) to H.264 MP4 via ffmpeg."""
    _require_api_key(request)
    ffmpeg = shutil_which("ffmpeg")
    if not ffmpeg:
        raise HTTPException(status_code=503, detail="ffmpeg not found on PATH")

    if file.size is not None and file.size > MAX_VIDEO_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Recording too large")

    ext = ".webm"
    if file.filename and "." in file.filename:
        ext = "." + file.filename.rsplit(".", 1)[-1].lower()
    if ext not in {".webm", ".mkv", ".mp4", ".mov", ".avi"}:
        ext = ".webm"

    tmp_in = OUTPUT_DIR / f"{uuid.uuid4().hex}{ext}"
    tmp_out = OUTPUT_DIR / f"{uuid.uuid4().hex}.mp4"
    try:
        total = await _save_upload_streaming(file, tmp_in, MAX_VIDEO_UPLOAD_BYTES)
        if total <= 0:
            raise HTTPException(status_code=400, detail="Empty upload")
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-i", str(tmp_in),
            "-map", "0:v:0?",
            "-map", "0:a:0?",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            "-y", str(tmp_out),
        ]
        p = _sp.run(cmd, stdout=_sp.PIPE, stderr=_sp.PIPE)
        if p.returncode != 0 or not tmp_out.exists() or tmp_out.stat().st_size < 256:
            err = (p.stderr or b"").decode("utf-8", "replace").strip()
            raise HTTPException(
                status_code=500,
                detail=err or f"ffmpeg conversion failed (rc={p.returncode})",
            )
    except HTTPException:
        _cleanup_paths(tmp_in, tmp_out)
        raise
    except Exception as e:
        _cleanup_paths(tmp_in, tmp_out)
        raise HTTPException(status_code=500, detail=f"Conversion failed: {e}")

    background_tasks.add_task(_cleanup_paths, str(tmp_in), str(tmp_out))
    return FileResponse(
        str(tmp_out),
        media_type="video/mp4",
        filename="neuroswap-recording.mp4",
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global _last_swap_err_log
    try:
        _require_api_key(websocket)
    except HTTPException:
        try:
            await websocket.close(code=1008)
        except Exception:
            pass
        return
    await websocket.accept()
    print("Client connected")

    local_intensity = 1.0
    face_map = {}
    face_map_lock = asyncio.Lock()
    tracker = _TrackManager()
    frame_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1)
    receiver_done = asyncio.Event()

    with _state_lock:
        cid_welcome = current_target_id
        targets_snapshot = [t["meta"] for t in recent_targets.values()]
        fa = face_analyzer
        sw = swapper
        ml = models_loading
        me = models_error
        mf = models_fallback
    try:
        await websocket.send_json({
            "type": "welcome",
            "intensity": local_intensity,
            "has_target": cid_welcome is not None and cid_welcome in recent_targets,
            "current_target_id": cid_welcome,
            "targets": targets_snapshot,
            "models_loaded": fa is not None and sw is not None,
            "models_loading": ml,
            "models_error": me,
            "models_fallback": mf
        })
    except Exception:
        pass

    try:
        def _drop_oldest_and_put(payload: bytes):
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except Exception:
                    pass
            try:
                frame_queue.put_nowait(payload)
            except Exception:
                pass

        async def _receiver():
            nonlocal local_intensity
            try:
                while True:
                    data = await websocket.receive_text()
                    try:
                        msg = json.loads(data)
                    except (ValueError, json.JSONDecodeError):
                        continue
                    if not isinstance(msg, dict):
                        continue
                    msg_type = msg.get("type")
                    if msg_type == "set_intensity":
                        try:
                            v = float(msg.get("value", 1.0))
                            local_intensity = max(0.0, min(1.0, v))
                        except (TypeError, ValueError):
                            pass
                        continue
                    if msg_type == "set_face_map":
                        raw_map = msg.get("map")
                        if isinstance(raw_map, dict):
                            cleaned = {}
                            for k, v in raw_map.items():
                                if isinstance(k, (str, int)) and isinstance(v, str):
                                    cleaned[str(k)] = v
                            async with face_map_lock:
                                face_map.clear()
                                face_map.update(cleaned)
                        continue
                    if msg_type != "frame":
                        continue
                    image_field = msg.get("image", "")
                    if not isinstance(image_field, str) or not image_field:
                        continue
                    b64_payload = image_field.split(",", 1)[1] if "," in image_field else image_field
                    if len(b64_payload) > (MAX_WS_IMAGE_BYTES * 4 // 3) + 4096:
                        continue
                    try:
                        img_data = base64.b64decode(b64_payload, validate=True)
                    except (IndexError, base64.binascii.Error, ValueError, TypeError):
                        continue
                    if not img_data or len(img_data) > MAX_WS_IMAGE_BYTES:
                        continue
                    _drop_oldest_and_put(img_data)
            except Exception:
                pass
            finally:
                receiver_done.set()

        recv_task = asyncio.create_task(_receiver())

        while not receiver_done.is_set():
            try:
                img_data = await asyncio.wait_for(frame_queue.get(), timeout=0.6)
            except asyncio.TimeoutError:
                continue

            with _state_lock:
                fa_local = face_analyzer
                sw_local = swapper
                snapshot_cid = current_target_id
                targets_by_id = dict(recent_targets)

            if fa_local is None or sw_local is None:
                try:
                    await websocket.send_json({
                        "type": "error",
                        "error": "models_loading",
                        "message": "Models still loading. Please wait..."
                    })
                except Exception:
                    pass
                continue

            if not snapshot_cid or snapshot_cid not in targets_by_id:
                try:
                    await websocket.send_json({
                        "type": "error",
                        "error": "no_target",
                        "message": "No target face set. Upload an image first."
                    })
                except Exception:
                    pass
                continue

            default_target_face = targets_by_id[snapshot_cid]["face"]
            t0 = time.perf_counter()

            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            source_faces = fa_local.get(frame)
            assigned, stale = tracker.update(source_faces)
            if stale:
                async with face_map_lock:
                    for tid in stale:
                        face_map.pop(str(tid), None)

            async with face_map_lock:
                face_map_snapshot = dict(face_map)

            faces_info = []
            swap_success_count = 0

            for tid, face, box in assigned:
                tid = str(tid)
                target_id = face_map_snapshot.get(tid) or snapshot_cid
                if target_id not in targets_by_id:
                    target_id = snapshot_cid
                target_face = targets_by_id[target_id]["face"] if target_id in targets_by_id else default_target_face

                det_score_this = float(getattr(face, "det_score", 0.0))
                landmarks = None
                if hasattr(face, "kps") and face.kps is not None:
                    try:
                        landmarks = face.kps.astype(float).tolist()
                    except Exception:
                        landmarks = None

                f_info = {
                    "track_id": tid,
                    "target_id": target_id,
                    "bbox": box,
                    "score": det_score_this,
                    "landmarks": landmarks
                }

                try:
                    if local_intensity >= 0.999:
                        frame = sw_local.get(frame, face, target_face, paste_back=True)
                    elif local_intensity > 0.001:
                        swapped = sw_local.get(frame, face, target_face, paste_back=True)
                        alpha = float(local_intensity)
                        frame = cv2.addWeighted(swapped, alpha, frame, 1.0 - alpha, 0)
                    f_info["swapped"] = True
                    swap_success_count += 1
                except Exception as swap_err:
                    f_info["swapped"] = False
                    now = time.time()
                    if now - _last_swap_err_log > 60.0:
                        print(f"[swap WARN] Skipped face: {swap_err}")
                        _last_swap_err_log = now

                faces_info.append(f_info)

            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                continue
            jpg_as_text = base64.b64encode(buffer).decode("utf-8")

            swap_quality = 0.0
            if faces_info:
                scores = [f["score"] for f in faces_info]
                avg_score = sum(scores) / max(1, len(scores))
                swap_rate = swap_success_count / max(1, len(faces_info))
                swap_quality = round(avg_score * swap_rate * 100, 1)

            try:
                await websocket.send_json({
                    "type": "processed_frame",
                    "image": f"data:image/jpeg;base64,{jpg_as_text}",
                    "faces": faces_info,
                    "latency_ms": latency_ms,
                    "swap_quality_pct": swap_quality,
                    "intensity": local_intensity,
                    "frame_shape": list(frame.shape[:2])
                })
            except Exception:
                break

        try:
            recv_task.cancel()
        except Exception:
            pass
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        print("Client disconnected")
