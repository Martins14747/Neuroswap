import os as _os
from pathlib import Path as _Path

BASE_DIR = _Path(__file__).parent

_default_root = _os.environ.get("INSIGHTFACE_ROOT")
if not _default_root:
    _candidates = [
        str(BASE_DIR / ".insightface_models"),
        str(_Path(_os.path.expanduser("~/.insightface"))),
    ]
    for _root in _candidates:
        _buffalo_dir = _Path(_root) / "models" / "buffalo_l"
        if _buffalo_dir.exists() and any(p.suffix.lower() == ".onnx" for p in _buffalo_dir.iterdir()):
            _os.environ["INSIGHTFACE_ROOT"] = _root
            break

from fastapi import FastAPI, WebSocket, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import base64
import json
import os
import uuid
import time
import threading
import mimetypes
import zipfile
import shutil
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

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR = BASE_DIR / "templates"
INDEX_HTML_CACHE = None

app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/static-root", StaticFiles(directory=str(BASE_DIR)), name="static-root")

MAX_RECENT_TARGETS = 8
MAX_UPLOAD_BYTES = 16 * 1024 * 1024
ALLOWED_IMAGE_MIMES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}
ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

face_analyzer = None
swapper = None
current_target_id = None
recent_targets = OrderedDict()
models_loading = False
models_error = None
models_fallback = False
_state_lock = threading.Lock()
_last_swap_err_log = 0.0


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


def _register_target(filename, image_bytes, face_obj):
    global current_target_id, recent_targets
    target_id = uuid.uuid4().hex
    with _state_lock:
        recent_targets[target_id] = {
            "face": face_obj,
            "filename": filename,
            "image_bytes": image_bytes,
            "meta": _serialize_target(face_obj, image_bytes, filename, target_id)
        }
        while len(recent_targets) > MAX_RECENT_TARGETS:
            recent_targets.popitem(last=False)
        current_target_id = target_id
    return target_id


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
                    _register_target("Keanu Reeves.jpeg", default_bytes, face)
                    print("Default target face (Keanu Reeves) loaded.")
            except Exception as de:
                print(f"  (default face load skipped: {de})")
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
    return faces[0] if faces else None


def _extract_face_from_bytes(file_bytes: bytes):
    if face_analyzer is None:
        return None
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    faces = face_analyzer.get(img)
    return faces[0] if faces else None


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
async def upload_target(file: UploadFile = File(...)):
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

    face = _extract_face_from_bytes(contents)
    if face is None:
        raise HTTPException(status_code=400, detail="No face detected in the uploaded image. Please try a different picture with a clear, well-lit face.")

    ext_save = ext if ext else (mimetypes.guess_extension(mime) or ".jpg")
    save_name = f"{uuid.uuid4().hex}{ext_save}"
    save_path = UPLOAD_DIR / save_name
    try:
        with open(save_path, "wb") as f:
            f.write(contents)
    except Exception as fe:
        print(f"[WARN] Could not persist upload: {fe}")

    tid = _register_target(filename, contents, face)
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
async def select_target(target_id: str):
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
async def list_targets():
    with _state_lock:
        cid = current_target_id
        targets_snapshot = [t["meta"] for t in recent_targets.values()]
    return JSONResponse({
        "current_target_id": cid,
        "targets": targets_snapshot
    })


@app.get("/target-status")
async def target_status():
    with _state_lock:
        cid = current_target_id
        current = (recent_targets[cid]["meta"] if cid in recent_targets else None)
        targets_snapshot = [t["meta"] for t in recent_targets.values()]
        ml = models_loading
        me = models_error
        fa = face_analyzer
        sw = swapper
        mf = models_fallback
    return JSONResponse({
        "has_target": cid is not None,
        "current_target_id": cid,
        "current_target": current,
        "targets": targets_snapshot,
        "models_loaded": fa is not None and sw is not None,
        "models_loading": ml,
        "models_error": me,
        "models_fallback": mf
    })


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global _last_swap_err_log
    await websocket.accept()
    print("Client connected")

    local_intensity = 1.0

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
        while True:
            try:
                data = await websocket.receive_text()
            except Exception:
                break

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

            if msg_type != "frame":
                continue

            if face_analyzer is None or swapper is None:
                try:
                    await websocket.send_json({
                        "type": "error",
                        "error": "models_loading",
                        "message": "Models still loading. Please wait..."
                    })
                except Exception:
                    pass
                continue

            with _state_lock:
                snapshot_cid = current_target_id
                if snapshot_cid and snapshot_cid in recent_targets:
                    target_face_snapshot = recent_targets[snapshot_cid]["face"]
                else:
                    target_face_snapshot = None
                    snapshot_cid = None

            if target_face_snapshot is None:
                try:
                    await websocket.send_json({
                        "type": "error",
                        "error": "no_target",
                        "message": "No target face set. Upload an image first."
                    })
                except Exception:
                    pass
                continue

            t0 = time.perf_counter()

            try:
                image_field = msg.get("image", "")
                if "," in image_field:
                    b64_payload = image_field.split(",", 1)[1]
                else:
                    b64_payload = image_field
                img_data = base64.b64decode(b64_payload, validate=False)
            except (IndexError, base64.binascii.Error, ValueError, TypeError):
                continue

            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            target_face = target_face_snapshot
            source_faces = face_analyzer.get(frame)

            faces_info = []
            swap_success_count = 0

            if source_faces:
                for face in source_faces:
                    box = face.bbox.astype(int).tolist()
                    det_score_this = float(getattr(face, 'det_score', 0.0))
                    landmarks = None
                    if hasattr(face, 'kps') and face.kps is not None:
                        try:
                            landmarks = face.kps.astype(float).tolist()
                        except Exception:
                            landmarks = None
                    f_info = {
                        "bbox": box,
                        "score": det_score_this,
                        "landmarks": landmarks
                    }
                    try:
                        if local_intensity >= 0.999:
                            frame = swapper.get(frame, face, target_face, paste_back=True)
                        elif local_intensity > 0.001:
                            swapped = swapper.get(frame.copy(), face, target_face, paste_back=True)
                            alpha = local_intensity
                            x1, y1, x2, y2 = box
                            H, W = frame.shape[:2]
                            x1c, y1c = max(0, x1), max(0, y1)
                            x2c, y2c = min(W, x2), min(H, y2)
                            if x2c > x1c and y2c > y1c:
                                src_roi = frame[y1c:y2c, x1c:x2c]
                                dst_roi = swapped[y1c:y2c, x1c:x2c]
                                blended = cv2.addWeighted(dst_roi, alpha, src_roi, 1 - alpha, 0)
                                frame[y1c:y2c, x1c:x2c] = blended
                            else:
                                frame = cv2.addWeighted(swapped, alpha, frame, 1 - alpha, 0)
                        f_info["swapped"] = True
                        swap_success_count += 1
                    except Exception as swap_err:
                        f_info["swapped"] = False
                        now = time.time()
                        global _last_swap_err_log
                        if now - _last_swap_err_log > 60.0:
                            print(f"[swap WARN] Skipped face: {swap_err}")
                            _last_swap_err_log = now
                    faces_info.append(f_info)

            latency_ms = round((time.perf_counter() - t0) * 1000, 2)

            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')

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
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        print("Client disconnected")
