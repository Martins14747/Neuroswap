import sys, os, time, threading, socket, json, base64, io, random, string
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Merge venv site-packages (for cv2/insightface) with system site-packages (for websocket-client)
import site
venv_sp = '/home/me/Documents/SOAIR/face-swap-web/venv/lib/python3.13/site-packages'
if venv_sp not in sys.path:
    sys.path.insert(0, venv_sp)
for p in site.getsitepackages() + [site.getusersitepackages(), '/usr/lib/python3/dist-packages']:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.append(p)

import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis

# ---------- patch InsightFace to offline fakes BEFORE importing main ----------
class FakeFace:
    def __init__(self, idx=0):
        self.bbox = np.array([50+idx*30, 50+idx*20, 200+idx*30, 220+idx*20], dtype=np.float32)
        self.det_score = 0.93
        self.kps = np.array([
            [90., 100.], [160., 100.],
            [125., 140.],
            [100., 180.], [150., 180.]
        ], dtype=np.float32)

class FakeAnalyzer:
    def prepare(self, *a, **kw): pass
    def get(self, img):
        return [FakeFace(0), FakeFace(1)] if img is not None and img.shape[0] > 100 else []

class FakeSwapper:
    def get(self, frame, face, target_face, paste_back=True):
        faked = frame.copy()
        b = face.bbox.astype(int)
        cv2.rectangle(faked, (b[0],b[1]), (b[2],b[3]), (128,0,255), 4)
        return faked

insightface.app.FaceAnalysis = lambda *a, **kw: FakeAnalyzer()
insightface.model_zoo.get_model = lambda *a, **kw: FakeSwapper()

import main
main.face_analyzer = FakeAnalyzer()
main.swapper = FakeSwapper()
main.models_loading = False
main.models_error = None
main._load_models_background = lambda: None

import uvicorn
import requests
import websocket  # pip package websocket-client

# ---------- find free port ----------
def free_port():
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    return port
PORT = free_port()
BASE = f'http://127.0.0.1:{PORT}'
WS_BASE = f'ws://127.0.0.1:{PORT}/ws'

# ---------- start uvicorn in background ----------
server_config = uvicorn.Config(main.app, host='127.0.0.1', port=PORT, log_level='error', access_log=False)
server = uvicorn.Server(server_config)
t = threading.Thread(target=server.run, daemon=True)
t.start()
# wait for server to come up
for i in range(80):
    try:
        r = requests.get(f'{BASE}/target-status', timeout=1)
        break
    except Exception:
        time.sleep(0.08)
else:
    print('server failed to come up'); sys.exit(1)

# ---------- helpers ----------
def make_jpg_bytes(w=400,h=300,color=(80,120,180)):
    img = np.zeros((h,w,3), dtype=np.uint8)
    img[:] = color
    cv2.circle(img, (w//2,h//2), 60, (255,255,255), -1)
    cv2.rectangle(img, (60,60), (240,240), (0,255,0), 3)
    ok, buf = cv2.imencode('.jpg', img)
    return buf.tobytes()

PASS=0; FAIL=0
def t(name, cond):
    global PASS, FAIL
    if cond: PASS+=1; print(f'  [OK]   {name}')
    else:    FAIL+=1; print(f'  [FAIL] {name}')
    return bool(cond)

# ================== TESTS ==================

print('\n=== 1. GET / (home) ===')
r = requests.get(BASE, timeout=5)
t('HTTP 200', r.status_code == 200)
t('Content-Type text/html', 'text/html' in r.headers.get('Content-Type',''))
t('has NEUROSWAP brand', 'NEUROSWAP' in r.text)
t('has ENGAGE CTA', 'ENGAGE' in r.text)
t('has DISENGAGE CTA', 'DISENGAGE' in r.text)
t('has dropzone element', 'dropzone' in r.text)
t('size > 40KB', len(r.text) > 40000)

print('\n=== 2. GET /target-status (initial) ===')
r = requests.get(f'{BASE}/target-status', timeout=5)
t('HTTP 200', r.status_code == 200)
d = r.json()
t('models_loaded==True', d['models_loaded'] is True)
t('models_loading==False', d['models_loading'] is False)
t('models_error null', d['models_error'] is None)
t('has_target==False initial', d['has_target'] is False)
t('7 required keys present', set(d.keys()) >= {'has_target','current_target_id','current_target','targets','models_loaded','models_loading','models_error'})

print('\n=== 3. POST /upload-target (empty file) ===')
r = requests.post(f'{BASE}/upload-target', files={'file': ('a.jpg', b'', 'image/jpeg')}, timeout=10)
t('HTTP 400', r.status_code == 400)

print('\n=== 4. POST /upload-target (wrong MIME: text/plain) ===')
r = requests.post(f'{BASE}/upload-target', files={'file': ('x.txt', b'hello world', 'text/plain')}, timeout=10)
t('HTTP 400 unsupported type', r.status_code == 400 and ('Unsupported' in str(r.json().get('detail','')) or 'type' in str(r.json().get('detail','')).lower()))

print('\n=== 5. POST /upload-target (valid face image) ===')
jpg = make_jpg_bytes(600, 500, (100,200,100))
r = requests.post(f'{BASE}/upload-target', files={'file': ('face.jpg', jpg, 'image/jpeg')}, timeout=20)
t('HTTP 200', r.status_code == 200)
d = r.json()
tid1 = d.get('target_id')
t('target_id present', bool(tid1))
t('response status==ok', d.get('status')=='ok')
t('targets list length==1', len(d.get('targets',[]))==1)
t('thumbnail is dataURL', str(d.get('target',{}).get('thumb','')).startswith('data:image/jpeg;base64,'))
t('thumb > 1KB (160x160 padded crop now)', len(d.get('target',{}).get('thumb','')) > 1024)
t('detection_score 0.93', d.get('target',{}).get('detection_score') == 0.93)
t('file on disk in uploads/', any(s.endswith('.jpg') or s.endswith('.jpeg') for s in os.listdir(str(main.UPLOAD_DIR))))

print('\n=== 6. POST /upload-target — MAX_RECENT_TARGETS enforced (upload 11 more -> cap at 8) ===')
for i in range(11):
    jpg2 = make_jpg_bytes(500,400,(50+i*20,50,200))
    r = requests.post(f'{BASE}/upload-target', files={'file': (f'c{i}.png', jpg2, 'image/png')}, timeout=20)
    t(f'upload #{i+2} HTTP 200', r.status_code == 200)
r = requests.get(f'{BASE}/targets', timeout=5)
cnt = len(r.json()['targets'])
t(f'targets count {cnt} <= 8', cnt <= 8)

print('\n=== 7. POST /select-target (valid id) ===')
tid = r.json()['targets'][0]['id']
r2 = requests.post(f'{BASE}/select-target/{tid}', timeout=5)
t('HTTP 200', r2.status_code == 200)
t('target_id echoed', r2.json().get('target_id') == tid)
rs = requests.get(f'{BASE}/target-status', timeout=5)
t('has_target now True', rs.json()['has_target'] is True)
t('current_target_id matches', rs.json()['current_target_id'] == tid)

print('\n=== 8. POST /select-target (404 missing) ===')
r = requests.post(f'{BASE}/select-target/deadbeef1234', timeout=5)
t('HTTP 404', r.status_code == 404)
t('detail msg present', 'detail' in r.json())

print('\n=== 9. GET /targets ===')
r = requests.get(f'{BASE}/targets', timeout=5)
t('HTTP 200', r.status_code == 200)
t('keys: {current_target_id, targets}', set(r.json().keys()) == {'current_target_id','targets'})

print('\n=== 10. Static file serving ===')
r = requests.get(f'{BASE}/static-root/wsproto_stub.py', timeout=5, stream=True)
t('/static-root/wsproto_stub.py HTTP 200', r.status_code == 200)
t('bytes received > 1000', len(r.content) > 1000)

print('\n=== 11. CORS middleware (OPTIONS preflight + GET) ===')
r = requests.options(f'{BASE}/target-status',
    headers={'Origin': 'http://evil.com',
             'Access-Control-Request-Method': 'GET',
             'Access-Control-Request-Headers': 'X-Test'},
    timeout=5)
t('OPTIONS 200/204', r.status_code in (200, 204))
acao = r.headers.get('Access-Control-Allow-Origin', '')
t('ACAO header present (EITHER * OR echoed Origin)', acao in ('*', 'http://evil.com'))
# CORS GET also echoes origin with credentials=true setup:
r2 = requests.get(f'{BASE}/target-status', headers={'Origin': 'http://evil.com'}, timeout=5)
t('GET has ACAO header matching origin', r2.headers.get('Access-Control-Allow-Origin') in ('*', 'http://evil.com'))

print('\n=== 12. WS handshake + welcome frame + set_intensity clamping + frame roundtrip ===')
ws_connected = False
try:
    ws = websocket.create_connection(WS_BASE, timeout=15, enable_multithread=True)
    ws_connected = True
    # welcome frame
    ws.settimeout(4)
    try:
        welcome = json.loads(ws.recv())
        t('welcome frame received first', welcome.get('type') == 'welcome')
        t('welcome has_intensity', 'intensity' in welcome)
        t('welcome has_target==True (we set target above)', welcome.get('has_target') is True)
        t('welcome models_loaded==True', welcome.get('models_loaded') is True)
        t('welcome targets list present', isinstance(welcome.get('targets'), list) and len(welcome['targets'])>0)
    except Exception as e:
        t('welcome frame received', False)
        print('     welcome frame err:', repr(e))

    ws.send(json.dumps({'type':'set_intensity','value': 1.9}))
    ws.send(json.dumps({'type':'set_intensity','value': -0.4}))
    ws.send(json.dumps({'type':'set_intensity','value': 0.55}))
    b64 = 'data:image/jpeg;base64,' + base64.b64encode(make_jpg_bytes(640,480, (50,100,150))).decode()
    ws.send(json.dumps({'type':'frame','image':b64}))
    resp = json.loads(ws.recv())
    t('type = processed_frame', resp['type'] == 'processed_frame')
    t('intensity clamped to 0.55', resp['intensity'] == 0.55)
    t('2 detected faces', len(resp['faces']) == 2)
    f0 = resp['faces'][0]
    t('f0 swapped=True', f0['swapped'] is True)
    t('f0 bbox[4] ints', len(f0['bbox']) == 4 and all(isinstance(v, int) for v in f0['bbox']))
    t('f0 landmarks 5x2 floats', f0['landmarks'] is not None and len(f0['landmarks'])==5 and all(len(p)==2 for p in f0['landmarks']))
    t('f0 score = 0.93', f0['score'] == 0.93)
    t('latency_ms >= 0', isinstance(resp['latency_ms'], (int,float)) and resp['latency_ms'] >= 0)
    t('swap_quality_pct > 0 (no longer references loop var)', resp['swap_quality_pct'] > 0)
    t('frame_shape [480, 640] as list', resp['frame_shape'] == [480, 640])
    t('image dataURL jpeg', str(resp['image']).startswith('data:image/jpeg;base64,'))
    t('image size > 5KB', len(resp['image']) > 5120)
    ws.close()
except Exception as e:
    t('WS handshake + roundtrip', False)
    print('  WS EXCEPTION:', repr(e))
    import traceback; traceback.print_exc()
    if ws_connected:
        try: ws.close()
        except: pass

print('\n=== 13. WS robustness — garbage / unknown msgs then recover ===')
try:
    ws = websocket.create_connection(WS_BASE, timeout=15)
    try:
        # skip welcome
        ws.settimeout(3)
        try: ws.recv()
        except: pass
        ws.send(json.dumps({'foo':1}))                   # JSON missing type
        ws.send(json.dumps({'type': 'totally_unknown'}))# unknown type
        ws.send('not even json at all !!!')             # non-JSON text
        ws.send(json.dumps({}))                         # empty dict
        ws.send(json.dumps([]))                         # list
        valid_b64 = 'data:image/jpeg;base64,' + base64.b64encode(make_jpg_bytes(640,480)).decode()
        ws.send(json.dumps({'type':'frame','image':valid_b64}))
        resp = json.loads(ws.recv())
        t('recovers after 5 bad messages', resp['type'] == 'processed_frame')
        t('frame still has 2 faces', len(resp['faces']) == 2)
    finally:
        try: ws.close()
        except: pass
except Exception as e:
    t('WS robustness (exception): '+str(e), False)

print('\n=== 14. WS — no comma in dataURL edge case (raw b64 without prefix) ===')
try:
    ws = websocket.create_connection(WS_BASE, timeout=15)
    try:
        ws.settimeout(3)
        try: ws.recv()
        except: pass
        raw_b64 = base64.b64encode(make_jpg_bytes(640,480)).decode()
        ws.send(json.dumps({'type':'frame','image': raw_b64}))  # no prefix
        # Send a valid one right after to verify socket still alive
        valid_b64 = 'data:image/jpeg;base64,' + base64.b64encode(make_jpg_bytes(640,480)).decode()
        ws.send(json.dumps({'type':'frame','image':valid_b64}))
        ws.settimeout(5)
        got1 = json.loads(ws.recv())
        t('still alive after malformed image URL', got1['type'] == 'processed_frame')
    finally:
        try: ws.close()
        except: pass
except Exception as e:
    t('WS no-comma edge case (exception): '+str(e), False)

print('\n=== 15. WS — models_loading state error messages ===')
orig_analyzer, orig_swapper = main.face_analyzer, main.swapper
main.face_analyzer = None
main.swapper = None
try:
    ws = websocket.create_connection(WS_BASE, timeout=15)
    try:
        ws.settimeout(3)
        try:
            welcome = json.loads(ws.recv())
            t('welcome shows models_loaded==False when unset', welcome.get('models_loaded') is False)
        except: pass
        valid_b64 = 'data:image/jpeg;base64,' + base64.b64encode(make_jpg_bytes(640,480)).decode()
        ws.send(json.dumps({'type':'frame','image':valid_b64}))
        resp = json.loads(ws.recv())
        t('sends error type when models missing', resp['type'] == 'error')
        t('message contains "Models"', 'odels' in resp.get('message',''))
        t('has error discriminator field', resp.get('error') == 'models_loading')
    finally:
        try: ws.close()
        except: pass
finally:
    main.face_analyzer, main.swapper = orig_analyzer, orig_swapper

print('\n=== 16. WS — no_target state error messages ===')
orig_cid = main.current_target_id
with main._state_lock:
    main.current_target_id = 'definitely-does-not-exist-1234'
try:
    ws = websocket.create_connection(WS_BASE, timeout=15)
    try:
        ws.settimeout(3)
        try: ws.recv()
        except: pass
        valid_b64 = 'data:image/jpeg;base64,' + base64.b64encode(make_jpg_bytes(640,480)).decode()
        ws.send(json.dumps({'type':'frame','image':valid_b64}))
        resp = json.loads(ws.recv())
        t('sends error type when target invalid', resp['type'] == 'error')
        t('error discriminator == no_target', resp.get('error') == 'no_target')
    finally:
        try: ws.close()
        except: pass
finally:
    with main._state_lock:
        main.current_target_id = orig_cid

# ================== SUMMARY ==================
print()
print('='*50)
print(f'RESULT: {PASS} PASSED, {FAIL} FAILED out of {PASS+FAIL} checks')
print('='*50)

# shutdown server
try:
    server.should_exit = True
    server.force_exit = True
except Exception:
    pass
time.sleep(0.3)
sys.exit(0 if FAIL == 0 else 1)
