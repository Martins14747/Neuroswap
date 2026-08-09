import sys, os, time, socket, json, base64, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import site
venv_sp = '/home/me/Documents/SOAIR/face-swap-web/venv/lib/python3.13/site-packages'
if venv_sp not in sys.path: sys.path.insert(0, venv_sp)
for p in site.getsitepackages() + [site.getusersitepackages(), '/usr/lib/python3/dist-packages']:
    if os.path.isdir(p) and p not in sys.path: sys.path.append(p)

os.environ.setdefault("CORS_ORIGINS", "http://x.com")

import numpy as np, cv2
import insightface

class FakeFace:
    def __init__(self, idx=0):
        self.bbox = np.array([50+idx*30, 50+idx*20, 200+idx*30, 220+idx*20], dtype=np.float32)
        self.det_score = 0.93
        self.kps = np.array([[90.,100.],[160.,100.],[125.,140.],[100.,180.],[150.,180.]], dtype=np.float32)
class FakeAnalyzer:
    def prepare(self, *a, **kw): pass
    def get(self, img):
        return [FakeFace(0), FakeFace(1)] if img is not None and img.shape[0] > 100 else []
class FakeSwapper:
    def get(self, frame, face, target_face, paste_back=True):
        faked = frame.copy(); b = face.bbox.astype(int)
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

import uvicorn, requests, websocket
import subprocess, tempfile

def free_port():
    s=socket.socket(); s.bind(('127.0.0.1',0)); p=s.getsockname()[1]; s.close(); return p
PORT = free_port()
BASE = f'http://127.0.0.1:{PORT}'
WS_BASE = f'ws://127.0.0.1:{PORT}/ws'

cfg = uvicorn.Config(main.app, host='127.0.0.1', port=PORT, log_level='error', access_log=False)
srv = uvicorn.Server(cfg)
threading.Thread(target=srv.run, daemon=True).start()
for _ in range(60):
    try: requests.get(f'{BASE}/target-status', timeout=0.5); break
    except: time.sleep(0.1)

def make_jpg(w=400,h=300,c=(80,120,180)):
    img=np.zeros((h,w,3),dtype=np.uint8); img[:]=c
    cv2.circle(img,(w//2,h//2),60,(255,255,255),-1); cv2.rectangle(img,(60,60),(240,240),(0,255,0),3)
    return cv2.imencode('.jpg', img)[1].tobytes()

P=0;F=0
def t(n,c):
    global P,F
    if c: P +=1
    else: F +=1
    print(('  [OK] ' if c else '  [FAIL] ')+n); return bool(c)

print('1. home')
r=requests.get(BASE,timeout=5); t('200',r.status_code==200); t('brand in html','NEUROSWAP' in r.text); t('>40KB',len(r.text)>40000)
print('2. target-status initial')
r=requests.get(f'{BASE}/target-status',timeout=5); d=r.json()
t('200',r.status_code==200); t('models_loaded',d['models_loaded'] is True); t('models_loading false',d['models_loading'] is False); t('no target',d['has_target'] is False)
print('3. empty upload 400')
r=requests.post(f'{BASE}/upload-target',files={'file':('a.jpg',b'','image/jpeg')},timeout=10); t('400 empty',r.status_code==400)
print('4. wrong mime 400')
r=requests.post(f'{BASE}/upload-target',files={'file':('x.txt',b'hello','text/plain')},timeout=10); t('400 bad mime',r.status_code==400)
print('5. valid upload')
jpg=make_jpg(600,500,(100,200,100))
r=requests.post(f'{BASE}/upload-target',files={'file':('face.jpg',jpg,'image/jpeg')},timeout=20); t('200',r.status_code==200)
d=r.json(); tid1=d['target_id']
t('has id',bool(tid1)); t('1 target',len(d['targets'])==1)
t('thumb dataURL',d['target']['thumb'].startswith('data:image/jpeg;base64,'))
t('thumb >1KB',len(d['target']['thumb'])>1024); t('score 0.93',d['target']['detection_score']==0.93)
print('6. 11 more uploads cap at 8')
for i in range(11):
    r=requests.post(f'{BASE}/upload-target',files={'file':(f'c{i}.png',make_jpg(500,400,(50+i*20,50,200)),'image/png')},timeout=20); t(f'upload {i+2}',r.status_code==200)
r=requests.get(f'{BASE}/targets',timeout=5); cnt=len(r.json()['targets']); t(f'cnt {cnt}<=8',cnt<=8)
last_tid=r.json()['targets'][-1]['id']
print('7. select first (change current)')
first_tid=r.json()['targets'][0]['id']
r2=requests.post(f'{BASE}/select-target/{first_tid}',timeout=5); t('200',r2.status_code==200); t('id echoed',r2.json()['target_id']==first_tid)
rs=requests.get(f'{BASE}/target-status',timeout=5); t('target set true',rs.json()['has_target'] is True)
print('8. 404 select')
r=requests.post(f'{BASE}/select-target/xxxx',timeout=5); t('404',r.status_code==404)
print('9. static serving')
r=requests.get(f'{BASE}/static-root/wsproto_stub.py',timeout=5); t('200',r.status_code==200); t('>1KB',len(r.content)>1000)
print('10. CORS')
r=requests.options(f'{BASE}/target-status',headers={'Origin':'http://x.com','Access-Control-Request-Method':'GET','Access-Control-Request-Headers':'X-Test'},timeout=5)
acao=r.headers.get('Access-Control-Allow-Origin',''); t('OPTIONS ok',r.status_code in (200,204)); t('ACAO echoed',acao == 'http://x.com')
print('11. WS — welcome, intensity, roundtrip')
try:
    ws=websocket.create_connection(WS_BASE,timeout=12)
    ws.settimeout(4)
    try:
        welcome=json.loads(ws.recv())
        t('welcome frame type',welcome.get('type')=='welcome')
        t('welcome has_target true',welcome.get('has_target') is True)
        t('welcome models_loaded true',welcome.get('models_loaded') is True)
        t('welcome intensity 1.0',welcome.get('intensity')==1.0)
        t('welcome targets list',len(welcome.get('targets',[]))>0)
    except Exception as e: t('welcome',False); print('   err:',repr(e))
    ws.send(json.dumps({'type':'set_intensity','value':0.55}))
    b64='data:image/jpeg;base64,'+base64.b64encode(make_jpg(640,480,(50,100,150))).decode()
    ws.send(json.dumps({'type':'frame','image':b64}))
    resp=json.loads(ws.recv())
    t('resp processed_frame',resp['type']=='processed_frame')
    t('intensity 0.55',resp['intensity']==0.55)
    t('2 faces',len(resp['faces'])==2)
    f0=resp['faces'][0]
    t('f0 swapped',f0['swapped'] is True); t('f0 bbox 4',len(f0['bbox'])==4)
    t('f0 landmarks 5x2',f0['landmarks'] and len(f0['landmarks'])==5 and all(len(p)==2 for p in f0['landmarks']))
    t('lat',isinstance(resp['latency_ms'],(int,float)) and resp['latency_ms']>=0)
    t('quality>0',resp['swap_quality_pct']>0); t('shape',resp['frame_shape']==[480,640])
    t('img dataURL',str(resp['image']).startswith('data:image/jpeg;base64,')); t('img>5KB',len(resp['image'])>5120)
    ws.close()
except Exception as e: t('WS RT',False); print('  EXC:',repr(e))
print('12. WS robustness — 5 bad msgs then recover')
try:
    ws=websocket.create_connection(WS_BASE,timeout=12); ws.settimeout(4)
    try: ws.recv()
    except: pass
    ws.send(json.dumps({'foo':1})); ws.send(json.dumps({'type':'unknown'}))
    ws.send('notjson!!!'); ws.send(json.dumps({})); ws.send(json.dumps([]))
    valid_b64='data:image/jpeg;base64,'+base64.b64encode(make_jpg(640,480)).decode()
    ws.send(json.dumps({'type':'frame','image':valid_b64}))
    resp=json.loads(ws.recv())
    t('recovered',resp['type']=='processed_frame'); t('still 2 faces',len(resp['faces'])==2)
    ws.close()
except Exception as e: t('robust WS',False); print('  EXC:',repr(e))
print('13. WS error states — models, no target')
orig_a,orig_s = main.face_analyzer,main.swapper
main.face_analyzer=None; main.swapper=None
try:
    ws=websocket.create_connection(WS_BASE,timeout=12); ws.settimeout(4)
    try:
        w=json.loads(ws.recv()); t('welcome mod-load=false',w.get('models_loaded') is False)
    except: pass
    ws.send(json.dumps({'type':'frame','image':'data:image/jpeg;base64,'+base64.b64encode(make_jpg(640,480)).decode()}))
    resp=json.loads(ws.recv())
    t('error type',resp['type']=='error'); t('disc models',resp.get('error')=='models_loading')
    ws.close()
finally: main.face_analyzer,main.swapper = orig_a,orig_s
orig_cid = main.current_target_id
with main._state_lock: main.current_target_id='invalid-cid-xx'
try:
    ws=websocket.create_connection(WS_BASE,timeout=12); ws.settimeout(4)
    try: ws.recv()
    except: pass
    ws.send(json.dumps({'type':'frame','image':'data:image/jpeg;base64,'+base64.b64encode(make_jpg(640,480)).decode()}))
    resp=json.loads(ws.recv())
    t('no target error',resp['type']=='error'); t('disc no_target',resp.get('error')=='no_target')
    ws.close()
finally:
    with main._state_lock: main.current_target_id=orig_cid

print('14. video upload/analyze/process/download')
try:
    ffmpeg = subprocess.run(['bash','-lc','command -v ffmpeg'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout.strip()
    ffprobe = subprocess.run(['bash','-lc','command -v ffprobe'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout.strip()
    t('ffmpeg present', bool(ffmpeg))
    t('ffprobe present', bool(ffprobe))
    if ffmpeg and ffprobe:
        tmpdir = tempfile.mkdtemp(prefix='neuroswap_test_')
        in_path = os.path.join(tmpdir, 'in.mp4')
        out_path = os.path.join(tmpdir, 'out.mp4')
        cmd = [
            ffmpeg, '-y',
            '-f', 'lavfi', '-i', 'testsrc=size=320x240:rate=15',
            '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=44100',
            '-t', '2',
            '-pix_fmt', 'yuv420p',
            '-c:v', 'libx264', '-preset', 'ultrafast',
            '-c:a', 'aac',
            '-shortest',
            in_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        with open(in_path, 'rb') as f:
            r = requests.post(f'{BASE}/upload-video', files={'file': ('in.mp4', f, 'video/mp4')}, timeout=120)
        t('upload-video 200', r.status_code == 200)
        vid = r.json().get('video_id')
        t('video_id returned', bool(vid))
        r = requests.post(f'{BASE}/analyze-video', json={'video_id': vid}, timeout=120)
        t('analyze-video 200', r.status_code == 200)
        tracks = r.json().get('tracks') or []
        t('tracks array', isinstance(tracks, list))
        r = requests.post(f'{BASE}/process-video', json={'video_id': vid, 'intensity': 1.0, 'face_map': {}}, timeout=120)
        t('process-video 200', r.status_code == 200)
        job = r.json().get('job_id')
        t('job_id returned', bool(job))
        status = None
        for _ in range(80):
            rs = requests.get(f'{BASE}/video-status/{job}', timeout=20)
            if rs.status_code != 200:
                time.sleep(0.25)
                continue
            status = rs.json()
            if status.get('status') in ('done','error'):
                break
            time.sleep(0.25)
        t('job done', status is not None and status.get('status') == 'done')
        if status and status.get('status') == 'done':
            dl = requests.get(f'{BASE}/download-video/{job}', timeout=120)
            t('download-video 200', dl.status_code == 200 and len(dl.content) > 10000)
            with open(out_path, 'wb') as f:
                f.write(dl.content)
            probe = subprocess.run([ffprobe, '-v', 'error', '-show_streams', '-of', 'json', out_path],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True).stdout
            meta = json.loads(probe or '{}')
            streams = meta.get('streams') or []
            has_v = any(s.get('codec_type') == 'video' for s in streams)
            has_a = any(s.get('codec_type') == 'audio' for s in streams)
            t('output has video', has_v)
            t('output has audio', has_a)
except Exception as e:
    t('video pipeline exception: '+repr(e), False)

print()
print(f'RESULT: {P} PASSED, {F} FAILED out of {P+F}')
try: srv.should_exit = True
except: pass
time.sleep(0.3)
sys.exit(0 if F==0 else 1)
