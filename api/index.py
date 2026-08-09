"""
Vercel serverless entrypoint (experimental).

This app relies on WebSockets, ffmpeg, and large ML models. Vercel serverless
is size- and time-limited — use Render (Docker) for production. See DEPLOYMENT.md.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mangum import Mangum  # noqa: E402
from main import app  # noqa: E402

handler = Mangum(app, lifespan="off")
