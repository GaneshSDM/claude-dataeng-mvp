import sys
from pathlib import Path
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root for core.*

BASE = "http://127.0.0.1:8000"


def get(path, **kw):
    try:
        r = requests.get(f"{BASE}{path}", timeout=5, **kw)
        return r.json() if r.ok else None
    except requests.RequestException:
        return None


def post(path, payload):
    try:
        r = requests.post(f"{BASE}{path}", json=payload, timeout=5)
        return r.json() if r.ok else {"error": r.text}
    except requests.RequestException as e:
        return {"error": str(e)}
