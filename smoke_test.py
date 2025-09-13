from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

API_URL = "http://127.0.0.1:8000"


def wait_for_ready(timeout: float = 180.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(API_URL + "/")
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def run_smoke_test() -> int:
    # Start API server as subprocess
    env = os.environ.copy()
    python_exe = sys.executable
    server = subprocess.Popen([python_exe, "windsurf_skill.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        print("[smoke] Waiting for API to become ready (first run may download models)...")
        if not wait_for_ready():
            print("[smoke][error] API did not become ready in time. Server output follows:")
            try:
                # Drain some logs for debugging
                if server.stdout:
                    for _ in range(200):
                        line = server.stdout.readline()
                        if not line:
                            break
                        print(line, end="")
            except Exception:
                pass
            return 1

        # Build example payload
        payload = {
            "records_a": [
                {"name": "Acme Corp", "address": "123 Main St", "city": "Springfield", "country": "US"},
                {"name": "Globex LLC", "address": "200 State Ave", "city": "Shelbyville", "country": "US"},
            ],
            "records_b": [
                {"name": "ACME Corporation", "address": "123 Main Street", "city": "Springfield", "country": "USA"},
                {"name": "Initech", "address": "300 Office Park", "city": "Springfield", "country": "US"},
            ],
            "text_fields": ["name", "address", "city", "country"],
        }

        print("[smoke] Posting example payload to /reconcile ...")
        resp = requests.post(API_URL + "/reconcile", json=payload, timeout=300)
        print(f"[smoke] Status: {resp.status_code}")
        try:
            data = resp.json()
            print("[smoke] Response JSON summary:")
            print(json.dumps(data.get("summary", data), indent=2))
        except Exception:
            print("[smoke] Raw response:")
            print(resp.text[:5000])

        return 0 if resp.ok else 2
    finally:
        try:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(run_smoke_test())
