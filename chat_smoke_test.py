from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Dict, Any

import requests

API_URL = "http://127.0.0.1:8000"


def wait_for_ready(timeout: float = 120.0) -> bool:
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


def post_chat(messages: list[Dict[str, Any]]):
    return requests.post(API_URL + "/chat", json={"messages": messages}, timeout=180)


def run() -> int:
    # Start API server as subprocess
    python_exe = sys.executable
    server = subprocess.Popen([python_exe, "windsurf_skill.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        print("[chat-smoke] Waiting for API to become ready...")
        if not wait_for_ready():
            print("[chat-smoke][error] API did not become ready in time.")
            if server.stdout:
                print("[chat-smoke] Last few server lines:")
                for _ in range(100):
                    line = server.stdout.readline()
                    if not line:
                        break
                    print(line, end="")
            return 1

        # Off-topic request (should refuse)
        off_topic = [
            {"role": "user", "content": "Tell me a joke about penguins."}
        ]
        print("[chat-smoke] Posting off-topic request (expect refusal)...")
        resp_off = post_chat(off_topic)
        print(f"[chat-smoke] Off-topic status: {resp_off.status_code}")
        try:
            print(json.dumps(resp_off.json(), indent=2))
        except Exception:
            print(resp_off.text[:1000])

        # On-topic request (should answer)
        on_topic = [
            {"role": "user", "content": "How does FAISS cosine similarity help decide suggest-merge vs no-merge?"}
        ]
        print("[chat-smoke] Posting on-topic request (expect answer)...")
        resp_on = post_chat(on_topic)
        print(f"[chat-smoke] On-topic status: {resp_on.status_code}")
        try:
            print(json.dumps(resp_on.json(), indent=2))
        except Exception:
            print(resp_on.text[:1000])

        ok = resp_off.ok and resp_on.ok
        return 0 if ok else 2
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
    sys.exit(run())
