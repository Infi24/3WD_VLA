import argparse
import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from share.share_viewer import stitch_images


class PromptStore:
    def __init__(self, prompt_file: str):
        self.prompt_file = Path(prompt_file)
        self.prompt_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.prompt_file.exists():
            self.prompt_file.write_text("", encoding="utf-8")

        self.lock = threading.Lock()
        self.client_offsets = {}

    def _read_lines(self):
        txt = self.prompt_file.read_text(encoding="utf-8", errors="ignore")
        return [ln.strip() for ln in txt.splitlines() if ln.strip()]

    def next_for_client(self, client_id: str):
        with self.lock:
            lines = self._read_lines()
            idx = self.client_offsets.get(client_id, 0)
            # If prompt file was truncated/reset between runs, rewind client cursor.
            if idx > len(lines):
                idx = 0
                self.client_offsets[client_id] = 0
            if idx < len(lines):
                prompt = lines[idx]
                self.client_offsets[client_id] = idx + 1
                return {
                    "status": "success",
                    "has_prompt": True,
                    "prompt": prompt,
                    "index": idx,
                    "total": len(lines),
                }
            return {
                "status": "success",
                "has_prompt": False,
                "prompt": "Waiting for instruction...",
                "index": idx,
                "total": len(lines),
            }


class FrameStore:
    def __init__(self, monitor_dir: str, clear_on_start: bool):
        self.monitor_dir = Path(monitor_dir)
        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.counter = 0
        if clear_on_start:
            for p in self.monitor_dir.glob("brain_ready.jpg"):
                p.unlink(missing_ok=True)
            for p in self.monitor_dir.glob("combined_*.jpg"):
                p.unlink(missing_ok=True)

    def write_pair(self, hand_b64: str, top_b64: str):
        hand = self._decode_b64_image(hand_b64)
        top = self._decode_b64_image(top_b64)
        if hand is None or top is None:
            raise ValueError("invalid image payload")

        merged = stitch_images(top, hand)
        with self.lock:
            self.counter += 1
            idx = self.counter
            latest_path = self.monitor_dir / "brain_ready.jpg"
            archive_path = self.monitor_dir / f"combined_{idx + 1}.jpg"
            if idx == 1:
                cv2.imwrite(str(latest_path), merged)
            cv2.imwrite(str(archive_path), merged)

        return {
            "status": "success",
            "saved": {
                "latest": str(latest_path),
                "archive": str(archive_path),
            },
            "pair_index": idx,
        }

    @staticmethod
    def _decode_b64_image(b64_str: str):
        try:
            raw = base64.b64decode(b64_str)
            arr = np.frombuffer(raw, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return None


def make_handler(prompt_store: PromptStore, frame_store: FrameStore):
    class BridgeHandler(BaseHTTPRequestHandler):
        server_version = "3WD-Bridge/1.0"

        def _send_json(self, code: int, obj: dict):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/health":
                return self._send_json(200, {"status": "ok", "ts": time.time()})

            if u.path == "/next_prompt":
                qs = parse_qs(u.query)
                client_id = qs.get("client_id", ["default"])[0]
                return self._send_json(200, prompt_store.next_for_client(client_id))

            return self._send_json(404, {"status": "error", "message": "not found"})

        def do_POST(self):
            u = urlparse(self.path)
            if u.path != "/frames":
                return self._send_json(404, {"status": "error", "message": "not found"})

            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                data = json.loads(body.decode("utf-8"))
                hand_b64 = data.get("image_hand_b64", "")
                top_b64 = data.get("image_top_b64", "")
                if not hand_b64 or not top_b64:
                    return self._send_json(400, {"status": "error", "message": "missing image fields"})

                out = frame_store.write_pair(hand_b64, top_b64)
                return self._send_json(200, out)
            except Exception as e:
                return self._send_json(500, {"status": "error", "message": str(e)})

        def log_message(self, fmt, *args):
            print(f"[bridge] {self.address_string()} - {fmt % args}")

    return BridgeHandler


def main():
    parser = argparse.ArgumentParser(description="Bridge for cross-machine smallbrain <-> cloud")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt_file", type=str, default="/root/autodl-tmp/share_cmd/robot_prompt.txt")
    parser.add_argument("--monitor_dir", type=str, default="/root/autodl-tmp/share_photo")
    parser.add_argument("--clear_monitor_dir", action="store_true")
    args = parser.parse_args()

    prompt_store = PromptStore(args.prompt_file)
    frame_store = FrameStore(args.monitor_dir, args.clear_monitor_dir)

    handler_cls = make_handler(prompt_store, frame_store)
    server = ThreadingHTTPServer((args.host, args.port), handler_cls)

    print(f"[bridge] listening on http://{args.host}:{args.port}")
    print(f"[bridge] prompt_file={args.prompt_file}")
    print(f"[bridge] monitor_dir={args.monitor_dir}")
    print("[bridge] endpoints: GET /health, GET /next_prompt?client_id=..., POST /frames")
    server.serve_forever()


if __name__ == "__main__":
    main()
