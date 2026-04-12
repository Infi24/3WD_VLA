import base64
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import cv2
import numpy as np
import torch

# ================= Import =================
sys.path.append("/home/robogoat/lerobot/src")
try:
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.robots.utils import make_robot_from_config
    from lerobot.robots.so_follower.so100_follower.config_so100_follower import (
        SO100FollowerConfig,
    )
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.utils.utils import get_safe_torch_device
    from lerobot.utils.control_utils import predict_action
except ImportError as e:
    print(f"Import failed: {e}")
    raise SystemExit(1)


# ================= Runtime config =================
PROMPT_MODE = os.getenv("PROMPT_MODE", "file")  # file | http
PROMPT_FILE_PATH = os.getenv("PROMPT_FILE_PATH", "/root/autodl-tmp/share_cmd/robot_prompt.txt")
PROMPT_POLL_INTERVAL = float(os.getenv("PROMPT_POLL_INTERVAL", "0.2"))

# IMPORTANT: change these two on local machine for cross-machine test.
BRIDGE_HOST = os.getenv("BRIDGE_HOST", "127.0.0.1")
BRIDGE_PORT = int(os.getenv("BRIDGE_PORT", "8000"))
BRIDGE_CLIENT_ID = os.getenv("BRIDGE_CLIENT_ID", "smallbrain_1")
BRIDGE_TIMEOUT_SEC = float(os.getenv("BRIDGE_TIMEOUT_SEC", "2.0"))
BRIDGE_USE_HTTP_FRAMES = os.getenv("BRIDGE_USE_HTTP_FRAMES", "0") == "1"

VIEWER_HOST = os.getenv("VIEWER_HOST", "127.0.0.1")
VIEWER_PORT = int(os.getenv("VIEWER_PORT", "9999"))
VIEWER_SEND_INTERVAL = float(os.getenv("VIEWER_SEND_INTERVAL", "999"))
VIEWER_INPUT_RGB = os.getenv("VIEWER_INPUT_RGB", "1") == "1"

TASK_STAGE_SEC = float(os.getenv("TASK_STAGE_SEC", "10.0"))
PROMPT_MAX_SEC = float(os.getenv("PROMPT_MAX_SEC", "20.0"))

REPO_ID = "EricChen06/so100_smolvla"
ROBOT_TYPE = "so100_follower"
STATS_PATH = "/home/robogoat/.cache/huggingface/hub/models--EricChen06--so100_smolvla/snapshots/3e92cc1bf6b8bc5ceb500050e7baea83f6755729/stats.json"


class FilePromptClient:
    """Read new lines from robot_prompt.txt as a FIFO task queue."""

    def __init__(self, prompt_file: str):
        self.prompt_file = Path(prompt_file)
        self.prompt_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.prompt_file.exists():
            self.prompt_file.write_text("", encoding="utf-8")

        self.current_prompt = "Waiting for instruction..."
        self.prompt_lock = threading.Lock()

        self.queue = []
        self.queue_lock = threading.Lock()
        self.next_line_idx = 0

        self.stop_event = threading.Event()
        self._load_existing_lines()
        self.reader_thread = threading.Thread(target=self._watch_file_loop, daemon=True)
        self.reader_thread.start()

        print(f"Prompt file: {self.prompt_file}")

    def _read_all_lines(self):
        txt = self.prompt_file.read_text(encoding="utf-8", errors="ignore")
        return [ln.strip() for ln in txt.splitlines() if ln.strip()]

    def _load_existing_lines(self):
        try:
            self.next_line_idx = len(self._read_all_lines())
        except Exception:
            self.next_line_idx = 0

    def _watch_file_loop(self):
        while not self.stop_event.is_set():
            try:
                lines = self._read_all_lines()
                if self.next_line_idx < len(lines):
                    new_lines = lines[self.next_line_idx :]
                    self.next_line_idx = len(lines)
                    with self.queue_lock:
                        self.queue.extend(new_lines)
                    print(f"Loaded {len(new_lines)} new prompt line(s)")
            except Exception as e:
                print(f"Prompt file read error: {e}")
            time.sleep(PROMPT_POLL_INTERVAL)

    def get_latest_prompt(self):
        with self.prompt_lock:
            return self.current_prompt

    def pull_next_prompt(self):
        with self.queue_lock:
            if not self.queue:
                return None
            nxt = self.queue.pop(0)
        with self.prompt_lock:
            self.current_prompt = nxt
        print(f"Switched prompt -> {nxt}")
        return nxt

    def set_waiting(self):
        with self.prompt_lock:
            self.current_prompt = "Waiting for instruction..."

    def close(self):
        self.stop_event.set()


class HttpPromptClient:
    """Pull prompt lines from cloud bridge: GET /next_prompt?client_id=..."""

    def __init__(self, host: str, port: int, client_id: str):
        self.base_url = f"http://{host}:{port}"
        self.client_id = client_id
        self.current_prompt = "Waiting for instruction..."
        self.prompt_lock = threading.Lock()
        print(f"Prompt bridge: {self.base_url} client_id={self.client_id}")

    def _get_json(self, url: str):
        req = Request(url, method="GET")
        with urlopen(req, timeout=BRIDGE_TIMEOUT_SEC) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def get_latest_prompt(self):
        with self.prompt_lock:
            return self.current_prompt

    def pull_next_prompt(self):
        qs = urlencode({"client_id": self.client_id})
        url = f"{self.base_url}/next_prompt?{qs}"
        try:
            data = self._get_json(url)
            if data.get("status") == "success" and data.get("has_prompt"):
                nxt = data.get("prompt", "").strip()
                if nxt:
                    with self.prompt_lock:
                        self.current_prompt = nxt
                    print(f"Switched prompt -> {nxt}")
                    return nxt
            return None
        except URLError as e:
            print(f"Bridge prompt pull error: {e}")
            return None
        except Exception as e:
            print(f"Bridge prompt decode error: {e}")
            return None

    def set_waiting(self):
        with self.prompt_lock:
            self.current_prompt = "Waiting for instruction..."

    def close(self):
        return None


class UdpViewerPublisher:
    """Send hand/top frames to share_viewer.py (UDP packet: tag + jpeg bytes)."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"Viewer UDP target: {self.host}:{self.port}")

    def _to_uint8(self, img):
        if isinstance(img, torch.Tensor):
            arr = img.detach().cpu().numpy()
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            return arr

        arr = np.asarray(img)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    def _encode_b64_jpg(self, img):
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return ""
        return base64.b64encode(enc.tobytes()).decode("utf-8")

    def send_pair(self, hand_img, top_img):
        try:
            hand = self._to_uint8(hand_img)
            top = self._to_uint8(top_img)
            # lerobot images are typically RGB; OpenCV expects BGR
            if VIEWER_INPUT_RGB:
                hand = cv2.cvtColor(hand, cv2.COLOR_RGB2BGR)
                top = cv2.cvtColor(top, cv2.COLOR_RGB2BGR)
            ok_h, enc_h = cv2.imencode(".jpg", hand, [cv2.IMWRITE_JPEG_QUALITY, 85])
            ok_t, enc_t = cv2.imencode(".jpg", top, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok_h and ok_t:
                self.sock.sendto(b"H" + enc_h.tobytes(), (self.host, self.port))
                self.sock.sendto(b"T" + enc_t.tobytes(), (self.host, self.port))

            if BRIDGE_USE_HTTP_FRAMES:
                payload = {
                    "image_hand_b64": self._encode_b64_jpg(hand),
                    "image_top_b64": self._encode_b64_jpg(top),
                }
                body = json.dumps(payload).encode("utf-8")
                req = Request(
                    f"http://{BRIDGE_HOST}:{BRIDGE_PORT}/frames",
                    data=body,
                    method="POST",
                    headers={"Content-Type": "application/json"},
                )
                with urlopen(req, timeout=BRIDGE_TIMEOUT_SEC):
                    pass
        except Exception as e:
            print(f"Viewer publish error: {e}")

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


def load_and_reorder_stats(path, device):
    path = Path(path)
    if not path.exists():
        print(f"Stats file not found: {path}")
        return None

    print(f"Loading stats: {path}")
    with open(path, "r", encoding="utf-8") as f:
        stats_dict = json.load(f)

    reorder_indices = [3, 2, 0, 4, 5, 1]
    for key in ["action", "observation.state"]:
        if key in stats_dict:
            for stat_type in ["mean", "std", "min", "max"]:
                if stat_type in stats_dict[key]:
                    original_list = stats_dict[key][stat_type]
                    reordered_list = [original_list[i] for i in reorder_indices]
                    stats_dict[key][stat_type] = torch.tensor(
                        reordered_list, dtype=torch.float32, device=device
                    )
    return stats_dict


def resolve_task_from_prompt(prompt: str, elapsed_sec: float) -> str:
    tokens = [t.strip() for t in prompt.split("|") if t.strip()]
    if tokens and tokens[-1].upper() == "END":
        tokens = tokens[:-1]

    if len(tokens) >= 2:
        return tokens[0] if elapsed_sec < TASK_STAGE_SEC else tokens[1]
    if len(tokens) == 1:
        return tokens[0]
    return prompt


def build_prompt_client():
    mode = PROMPT_MODE.lower().strip()
    if mode == "http":
        return HttpPromptClient(BRIDGE_HOST, BRIDGE_PORT, BRIDGE_CLIENT_ID)
    return FilePromptClient(PROMPT_FILE_PATH)


def main():
    device = get_safe_torch_device("cuda")
    print(f"Using device: {device}")

    prompt_client = build_prompt_client()
    viewer_pub = UdpViewerPublisher(VIEWER_HOST, VIEWER_PORT)

    policy_cfg = PreTrainedConfig.from_pretrained(REPO_ID)
    policy_cls = get_policy_class(policy_cfg.type)
    policy = policy_cls.from_pretrained(REPO_ID)
    policy.to(device)
    policy.eval()

    physical_stats = load_and_reorder_stats(STATS_PATH, device)
    if physical_stats is None:
        print("Critical warning: stats load failed")

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=REPO_ID,
        dataset_stats=physical_stats,
    )

    print("Connecting robot...")
    handeye_conf = OpenCVCameraConfig(
    index_or_path="/dev/v4l/by-path/pci-0000:00:14.0-usb-0:6.3:1.0-video-index0",
    width=640, height=360, fps=30
    )
    top_conf = OpenCVCameraConfig(
        index_or_path="/dev/v4l/by-path/pci-0000:00:14.0-usb-0:2:1.0-video-index0",
        width=640, height=360, fps=30
    )

    robot_cfg = SO100FollowerConfig(
        port="/dev/ttyACM0",
        calibration_dir=None,
        id=None,
        cameras={"handeye": handeye_conf, "top": top_conf},
    )
    robot = make_robot_from_config(robot_cfg)
    robot.connect()
    print("Robot connected")

    hz = 30
    dt = 1.0 / hz
    start_time = time.time()
    last_viewer_send_ts = 0.0

    physical_motor_names = [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ]

    try:
        while True:
            loop_start = time.time()

            new_prompt = prompt_client.pull_next_prompt()
            if new_prompt is not None:
                start_time = time.time()

            elapsed = time.time() - start_time
            cloud_prompt = prompt_client.get_latest_prompt()

            if cloud_prompt != "Waiting for instruction..." and elapsed >= PROMPT_MAX_SEC:
                prompt_client.set_waiting()
                cloud_prompt = "Waiting for instruction..."

            raw_obs = robot.get_observation()
            now = time.time()
            if now - last_viewer_send_ts >= VIEWER_SEND_INTERVAL:
                viewer_pub.send_pair(raw_obs["handeye"], raw_obs["top"])
                last_viewer_send_ts = now

            if cloud_prompt == "Waiting for instruction...":
                print("Waiting for next prompt ...")
                time.sleep(max(0, dt - (time.time() - loop_start)))
                continue

            task = resolve_task_from_prompt(cloud_prompt, elapsed)

            current_state = [raw_obs[name] for name in physical_motor_names]
            observation_dict = {
                "observation.state": np.array(current_state, dtype=np.float32),
                "observation.images.handeye": raw_obs["handeye"],
                "observation.images.top": raw_obs["top"],
            }

            action_tensor = predict_action(
                observation=observation_dict,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                task=task,
                robot_type=ROBOT_TYPE,
                use_amp=False,
            )

            if action_tensor is not None:
                action_np = action_tensor.cpu().numpy().squeeze()
                action_dict = dict(zip(physical_motor_names, action_np))
                robot.send_action(action_dict)

            time.sleep(max(0, dt - (time.time() - loop_start)))

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        try:
            prompt_client.close()
        except Exception:
            pass
        try:
            viewer_pub.close()
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
