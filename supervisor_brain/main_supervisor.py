# /root/autodl-tmp/main_supervisor.py
import time, os, logging
from pathlib import Path

from supervisor_core import SupervisorCore, SystemState
from robot_client import RobotClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [SUPERVISOR] - %(message)s")

MONITOR_DIR = os.getenv("SUPERVISOR_MONITOR_DIR", "/root/autodl-tmp/share_photo")
CHECK_INTERVAL = float(os.getenv("SUPERVISOR_CHECK_INTERVAL", "0.5"))

API_KEY = os.getenv("SUPERVISOR_API_KEY", "")
ROBOT_URL = os.getenv("SUPERVISOR_ROBOT_URL", "http://ROBOT_IP:8000/cmd")

def latest_image(path: str):
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        files += list(Path(path).glob(ext))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)

def main():
    os.makedirs(MONITOR_DIR, exist_ok=True)

    if not API_KEY:
        raise ValueError("SUPERVISOR_API_KEY is required")

    core = SupervisorCore(api_key=API_KEY)
    robot = RobotClient(robot_url=ROBOT_URL, timeout=2.0)

    last_processed_time = 0.0
    logging.info(f"Start Monitoring: {MONITOR_DIR}")

    while True:
        try:
            lf = latest_image(MONITOR_DIR)
            if lf is None:
                time.sleep(CHECK_INTERVAL)
                continue

            mtime = lf.stat().st_mtime
            if mtime > last_processed_time:
                time.sleep(0.2)  # 防止文件写入中
                logging.info(f"Detected new image: {lf.name}")

                vision = core.vlm_reasoning(str(lf))
                state, cmd, meta = core.decide(vision)

                print("-"*40)
                print("STATE:", state)
                print("META :", meta)

                if cmd:
                    try:
                        resp = robot.send(cmd, meta=meta)
                        print("ROBOT RESP:", resp)
                    except Exception as e:
                        logging.error(f"Robot send failed: {e}")

                last_processed_time = mtime

        except Exception as e:
            logging.error(f"Monitor loop error: {e}")

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
