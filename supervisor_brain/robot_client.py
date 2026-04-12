# /root/autodl-tmp/robot_client.py
import time
import requests

class RobotClient:
    def __init__(self, robot_url: str, timeout: float = 2.0):
        self.robot_url = robot_url
        self.timeout = timeout

    def send(self, cmd: str, meta: dict | None = None) -> dict:
        payload = {"cmd": cmd, "ts": time.time(), "meta": meta or {}}
        r = requests.post(self.robot_url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()
