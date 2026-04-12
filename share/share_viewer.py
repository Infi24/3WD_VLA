import os
import socket

import cv2
import numpy as np

# Runtime config (can be overridden by env vars)
HOST = os.getenv("VIEWER_HOST", "0.0.0.0")
PORT = int(os.getenv("VIEWER_PORT", "9999"))
SAVE_DIR = os.getenv("VIEWER_SAVE_DIR", "/root/autodl-tmp/share_photo")
TARGET_HEIGHT = int(os.getenv("VIEWER_TARGET_HEIGHT", "384"))
ADD_SEPARATOR = os.getenv("VIEWER_ADD_SEPARATOR", "1") == "1"
SEPARATOR_WIDTH = int(os.getenv("VIEWER_SEPARATOR_WIDTH", "5"))
CLEAR_ON_START = os.getenv("VIEWER_CLEAR_ON_START", "1") == "1"
SEPARATOR_COLOR = (128, 128, 128)  # BGR


def stitch_images(top_img, hand_img):
    # Keep the same stitch format as training distribution:
    # Top (left) | separator | Hand (right), fixed target height.
    h_top, w_top = top_img.shape[:2]
    h_hand, w_hand = hand_img.shape[:2]

    scale_top = TARGET_HEIGHT / h_top
    scale_hand = TARGET_HEIGHT / h_hand
    new_w_top = int(w_top * scale_top)
    new_w_hand = int(w_hand * scale_hand)

    top_resized = cv2.resize(
        top_img, (new_w_top, TARGET_HEIGHT), interpolation=cv2.INTER_LANCZOS4
    )
    hand_resized = cv2.resize(
        hand_img, (new_w_hand, TARGET_HEIGHT), interpolation=cv2.INTER_LANCZOS4
    )

    if ADD_SEPARATOR:
        sep = np.full(
            (TARGET_HEIGHT, SEPARATOR_WIDTH, 3), SEPARATOR_COLOR, dtype=np.uint8
        )
        return np.hstack((top_resized, sep, hand_resized))
    return np.hstack((top_resized, hand_resized))


def start_viewer():
    os.makedirs(SAVE_DIR, exist_ok=True)
    if CLEAR_ON_START:
        for f in os.listdir(SAVE_DIR):
            if f == "brain_ready.jpg" or f.startswith("combined_"):
                try:
                    os.remove(os.path.join(SAVE_DIR, f))
                except OSError:
                    pass
    print(f"[Viewer] save_dir={os.path.abspath(SAVE_DIR)}")
    print(
        f"[Viewer] host={HOST} port={PORT} target_h={TARGET_HEIGHT} sep={ADD_SEPARATOR}"
    )

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((HOST, PORT))
    sock.setblocking(True)
    print(f"[Viewer] listening UDP on {HOST}:{PORT}")

    pair_idx = 0
    last_hand = None
    last_top = None

    try:
        while True:
            data, _ = sock.recvfrom(65535)
            if not data:
                continue

            tag = data[0]
            img_data = data[1:]
            frame = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            if tag == ord("H"):
                cv2.imshow("HandEye", frame)
                last_hand = frame
            elif tag == ord("T"):
                cv2.imshow("TopView", frame)
                last_top = frame

            if (last_hand is not None) and (last_top is not None):
                pair_idx += 1
                combined_img = stitch_images(last_top, last_hand)
                latest_path = os.path.join(SAVE_DIR, "brain_ready.jpg")
                archive_path = os.path.join(SAVE_DIR, f"combined_{pair_idx + 1}.jpg")
                if pair_idx == 1:
                    cv2.imwrite(latest_path, combined_img)
                cv2.imwrite(archive_path, combined_img)
                print(f"[Viewer] pair={pair_idx} latest={latest_path}")
                last_hand = None
                last_top = None

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        cv2.destroyAllWindows()
        print("[Viewer] closed")


if __name__ == "__main__":
    start_viewer()
