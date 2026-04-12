import argparse
from pathlib import Path

import cv2
import numpy as np


def stitch_images(
    top_img,
    hand_img,
    target_height: int = 384,
    add_separator: bool = True,
    separator_width: int = 5,
    separator_color=(128, 128, 128),
):
    h_top, w_top = top_img.shape[:2]
    h_hand, w_hand = hand_img.shape[:2]
    scale_top = target_height / h_top
    scale_hand = target_height / h_hand
    new_w_top = int(w_top * scale_top)
    new_w_hand = int(w_hand * scale_hand)

    top_resized = cv2.resize(
        top_img, (new_w_top, target_height), interpolation=cv2.INTER_LANCZOS4
    )
    hand_resized = cv2.resize(
        hand_img, (new_w_hand, target_height), interpolation=cv2.INTER_LANCZOS4
    )
    if add_separator:
        sep = np.full((target_height, separator_width, 3), separator_color, dtype=np.uint8)
        return np.hstack((top_resized, sep, hand_resized))
    return np.hstack((top_resized, hand_resized))


def iter_classes(top_root: Path, hand_root: Path):
    top_classes = {p.name for p in top_root.iterdir() if p.is_dir()}
    hand_classes = {p.name for p in hand_root.iterdir() if p.is_dir()}
    for c in sorted(top_classes & hand_classes):
        yield c


def main():
    parser = argparse.ArgumentParser(description="Build merged top+hand dataset")
    parser.add_argument("--top_root", required=True, type=str)
    parser.add_argument("--hand_root", required=True, type=str)
    parser.add_argument("--out_root", required=True, type=str)
    parser.add_argument("--target_height", default=384, type=int)
    parser.add_argument("--add_separator", action="store_true")
    parser.add_argument("--separator_width", default=5, type=int)
    args = parser.parse_args()

    top_root = Path(args.top_root)
    hand_root = Path(args.hand_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    total = 0
    for cls in iter_classes(top_root, hand_root):
        top_cls = top_root / cls
        hand_cls = hand_root / cls
        out_cls = out_root / cls
        out_cls.mkdir(parents=True, exist_ok=True)

        top_files = {p.name: p for p in top_cls.glob("*.jpg")}
        hand_files = {p.name: p for p in hand_cls.glob("*.jpg")}
        common = sorted(set(top_files.keys()) & set(hand_files.keys()), key=lambda x: int(Path(x).stem))
        if not common:
            print(f"[WARN] no matched files in class={cls}")
            continue

        for name in common:
            top_img = cv2.imread(str(top_files[name]))
            hand_img = cv2.imread(str(hand_files[name]))
            if top_img is None or hand_img is None:
                print(f"[WARN] failed to read {cls}/{name}")
                continue

            merged = stitch_images(
                top_img=top_img,
                hand_img=hand_img,
                target_height=args.target_height,
                add_separator=args.add_separator,
                separator_width=args.separator_width,
            )
            out_path = out_cls / name
            cv2.imwrite(str(out_path), merged)
            total += 1

        print(f"[OK] class={cls} merged={len(common)}")

    print(f"[DONE] total merged images: {total}")


if __name__ == "__main__":
    main()

