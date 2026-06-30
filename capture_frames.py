"""Capture one frame from each robot camera and save as JPEG.

Run this (no robot arm control, no DDS) to see exactly what the policy server
sees during inference — compare against training frames from HuggingFace to
diagnose camera-setup mismatch.

Usage:
    python capture_frames.py
    python capture_frames.py --image-server 192.168.123.164 --out-dir frames/
"""

import argparse
import os
import cv2
import numpy as np
from g1_client.camera_client import CameraClient

KEYS = [
    "observation.images.cam_left_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]
SAVE_NAMES = {
    "observation.images.cam_left_high":   "cam_left_high.jpg",
    "observation.images.cam_left_wrist":  "cam_left_wrist.jpg",
    "observation.images.cam_right_wrist": "cam_right_wrist.jpg",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image-server", default="192.168.123.164",
                   help="G1 PC2 image-server host (default 192.168.123.164)")
    p.add_argument("--out-dir", default=".",
                   help="Directory to save captured frames (default: current dir)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Connecting to image server at {args.image_server} …")
    cam = CameraClient(host=args.image_server)

    print("Capturing frames …")
    imgs = cam.get_obs_images()
    cam.close()

    for key in KEYS:
        jpeg_bytes = imgs[key]
        # Decode from JPEG (comes out BGR from cv2) then convert to RGB for display.
        bgr = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        path = os.path.join(args.out_dir, SAVE_NAMES[key])
        # Save as RGB JPEG so image viewers and HuggingFace dataset viewer show
        # the same colours — lets you do a direct visual comparison.
        cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        h, w = bgr.shape[:2]
        print(f"  {SAVE_NAMES[key]}  {w}x{h}  ({len(jpeg_bytes)//1024} KiB)  → {path}")

    print(f"\nDone. Compare these against frames from the training dataset:")
    print("  https://huggingface.co/datasets/yigao7117/pick_red_bottle")
    print("\nLook for differences in:")
    print("  • Camera angle / tilt (the arm should appear at the same part of the frame)")
    print("  • Camera height above the table / gripper relative to scene")
    print("  • Horizontal position — arm centered or offset?")
    print("  • Background (plain vs cluttered)")


if __name__ == "__main__":
    main()
