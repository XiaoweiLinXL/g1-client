"""Validate a deployed checkpoint against training dataset ground truth.

Loads samples from a LeRobot dataset, sends them to the running policy server,
and compares the first predicted action against the ground truth action.

A low L2 error on training samples confirms the model learned the data.
A high error suggests a config mismatch (wrong norm stats, wrong obs keys, etc).

Usage:
    python test_checkpoint_on_dataset.py \
        --repo-id XiaoweiLinXL/pi05-unitree-g1-put-away-tools-v2.1 \
        --server-host localhost \
        --server-port 8000 \
        --num-samples 10
"""

import argparse
import os
import numpy as np
import torch
import cv2
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
from g1_client.policy_client import PolicyClient


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True, help="HuggingFace dataset repo ID")
    p.add_argument("--server-host", default="localhost")
    p.add_argument("--server-port", type=int, default=8000)
    p.add_argument("--num-samples", type=int, default=10,
                   help="Number of samples to test")
    p.add_argument("--sample-stride", type=int, default=200,
                   help="Step between sampled frame indices (spread across dataset)")
    p.add_argument("--save-images", default=None, metavar="DIR",
                   help="Save camera frames for each sample to this directory")
    return p.parse_args()


def to_rgb_uint8(frame) -> np.ndarray:
    """Convert a lerobot video frame to HWC uint8 RGB numpy array."""
    if isinstance(frame, torch.Tensor):
        frame = frame.numpy()
    frame = np.asarray(frame)
    if np.issubdtype(frame.dtype, np.floating):
        frame = (frame * 255).clip(0, 255).astype(np.uint8)
    if frame.ndim == 3 and frame.shape[0] == 3:  # CHW → HWC
        frame = frame.transpose(1, 2, 0)
    return frame


def main():
    args = parse_args()

    print(f"Loading dataset metadata: {args.repo_id}")
    meta = lerobot_dataset.LeRobotDatasetMetadata(args.repo_id, revision="main")
    ds = lerobot_dataset.LeRobotDataset(args.repo_id)
    print(f"  {len(ds)} frames, {len(meta.tasks)} task(s): {meta.tasks}\n")

    policy = PolicyClient(host=args.server_host, port=args.server_port)
    print(f"Connected to {args.server_host}:{args.server_port}\n")

    arm_errors, grip_errors = [], []
    indices = list(range(0, len(ds), args.sample_stride))[: args.num_samples]

    for idx in indices:
        sample = ds[idx]

        task_index = int(np.asarray(sample.get("task_index", [0])).flat[0])
        prompt = meta.tasks.get(task_index, "")

        obs = {
            "observation/image":             to_rgb_uint8(sample["observation.images.cam_left_high"]),
            "observation/left_wrist_image":  to_rgb_uint8(sample["observation.images.cam_left_wrist"]),
            "observation/right_wrist_image": to_rgb_uint8(sample["observation.images.cam_right_wrist"]),
            "observation/state":             np.asarray(sample["observation.state"], dtype=np.float32),
            "prompt":                        prompt,
        }

        result = policy.infer(obs)
        pred = np.asarray(result["actions"])       # [H, 16]
        gt   = np.asarray(sample["action"], dtype=np.float32)  # [16]

        e_arm  = float(np.linalg.norm(pred[0, :14] - gt[:14]))
        e_grip = float(np.linalg.norm(pred[0, 14:] - gt[14:]))
        arm_errors.append(e_arm)
        grip_errors.append(e_grip)

        fmt = lambda arr: [f"{x:.3f}" for x in arr.tolist()]
        print(f"[frame {idx:5d}]  arm L2={e_arm:.4f}  gripper L2={e_grip:.4f}")
        print(f"  pred arm:  {fmt(pred[0, :14])}")
        print(f"  gt   arm:  {fmt(gt[:14])}")
        print(f"  pred grip: {fmt(pred[0, 14:])}  gt grip: {fmt(gt[14:])}")

        if args.save_images:
            os.makedirs(args.save_images, exist_ok=True)
            frames = {
                "left_high":   obs["observation/image"],
                "left_wrist":  obs["observation/left_wrist_image"],
                "right_wrist": obs["observation/right_wrist_image"],
            }
            # Stack side by side into one composite image
            composite = np.concatenate([
                cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames.values()
            ], axis=1)
            path = os.path.join(args.save_images, f"frame_{idx:05d}.jpg")
            cv2.imwrite(path, composite)
            print(f"  saved → {path}")

    print("\n── summary ─────────────────────────────────────────")
    print(f"  samples tested:        {len(arm_errors)}")
    print(f"  mean arm L2 error:     {np.mean(arm_errors):.4f}  (rad)")
    print(f"  mean gripper L2 error: {np.mean(grip_errors):.4f}  (rad)")
    print(f"  max  arm L2 error:     {np.max(arm_errors):.4f}")


if __name__ == "__main__":
    main()
