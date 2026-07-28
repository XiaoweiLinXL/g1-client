"""Validate a GR00T checkpoint against training dataset ground truth.

Reads the locally-cached LeRobot dataset (parquet + mp4 files) directly,
bypassing the `datasets` library's video-decoding pipeline which requires
torchcodec. Only pyarrow and cv2 are needed.

Cache location: ~/.cache/huggingface/lerobot/<repo-id>/

Low L2 error on training samples → model learned the data correctly.
High L2 error → config mismatch (wrong obs keys, norm stats, camera format, etc.).

Usage:
    python3 test_groot_on_dataset.py \\
        --repo-id XiaoweiLinXL/unitree_load_bottle_water \\
        --server-host a10-pi05-embodyx.southcentralus.cloudapp.azure.com \\
        --server-port 5555 \\
        --num-samples 10
"""

import argparse
import glob
import os

import cv2
import numpy as np
import pyarrow.parquet as pq

from g1_client.groot_policy import GR00TPolicy

_HF_LEROBOT_CACHE = os.path.expanduser("~/.cache/huggingface/lerobot")

_CAM_NAMES = [
    "observation.images.cam_left_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]
# Keys expected by groot_policy.GR00TPolicy.infer()
_OBS_KEYS = [
    "observation.images.cam_left_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]


def _cache_dir(repo_id: str) -> str:
    return os.path.join(_HF_LEROBOT_CACHE, repo_id)


def _video_path(cache: str, cam: str, episode_index: int) -> str:
    # videos/chunk-000/<cam>/episode_NNNNNN.mp4
    # chunks index by episode — just glob for the right episode
    pattern = os.path.join(cache, "videos", "chunk-*", cam, f"episode_{episode_index:06d}.mp4")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No video found for {cam} episode {episode_index}: {pattern}")
    return matches[0]


def _read_frame(video_file: str, timestamp: float) -> np.ndarray:
    """Extract one frame from an mp4 at the given timestamp (seconds), return HxWx3 BGR uint8.
    Uses PyAV (ffmpeg) so AV1-encoded videos work."""
    import av
    with av.open(video_file) as container:
        stream = container.streams.video[0]
        # Seek to just before the target timestamp
        container.seek(int(timestamp * av.time_base ** -1), stream=stream)
        for frame in container.decode(stream):
            t = float(frame.pts * stream.time_base)
            if t >= timestamp - 0.5 / (stream.average_rate or 30):
                rgb = frame.to_ndarray(format="rgb24")
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    raise RuntimeError(f"No frame found at t={timestamp:.3f}s in {video_file}")


def _bgr_to_jpeg(bgr: np.ndarray) -> bytes:
    _, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return enc.tobytes()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True, help="HuggingFace / LeRobot dataset repo ID")
    p.add_argument("--server-host", required=True)
    p.add_argument("--server-port", type=int, default=5555)
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--sample-stride", type=int, default=200,
                   help="Step between sampled frame indices")
    p.add_argument("--cache-dir", default=_HF_LEROBOT_CACHE,
                   help="Root of the LeRobot HF cache (default: ~/.cache/huggingface/lerobot)")
    args = p.parse_args()

    cache = os.path.join(args.cache_dir, args.repo_id)
    if not os.path.isdir(cache):
        print(f"ERROR: cache not found at {cache}")
        print("Run `python3 test_groot_on_dataset.py` once to trigger the download,")
        print("or check that --repo-id matches the cached directory name.")
        return

    # Load all parquet files
    pq_files = sorted(glob.glob(os.path.join(cache, "data", "chunk-*", "*.parquet")))
    if not pq_files:
        print(f"No parquet files found under {cache}/data/")
        return
    table = pq.read_table(pq_files)
    rows = table.to_pydict()
    n = len(rows["index"])
    print(f"Dataset: {args.repo_id}")
    print(f"  {n} frames, {len(pq_files)} parquet files")
    print(f"  state dim: {len(rows['observation.state'][0])}  action dim: {len(rows['action'][0])}\n")

    # Read task description from metadata if available
    task = "load the water bottle"
    meta_path = os.path.join(cache, "meta", "tasks.jsonl")
    if os.path.exists(meta_path):
        import json
        with open(meta_path) as f:
            for line in f:
                obj = json.loads(line)
                task = obj.get("task", task)
                break
    print(f"  task: {task!r}\n")

    policy = GR00TPolicy(host=args.server_host, port=args.server_port)

    arm_errors, grip_errors, waist_errors = [], [], []
    indices = list(range(0, n, args.sample_stride))[: args.num_samples]

    for idx in indices:
        ep_idx   = int(rows["episode_index"][idx])
        ts       = float(rows["timestamp"][idx])
        state    = np.asarray(rows["observation.state"][idx], dtype=np.float32)  # (17,)
        gt       = np.asarray(rows["action"][idx], dtype=np.float32)             # (17,)

        # Load images for this frame
        images = {}
        missing = []
        for cam in _CAM_NAMES:
            try:
                vpath = _video_path(cache, cam, ep_idx)
                bgr = _read_frame(vpath, ts)
                images[cam] = _bgr_to_jpeg(bgr)
            except (FileNotFoundError, RuntimeError) as e:
                missing.append(cam)
                images[cam] = _bgr_to_jpeg(np.zeros((256, 320, 3), dtype=np.uint8))
                print(f"  WARNING [{idx}] {e}")

        obs = {
            "observation.images.cam_left_high":   images[_CAM_NAMES[0]],
            "observation.images.cam_left_wrist":  images[_CAM_NAMES[1]],
            "observation.images.cam_right_wrist": images[_CAM_NAMES[2]],
            "observation.state": state,
            "prompt": task,
        }

        result = policy.infer(obs)
        pred = np.asarray(result["actions"])[0]  # (17,) absolute step 0

        e_arm   = float(np.linalg.norm(pred[:14]  - gt[:14]))
        e_grip  = float(np.linalg.norm(pred[14:16] - gt[14:16]))
        e_waist = float(abs(pred[16] - gt[16]))
        arm_errors.append(e_arm)
        grip_errors.append(e_grip)
        waist_errors.append(e_waist)

        fmt = lambda a: " ".join(f"{x:+.3f}" for x in np.asarray(a).tolist())
        print(f"[frame {idx:5d}  ep={ep_idx:03d}  t={ts:.2f}s]  arm_L2={e_arm:.4f}  grip_L2={e_grip:.4f}  waist_err={e_waist:.4f}")
        print(f"  pred arm : {fmt(pred[:14])}")
        print(f"  gt   arm : {fmt(gt[:14])}")
        print(f"  pred grip: L={pred[14]:+.3f} R={pred[15]:+.3f}  gt: L={gt[14]:+.3f} R={gt[15]:+.3f}")
        print(f"  pred waist: {pred[16]:+.4f}  gt waist: {gt[16]:+.4f}")
        print()

    print("── summary ─────────────────────────────────────────")
    print(f"  samples tested:        {len(arm_errors)}")
    print(f"  mean arm L2 error:     {np.mean(arm_errors):.4f} rad")
    print(f"  mean gripper L2 error: {np.mean(grip_errors):.4f} rad")
    print(f"  mean waist abs error:  {np.mean(waist_errors):.4f} rad")

    if arm_errors:
        mean_arm = np.mean(arm_errors)
        if mean_arm < 0.1:
            print("\n  ✓ Low error — model learned the training data well.")
        elif mean_arm < 0.5:
            print("\n  ~ Moderate error — model partially learned, or obs format mismatch.")
        else:
            print("\n  ✗ High error — likely obs format mismatch or undertrained checkpoint.")


if __name__ == "__main__":
    main()
