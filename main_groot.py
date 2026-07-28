"""Inference loop for the G1 driving an Isaac-GR00T policy server.

WHAT'S DIFFERENT vs main_openpi_sync.py
----------------------------------------
Transport:     ZMQ REQ/REP (port 5555) instead of WebSocket
Protocol:      GR00T nested obs / split-action dict (converted inside GR00TPolicy)
Action rep:    all channels are ABSOLUTE targets (the server's StateActionProcessor
               converts relative training deltas → absolute using the obs-time state
               we send); grippers are also ABSOLUTE.
Loop style:    synchronous (infer, then execute full horizon, then repeat)

Robot init / standby / kp-switch / cleanup sequence is identical to the openpi paths.

OBSERVATION / ACTION CONTRACT (must match your g1_config.py in Isaac-GR00T)
-----------------------------------------------------------------------------
obs (built per chunk):
    observation.images.cam_left_high    JPEG bytes (BGR q90), as from CameraClient
    observation.images.cam_left_wrist   JPEG bytes (BGR q90)
    observation.images.cam_right_wrist  JPEG bytes (BGR q90)
    observation.state                   float32 (17,) = [14 arm q | L grip | R grip | waist_yaw]
    prompt                              str

action returned by GR00TPolicy.infer(): ndarray [H, 17]
    [:, 0:7]   left  arm joint targets  (ABSOLUTE, rad)
    [:, 7:14]  right arm joint targets  (ABSOLUTE, rad)
    [:, 14]    left  gripper target     (ABSOLUTE, rad, [GRIPPER_MIN, GRIPPER_MAX])
    [:, 15]    right gripper target     (ABSOLUTE, rad)
    [:, 16]    waist yaw target         (ABSOLUTE, rad)

Precondition: robot already in 'ai' motion mode (set via the Unitree app).

This branch targets the put-away-tools ABSOLUTE, no-waist checkpoint
(XiaoweiLinXL/groot-unitree-load-bottle-water-20k — the repo name is legacy; it holds the
put-away-tools model). That model has NO waist joint, so always pass --no-waist, and the
16-dim state/action layout drops the trailing waist column.

Usage (put-away-tools, no-waist) — from the repo root:
  python main_groot.py \\
      --iface enp0s31f6 \\
      --server-host 127.0.0.1 --server-port 5555 \\
      --no-waist \\
      --prompt "put the battery into the battery bin and the screw driver into the philips bin"

Test mode (no robot required — tests server connectivity through the SSH tunnel):
  python main_groot.py \\
      --test \\
      --server-host 127.0.0.1 --server-port 5555 \\
      --no-waist --dry-run

The server runs on the Azure A10 and is reached over an SSH tunnel from the laptop:
  ssh -i a10-1.5-inference-fabricio_key.pem -N \\
      -L 5555:localhost:5555 \\
      fabricio@a10-pi05-embodyx.southcentralus.cloudapp.azure.com
so --server-host is 127.0.0.1 (the local end of the tunnel).

NOTE: the legacy bottle-water checkpoint used a waist joint (17-dim) and the default
prompt "Load the bottle water to the shelf"; that path still works if you drop --no-waist
and pass the matching --prompt.
"""

import argparse
import logging
import time

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("g1_groot.main")

# Gripper limits — overridden by real hardware imports in _run_real().
GRIPPER_MIN = 0.0
GRIPPER_MAX = 1.0

ARM_JOINT_NAMES = [
    "L_pitch", "L_roll ", "L_yaw ", "L_elbow", "L_wrR ", "L_wrP ", "L_wrY ",
    "R_pitch", "R_roll ", "R_yaw ", "R_elbow", "R_wrR ", "R_wrP ", "R_wrY ",
]

# Init pose for GR00T tasks — mean of episode-start states across all 100 training
# episodes of put-away-tools-v2_new_cam (observation.state[:14]).  Distinct from
# INIT_POSE_READY (arms-at-sides neutral) which the model never saw at the beginning
# of a task.  Recomputed for the put-away-tools (absolute, no-waist) checkpoint.
GROOT_INIT_POSE = np.array([
    +0.064,  0.008,  0.009, -0.208, -0.072, -0.014, -0.035,   # left
    +0.044, -0.012,  0.045, -0.204, -0.013, -0.021,  0.017,   # right
], dtype=np.float64)

_FAKE_IMG_H, _FAKE_IMG_W = 480, 640
_CAM_KEYS = [
    "observation.images.cam_left_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]


# ---------- fake hardware stubs (--test mode only) ----------

def _make_fake_jpeg() -> bytes:
    bgr = np.random.randint(0, 256, (_FAKE_IMG_H, _FAKE_IMG_W, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("cv2.imencode failed in fake camera")
    return buf.tobytes()


class _FakeCamera:
    def get_obs_images(self) -> dict:
        return {k: _make_fake_jpeg() for k in _CAM_KEYS}

    def close(self):
        pass


class _FakeArm:
    def __init__(self):
        self._q = np.zeros(14, dtype=np.float32)

    def start(self):
        log.info("[TEST] FakeArm started")

    def stop(self):
        pass

    def disable_arm_sdk(self):
        pass

    def faulted(self) -> bool:
        return False

    def get_arm_q(self) -> np.ndarray:
        return self._q.copy()

    def set_arm_target(self, q):
        self._q = np.asarray(q, dtype=np.float32)

    def get_waist_yaw(self) -> float:
        return 0.0

    def set_waist_yaw_target(self, q):
        pass

    def set_arm_kp(self, kp):
        log.info(f"[TEST] FakeArm.set_arm_kp({kp}) — no-op")

    def move_to_pose(self, pose, duration=2.0, velocity_limit=8.0):
        log.info(f"[TEST] FakeArm.move_to_pose over {duration:.1f}s — no-op")
        time.sleep(min(duration, 0.1))


class _FakeGripper:
    _GRIPPER_MIN = 0.0
    _GRIPPER_MAX = 1.0

    def __init__(self):
        self._left = 0.0
        self._right = 0.0

    def start(self):
        log.info("[TEST] FakeGripper started")

    def stop(self):
        pass

    def get_state(self):
        return self._left, self._right

    def set_targets(self, left, right):
        self._left = left
        self._right = right

    def move_to_targets(self, left, right, duration=0.5):
        log.info(f"[TEST] FakeGripper.move_to_targets({left}, {right}) over {duration:.1f}s — no-op")
        time.sleep(min(duration, 0.05))


# ---------- observation assembly ----------

def build_obs(cam, arm, grip, prompt: str) -> dict:
    """Assemble one GR00T observation. Camera frames stay as JPEG bytes;
    GR00TPolicy.infer() decodes them and reformats to the nested GR00T dict."""
    imgs = cam.get_obs_images()
    left_q, right_q = grip.get_state()
    arm_q = arm.get_arm_q()  # (14,)
    waist_yaw = arm.get_waist_yaw()  # actual WaistYaw from lowstate
    state = np.concatenate([arm_q, [left_q, right_q, waist_yaw]]).astype(np.float32)  # (17,)
    return {**imgs, "observation.state": state, "prompt": prompt}


def log_chunk_ranges(chunk_id: int, deltas: np.ndarray) -> None:
    """One-line per-joint range: print the delta magnitudes before dispatching."""
    arm = deltas[:, :14]
    gl = deltas[:, 14]
    gr = deltas[:, 15]
    log.info(f"[chunk {chunk_id}] H={deltas.shape[0]} arm targets + gripper targets (rad):")
    header = f"    {'step':>4}  " + "  ".join(f"{n:>8}" for n in ARM_JOINT_NAMES) + "  L_grip   R_grip"
    log.info(header)
    for t in range(deltas.shape[0]):
        row = "  ".join(f"{arm[t, i]:+8.4f}" for i in range(14))
        log.info(f"    {t:>4}  {row}  {gl[t]:+7.4f}  {gr[t]:+7.4f}")


# ---------- inference loop ----------

# Safety clamp applied to predicted arm targets before dispatch. These MUST cover the
# joint range the policy was trained on, otherwise legitimate motion is truncated. The
# put-away-tools-v2_new_cam training data reaches elbow ~-0.95 rad (state idx 3 = L_elbow
# [-0.937, 0.830], idx 10 = R_elbow [-0.959, 0.767]) and wrist-pitch ~1.57, so the elbow
# mins were relaxed from -0.5 -> -1.1 and wrist-pitch maxes from 1.5 -> 1.7. The old -0.5
# elbow floor clipped the reach-into-bin motion and stalled the place after grasping.
_ARM_JOINT_MIN = np.array([-2.8,-0.4,-2.4,-1.1,-1.9,-1.5,-1.5,-2.8,-2.2,-2.4,-1.1,-1.9,-1.5,-1.5])
_ARM_JOINT_MAX = np.array([ 1.4, 2.2, 2.4, 2.9, 1.9, 1.7, 1.5, 1.4, 0.4, 2.4, 2.9, 1.9, 1.7, 1.5])


def _run_inference_loop(arm, grip, cam, policy, args) -> None:
    """Synchronous receding-horizon loop: infer chunk, execute, repeat.

    Arm actions are ABSOLUTE joint targets. The server's StateActionProcessor already
    converts training-time relative deltas → absolute using the obs-time state we send.
    """
    dt = 1.0 / args.control_hz
    prompt = args.prompt

    for c in range(args.max_chunks):
        log.info(f"[chunk {c}] Capturing obs and inferring...")
        obs = build_obs(cam, arm, grip, prompt)
        current_arm_q = obs["observation.state"][:14].copy()

        t_infer = time.time()
        result = policy.infer(obs)
        infer_ms = (time.time() - t_infer) * 1e3
        actions = np.asarray(result["actions"], dtype=np.float64)  # [H, 17]
        if actions.ndim != 2 or actions.shape[1] < 16:
            raise RuntimeError(f"Unexpected action shape {actions.shape} (want [H, >=16])")
        log.info(f"[chunk {c}] infer={infer_ms:.0f}ms  H={actions.shape[0]}")

        # Log current arm state vs step-0 prediction for sanity checking.
        header = "  " + "  ".join(f"{n:>8}" for n in ARM_JOINT_NAMES)
        cur_row = "  ".join(f"{current_arm_q[i]:+8.4f}" for i in range(14))
        act_row = "  ".join(f"{actions[0, i]:+8.4f}" for i in range(14))
        dlt_row = "  ".join(f"{actions[0, i] - current_arm_q[i]:+8.4f}" for i in range(14))
        log.info(f"[chunk {c}] obs  arm_q : {header}")
        log.info(f"[chunk {c}]   current  : {cur_row}")
        log.info(f"[chunk {c}]   action[0]: {act_row}")
        log.info(f"[chunk {c}]   delta[0] : {dlt_row}  (action[0] - current)")
        log.info(f"[chunk {c}]   grippers : L={actions[0,14]:+.4f}  R={actions[0,15]:+.4f}")
        waist_obs = obs['observation.state'][16]
        waist_act = actions[0, 16] if actions.shape[1] > 16 else float('nan')
        log.info(f"[chunk {c}]   waist_yaw sent: {waist_obs:+.4f} rad  action[0]: {waist_act:+.4f} rad")

        # Warn if any step exceeds joint limits (would be clipped by arm controller).
        out_of_range = np.any(
            (actions[:, :14] < _ARM_JOINT_MIN[None]) | (actions[:, :14] > _ARM_JOINT_MAX[None])
        )
        if out_of_range:
            log.warning(f"[chunk {c}] Some predicted arm targets exceed joint limits "
                        f"— will be clipped by arm_controller!")
        log_chunk_ranges(c, actions)

        if args.dry_run:
            log.info("[dry-run] Exiting after first inference — no motion commanded.")
            return

        H = actions.shape[0]
        n = H if args.exec_steps <= 0 else min(args.exec_steps, H)
        for t in range(n):
            if arm.faulted():
                raise RuntimeError("ArmController control thread faulted — aborting")
            tic = time.time()

            arm.set_arm_target(actions[t, :14])
            if actions.shape[1] > 16:
                arm.set_waist_yaw_target(float(actions[t, 16]))

            grip.set_targets(
                float(np.clip(actions[t, 14], GRIPPER_MIN, GRIPPER_MAX)),
                float(np.clip(actions[t, 15], GRIPPER_MIN, GRIPPER_MAX)),
            )

            sleep = dt - (time.time() - tic)
            if sleep > 0:
                time.sleep(sleep)


# ---------- pipeline stages ----------

def _initialize_pose(arm, grip, args, init_pose) -> None:
    log.info(f"Moving arms to ready pose over {args.init_duration:.1f}s "
             f"(velocity_limit={args.velocity_limit} rad/s)")
    arm.move_to_pose(init_pose, duration=args.init_duration,
                     velocity_limit=args.velocity_limit)
    half = args.gripper_init_duration / 2
    log.info(f"Closing grippers to {GRIPPER_MIN} over {half:.1f}s")
    grip.move_to_targets(GRIPPER_MIN, GRIPPER_MIN, duration=half)
    log.info(f"Opening grippers to ({args.init_gripper_left}, {args.init_gripper_right}) "
             f"over {half:.1f}s")
    grip.move_to_targets(args.init_gripper_left, args.init_gripper_right, duration=half)
    log.info("Init complete.")
    time.sleep(args.settle_duration)
    log.info("Arms settled at ready pose.")


def _wait_for_operator(args) -> None:
    if args.auto_start:
        return
    log.info("===============================================================")
    log.info("STANDBY: arms locked at ready pose.")
    log.info("Set up the scene, then press [Enter] to connect and start.")
    log.info("Press [Ctrl+C] at any time to abort safely.")
    log.info("===============================================================")
    try:
        input("")
    except EOFError:
        log.info("EOF on stdin — proceeding without prompt")


def _cleanup(arm, grip, cam, policy) -> None:
    log.info("Shutting down — releasing arm_sdk")
    try:
        arm.stop()
    except BaseException as e:
        log.warning(f"arm.stop() failed: {e}")
    try:
        arm.disable_arm_sdk()
    except BaseException as e:
        log.warning(f"disable_arm_sdk failed: {e}")
    if grip is not None:
        try:
            grip.stop()
        except BaseException as e:
            log.warning(f"grip.stop() failed: {e}")
    if cam is not None:
        try:
            cam.close()
        except BaseException as e:
            log.warning(f"cam.close() failed: {e}")
    if policy is not None:
        try:
            policy.close()
        except BaseException as e:
            log.warning(f"policy.close() failed: {e}")


def run(args) -> None:
    if args.test:
        _run_test(args)
    else:
        _run_real(args)


def _run_test(args) -> None:
    log.info("*** TEST MODE — no robot hardware will be used ***")
    arm = _FakeArm()
    arm.start()
    grip = _FakeGripper()
    grip.start()
    cam = _FakeCamera()
    policy = None
    from g1_client.groot_policy import GR00TPolicy
    try:
        log.info(f"Connecting to GR00T server {args.server_host}:{args.server_port}")
        policy = GR00TPolicy(host=args.server_host, port=args.server_port,
                             use_waist=not args.no_waist,
                             image_hw=tuple(args.image_hw))
        log.info("Pinging server...")
        ok = policy.ping()
        log.info(f"Ping: {'OK' if ok else 'no response (continuing anyway)'}")
        _run_inference_loop(arm, grip, cam, policy, args)
    finally:
        _cleanup(arm, grip, cam, policy)


def _run_real(args) -> None:
    global GRIPPER_MIN, GRIPPER_MAX
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from g1_client.arm_controller import ArmController
    from g1_client.gripper_controller import GripperController
    from g1_client.gripper_controller import GRIPPER_MIN as _GMIN, GRIPPER_MAX as _GMAX
    from g1_client.camera_client import CameraClient
    from g1_client.groot_policy import GR00TPolicy
    GRIPPER_MIN = _GMIN
    GRIPPER_MAX = _GMAX

    log.info(f"Initializing DDS on {args.iface}")
    ChannelFactoryInitialize(0, args.iface)

    arm = ArmController(publish_hz=50.0, velocity_limit=args.velocity_limit)
    arm.start()
    grip = None
    cam = None
    policy = None
    try:
        grip = GripperController(publish_hz=200.0)
        grip.start()
        cam = CameraClient(host=args.image_server)
        _initialize_pose(arm, grip, args, GROOT_INIT_POSE)
        _wait_for_operator(args)
        log.info(f"Switching arm kp to inference value: {args.inference_kp_arm}")
        arm.set_arm_kp(args.inference_kp_arm)
        log.info(f"Connecting to GR00T server {args.server_host}:{args.server_port}")
        policy = GR00TPolicy(host=args.server_host, port=args.server_port,
                             use_waist=not args.no_waist,
                             image_hw=tuple(args.image_hw))
        log.info("Pinging server...")
        ok = policy.ping()
        log.info(f"Ping: {'OK' if ok else 'no response (continuing anyway)'}")
        _run_inference_loop(arm, grip, cam, policy, args)
        _initialize_pose(arm, grip, args, GROOT_INIT_POSE)
    finally:
        _cleanup(arm, grip, cam, policy)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--test", action="store_true",
                   help="Use fake camera/arm/gripper stubs instead of real hardware. "
                        "Tests the full inference loop against a live GR00T server. "
                        "--iface is not required in this mode.")
    p.add_argument("--iface", default=None,
                   help="Network interface to robot, e.g. enp0s31f6 (required without --test)")
    p.add_argument("--server-host", required=True,
                   help="GR00T server host or IP (Azure VM hostname)")
    p.add_argument("--server-port", type=int, default=5555,
                   help="GR00T ZMQ server port (default 5555)")
    p.add_argument("--image-server", default="192.168.123.164",
                   help="G1 PC2 image-server host (default 192.168.123.164)")
    p.add_argument("--prompt",
                   default="put the battery into the battery bin and the screw driver into the philips bin",
                   help="Language instruction for the policy. Must match the training task string "
                        "exactly (meta/tasks.jsonl of the checkpoint's dataset).")
    p.add_argument("--max-chunks", type=int, default=300,
                   help="Number of action chunks to run before stopping")
    p.add_argument("--control-hz", type=float, default=30.0,
                   help="Per-step dispatch rate; must match training fps (default 30)")
    p.add_argument("--exec-steps", type=int, default=0,
                   help="Steps to execute per chunk; 0 = full action horizon (default 0)")
    # ---- safety / motion limits (same as openpi paths) ----
    p.add_argument("--velocity-limit", type=float, default=8.0)
    p.add_argument("--inference-kp-arm", type=float, default=80.0)
    p.add_argument("--init-duration", type=float, default=2.0)
    p.add_argument("--gripper-init-duration", type=float, default=1.0)
    p.add_argument("--settle-duration", type=float, default=1.0)
    p.add_argument("--init-gripper-left", type=float, default=5.0)
    p.add_argument("--init-gripper-right", type=float, default=5.0)
    p.add_argument("--no-waist", action="store_true",
                   help="Omit waist joint from state sent to server and do not command "
                        "waist from actions. Use with checkpoints trained without waist.")
    p.add_argument("--image-hw", type=int, nargs=2, default=[256, 342],
                   metavar=("H", "W"),
                   help="Image size (H W) to resize frames to before sending to server. "
                        "Default 256 342 preserves the 480x640 (3:4) training aspect ratio of "
                        "put-away-tools-v2_new_cam (letterbox padding is off in the processor, so "
                        "keep the training aspect). The processor resizes to the model input "
                        "internally; smaller = less tunnel bandwidth.")
    p.add_argument("--auto-start", action="store_true",
                   help="Skip the post-init Enter prompt and start immediately.")
    p.add_argument("--dry-run", action="store_true",
                   help="Run one inference step, log the predicted actions vs current "
                        "state, then exit without commanding any motion. Useful for "
                        "sanity-checking the server output.")
    args = p.parse_args()

    if not args.test and args.iface is None:
        p.error("--iface is required without --test")

    run(args)


if __name__ == "__main__":
    main()
