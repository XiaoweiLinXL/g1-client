"""Main inference loop for the G1, talking to a standard openpi serve_policy.py
server instead of the LingBot-VA cloud server.

WHAT CHANGED vs main.py
-----------------------
Nothing in the robot control path. arm_controller.py, gripper_controller.py and
camera_client.py are used verbatim. Only the data send/receive changes:

  * PolicyClient (LingBot, stateful reset/cold_start/async_step, [16,F,S])
        --> OpenPIPolicy (stateless infer(obs) -> {"actions": [H,16]})
  * the two-chunk FDM-grounded overlap loop (Algorithm 2)
        --> a plain receding-horizon loop: infer a chunk, dispatch it at a
            fixed rate, prefetch the next chunk on a daemon thread so the
            chunk boundary doesn't stall (same idea as the UR3 PrefetchBroker).

The robot init / standby / kp-switch / cleanup sequence is preserved exactly,
because those are safety-critical and have nothing to do with the server.

OBSERVATION / ACTION CONTRACT (must match your G1 openpi DataConfig)
--------------------------------------------------------------------
This is the one thing you must keep in lockstep with the server checkpoint —
the obs keys here must match the keys your RepackTransform consumes, and the
state/action dimension order must match how the LeRobot dataset was recorded.

  obs (sent every tick):
    observation.images.cam_left_high   uint8 HxWx3 RGB
    observation.images.cam_left_wrist  uint8 HxWx3 RGB
    observation.images.cam_right_wrist uint8 HxWx3 RGB
    observation.state                  float32 (16,) = [14 arm q | L grip | R grip]
    prompt                             str

  action returned: ndarray [H, 16]
    [:, 0:14] absolute arm joint targets (rad), order == ARM_JOINTS
    [:, 14]   left gripper  (rad, [GRIPPER_MIN, GRIPPER_MAX])
    [:, 15]   right gripper (rad, [GRIPPER_MIN, GRIPPER_MAX])

If your dataset stored the gripper normalized (e.g. 0..1) instead of rad, scale
[:, 14:16] here before set_targets — that is the only spot likely to need a
tweak, and it is data handling, not control code.

Precondition: robot already in 'ai' motion mode (set via the Unitree app).

Usage (run from the lingbot_g1_client/ directory):
  python main_openpi.py \\
      --iface enp0s31f6 \\
      --server-host 1.2.3.4 \\
      --server-port 8000 \\
      --prompt "pick up the pink object and place it on the blue cross mark"

Test mode (no robot required):
  python main_openpi.py \\
      --test \\
      --server-host 1.2.3.4 \\
      --server-port 8000 \\
      --prompt "pick up the pink object"
"""

import argparse
import logging
import threading
import time

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("g1_openpi.main")

# Gripper limits — overridden by real hardware imports in _run_real().
GRIPPER_MIN = 0.0
GRIPPER_MAX = 1.0

# Action tensor layout for the G1 openpi checkpoint: [H, 16].
ARM_CHANNELS = slice(0, 14)
LEFT_GRIPPER_CHANNEL = 14
RIGHT_GRIPPER_CHANNEL = 15

ARM_JOINT_NAMES = [
    "L_pitch", "L_roll ", "L_yaw ", "L_elbow", "L_wrR ", "L_wrP ", "L_wrY ",
    "R_pitch", "R_roll ", "R_yaw ", "R_elbow", "R_wrR ", "R_wrP ", "R_wrY ",
]

# Fake image resolution matches the real cameras.
_FAKE_IMG_H, _FAKE_IMG_W = 480, 640
_FAKE_CAM_KEYS = [
    "observation.images.cam_left_high",   # mapped to observation/cam_left_high in build_obs
    "observation.images.cam_left_wrist",  # mapped to observation/cam_left_wrist
    "observation.images.cam_right_wrist", # mapped to observation/cam_right_wrist
]


# ---------- fake hardware stubs (--test mode only) ----------

def _make_fake_jpeg() -> bytes:
    """Encode a random RGB image as JPEG bytes (same format as CameraClient)."""
    bgr = np.random.randint(0, 256, (_FAKE_IMG_H, _FAKE_IMG_W, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("cv2.imencode failed in fake camera")
    return buf.tobytes()


class _FakeCamera:
    def get_obs_images(self) -> dict:
        return {k: _make_fake_jpeg() for k in _FAKE_CAM_KEYS}

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


# ---------- observation assembly (the "receive" side) ----------

def _jpeg_to_rgb(jpeg_bytes: bytes) -> np.ndarray:
    """Decode the camera client's JPEG bytes back to an RGB uint8 array.

    camera_client.get_obs_images() returns BGR-encoded JPEG (it was built for a
    server that does its own decode). openpi expects decoded RGB arrays, so we
    decode here in the glue rather than editing camera_client.py. The q90 round
    trip is well within the model's training distribution (LeRobot mp4/H.264).
    """
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("cv2.imdecode failed on a camera frame")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def build_obs(cam, arm, grip, prompt: str, send_jpeg: bool = False) -> dict:
    """Assemble one openpi observation from the (unchanged) controllers.

    send_jpeg=False (default): images are decoded RGB uint8 arrays — the legacy
    wire format. ~720 KiB/obs of upload, which is the dominant network cost.

    send_jpeg=True: ship the camera's compressed JPEG bytes straight through
    (~60 KiB/obs, ~12x smaller upload). The SERVER must then decode AND swap
    channels, exactly what _jpeg_to_rgb does here:
        rgb = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    Get this wrong server-side and it's the silent channel-swap / garbage class
    — keep client encode and server decode in lockstep.
    """
    imgs = cam.get_obs_images()  # dict of JPEG bytes (BGR, q90), LeRobot keys
    left_q, right_q = grip.get_state()
    arm_q = arm.get_arm_q()  # (14,)
    state = np.concatenate([arm_q, [left_q, right_q]]).astype(np.float32)  # (16,)

    def _img(key):
        return imgs[key] if send_jpeg else _jpeg_to_rgb(imgs[key])
    return {
        "observation/cam_left_high":  _img("observation.images.cam_left_high"),
        "observation/cam_left_wrist": _img("observation.images.cam_left_wrist"),
        "observation/cam_right_wrist": _img("observation.images.cam_right_wrist"),
        "observation/state": state,
        "prompt": prompt,
    }


def log_chunk_ranges(chunk_id: int, actions: np.ndarray) -> None:
    """One-line per-joint range sanity print before a chunk streams to the arm."""
    arm = actions[:, ARM_CHANNELS]
    gl = actions[:, LEFT_GRIPPER_CHANNEL]
    gr = actions[:, RIGHT_GRIPPER_CHANNEL]
    log.info(f"[chunk {chunk_id}] H={actions.shape[0]} raw actions (rad):")
    header = f"    {'step':>4}  " + "  ".join(f"{n:>8}" for n in ARM_JOINT_NAMES) + "  L_grip   R_grip"
    log.info(header)
    for t in range(actions.shape[0]):
        row = "  ".join(f"{arm[t, i]:+8.4f}" for i in range(len(ARM_JOINT_NAMES)))
        log.info(f"    {t:>4}  {row}  {gl[t]:+7.4f}  {gr[t]:+7.4f}")


# ---------- inference loop (the "send" side) ----------

def _pct(xs, p):
    return float(np.percentile(xs, p)) if xs else float("nan")


def _stat(xs):
    """(min, p50, p95, max, mean) of a list of numbers."""
    return (min(xs), _pct(xs, 50), _pct(xs, 95), max(xs), float(np.mean(xs)))


def _extract_server_ms(result):
    """Server-reported inference time (ms) if present, else None — lets us split
    wait_recv into network vs on-GPU compute."""
    if not isinstance(result, dict):
        return None
    for k in ("server_timing", "policy_timing", "timing"):
        v = result.get(k)
        if isinstance(v, dict):
            for tk, scale in (("infer_ms", 1.0), ("inference_ms", 1.0),
                              ("infer_time_ms", 1.0), ("infer_s", 1e3),
                              ("inference_s", 1e3)):
                if tk in v:
                    try:
                        return float(v[tk]) * scale
                    except (TypeError, ValueError):
                        return None
    return None


def _timing_rec(tm, server_ms=None):
    """Turn PolicyClient.last_timing into a uniform per-infer record (ms)."""
    return {
        "wall_ms": tm.get("total_s", 0.0) * 1e3,
        "pack_ms": tm.get("pack_s", 0.0) * 1e3,
        "send_ms": tm.get("send_s", 0.0) * 1e3,
        "wait_recv_ms": tm.get("wait_recv_s", 0.0) * 1e3,
        "unpack_ms": tm.get("unpack_s", 0.0) * 1e3,
        "server_ms": server_ms,
        "bytes_sent": tm.get("bytes_sent", -1),
        "bytes_recv": tm.get("bytes_recv", -1),
    }


def _summarize_timing(infer_recs, chunk_recs, args):
    """Per-step latency breakdown + bottleneck + stall verdict (same stats as
    test_policy_server, plus the execute-vs-stall signal only a real run gives)."""
    if not infer_recs:
        return
    # Budget = the lead window: how long infer has before it must finish.
    budget_ms = (args.prefetch_lead / args.control_hz) * 1e3
    log.info("=" * 64)
    comps = [("pack", "pack_ms"), ("send", "send_ms"),
             ("wait_recv", "wait_recv_ms"), ("unpack", "unpack_ms"),
             ("wall(total)", "wall_ms")]
    log.info(f"per-infer latency over {len(infer_recs)} calls (ms):")
    log.info(f"  {'component':<14} {'min':>7} {'p50':>7} {'p95':>7} {'max':>7} {'mean':>7}")
    means = {}
    for label, key in comps:
        mn, p50, p95, mx, me = _stat([r[key] for r in infer_recs])
        means[label] = me
        log.info(f"  {label:<14} {mn:7.1f} {p50:7.1f} {p95:7.1f} {mx:7.1f} {me:7.1f}")
    srv = [r["server_ms"] for r in infer_recs if r["server_ms"] is not None]
    if srv:
        mn, p50, p95, mx, me = _stat(srv)
        log.info(f"  {'server_infer':<14} {mn:7.1f} {p50:7.1f} {p95:7.1f} {mx:7.1f} {me:7.1f}")
        log.info(f"  => network (wait_recv - server_infer) ~= {means['wait_recv']-me:.1f} ms "
                 f"mean (GPU compute ~= {me:.1f} ms)")
    sub = {k: means[k] for k in ("pack", "send", "wait_recv", "unpack")}
    bn = max(sub, key=sub.get)
    log.info(f"BOTTLENECK (mean): {bn} = {sub[bn]:.1f} ms "
             f"({sub[bn]/means['wall(total)']*100:.0f}% of infer total)")
    if bn in ("pack", "send"):
        up = [r["bytes_sent"] for r in infer_recs if r["bytes_sent"] > 0]
        if up:
            log.info(f"  upload payload ~= {np.mean(up)/1024:.0f} KiB/infer (decoded RGB "
                     f"frames) — sending JPEG bytes would cut pack+send ~10-15x.")
    if chunk_recs:
        ex = [r["exec_s"] for r in chunk_recs]
        jw = [r["join_wait_s"] * 1e3 for r in chunk_recs]
        stalled = sum(1 for x in jw if x > 1.0)
        log.info(f"execute/chunk: mean={np.mean(ex):.2f}s | join_wait: mean={np.mean(jw):.0f}ms "
                 f"p95={_pct(jw,95):.0f}ms | stalled {stalled}/{len(jw)} chunks "
                 f"(overlap budget {budget_ms:.0f}ms)")
        if stalled:
            log.warning(f"{stalled} chunk(s) STALLED at the boundary (infer didn't fit the "
                        f"overlap window) — that's the 顿. Raise --prefetch-lead / lower "
                        f"--control-hz, or fix the bottleneck above.")
        else:
            log.info("no chunk stalled — inference fully hidden behind execution.")
    log.info("=" * 64)


def _infer_worker(policy, obs: dict, box: dict) -> None:
    """Run one blocking infer on a daemon thread; stash result/exception/timing.

    Extracts and validates the action array (same as the first-chunk path) so
    the loop receives a [H, 16] ndarray, not the raw {"actions": ...} dict, and
    stashes the latency breakdown so the main thread can profile it post-join.
    """
    try:
        result = policy.infer(obs)
        box["timing"] = dict(policy.last_timing or {})
        box["server_ms"] = _extract_server_ms(result)
        actions = np.asarray(result["actions"], dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] < 16:
            raise RuntimeError(f"Unexpected action shape {actions.shape} (want [H, 16])")
        box["actions"] = actions
    except BaseException as e:  # surface to the main thread
        box["err"] = e

def _run_inference_loop_sync(arm, grip, cam, policy, args) -> None:
    """Synchronous loop: execute chunk fully, then fetch obs and infer next chunk."""
    dt = 1.0 / args.control_hz
    prompt = args.prompt

    for c in range(args.max_chunks):
        # Fetch obs and infer
        log.info(f"[chunk {c}] Inferring...")
        obs = build_obs(cam, arm, grip, prompt, args.send_jpeg)
        result = policy.infer(obs)
        actions = np.asarray(result["actions"], dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] < 16:
            raise RuntimeError(f"Unexpected action shape {actions.shape} (want [H, 16])")
        log_chunk_ranges(c, actions)

        H = actions.shape[0]
        n = H if args.exec_steps <= 0 else min(args.exec_steps, H)

        # Execute chunk step by step
        for i in range(n):
            if arm.faulted():
                raise RuntimeError("ArmController control thread faulted — aborting")
            tic = time.time()
            a = actions[i].astype(np.float64)
            arm.set_arm_target(a[ARM_CHANNELS])
            grip.set_targets(
                float(np.clip(a[LEFT_GRIPPER_CHANNEL], GRIPPER_MIN, GRIPPER_MAX)),
                float(np.clip(a[RIGHT_GRIPPER_CHANNEL], GRIPPER_MIN, GRIPPER_MAX)),
            )
            sleep = dt - (time.time() - tic)
            if sleep > 0:
                time.sleep(sleep)


def _run_inference_loop_async(arm, grip, cam, policy, args) -> None:
    """Overlapped receding-horizon loop — lingbot Algorithm 2 pattern adapted
    for the stateless openpi infer API.

    At the START of each chunk, snapshot a fresh obs and fire the next infer on
    a daemon thread so the full chunk execution time is the overlap budget.
    The main thread executes the current chunk step-by-step at control_hz while
    the daemon thread waits on the GPU. Join at the boundary: if infer finished
    during execution the join is instant; if not, the robot pauses briefly
    (visible as join_wait > 0 in the log).

    Timeline per chunk:
      t=0     snapshot obs, spawn infer thread
      t=0..T  execute current chunk at control_hz on main thread
      t=T     join thread (zero stall if infer < T, brief pause if infer > T)
              swap in next chunk, repeat
    """
    dt = 1.0 / args.control_hz
    prompt = args.prompt
    if args.send_jpeg:
        log.warning("--send-jpeg ON: sending compressed JPEG bytes. The SERVER must "
                    "imdecode + cv2.COLOR_BGR2RGB these keys, or it sees garbage.")

    # First chunk is a blocking infer (nothing to overlap it against yet).
    log.info(f"First inference (prompt={prompt!r})")
    result = policy.infer(build_obs(cam, arm, grip, prompt, args.send_jpeg))
    actions = np.asarray(result["actions"], dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] < 16:
        raise RuntimeError(f"Unexpected action shape {actions.shape} (want [H, 16])")
    log_chunk_ranges(0, actions)

    infer_recs = [_timing_rec(dict(policy.last_timing or {}), _extract_server_ms(result))]
    chunk_recs = []
    last_cmd = None
    blend_count = 0  # updated dynamically after each join

    for c in range(1, args.max_chunks + 1):
        H = actions.shape[0]
        n = H if args.exec_steps <= 0 else min(args.exec_steps, H)
        lead = min(args.prefetch_lead, n)

        box: dict = {}
        th = None

        steps_since_obs = 0
        exec_t0 = time.time()
        for i in range(n):
            if arm.faulted():
                raise RuntimeError("ArmController control thread faulted — aborting")
            tic = time.time()
            a = actions[i].astype(np.float64)
            # Blend the first blend_count steps from last_cmd into the new chunk.
            # blend_count = elapsed_steps since obs capture, so the stale leading
            # steps (which the model predicted for time already passed) are eased
            # in from the robot's actual current pose rather than snapping or skipping.
            if last_cmd is not None and i < blend_count:
                alpha = (i + 1) / (blend_count + 1)
                a = (1.0 - alpha) * last_cmd + alpha * a
            arm.set_arm_target(a[ARM_CHANNELS])
            grip.set_targets(
                float(np.clip(a[LEFT_GRIPPER_CHANNEL], GRIPPER_MIN, GRIPPER_MAX)),
                float(np.clip(a[RIGHT_GRIPPER_CHANNEL], GRIPPER_MIN, GRIPPER_MAX)),
            )
            last_cmd = a

            if th is None and (n - i) <= lead:
                obs_next = build_obs(cam, arm, grip, prompt, args.send_jpeg)
                th = threading.Thread(target=_infer_worker,
                                      args=(policy, obs_next, box),
                                      daemon=True, name=f"infer-{c}")
                th.start()

            if th is not None:
                steps_since_obs += 1

            sleep = dt - (time.time() - tic)
            if sleep > 0:
                time.sleep(sleep)

            # Switch to the new chunk as soon as it arrives, without waiting
            # for the current chunk to finish.
            if th is not None and ("actions" in box or "err" in box):
                break

        exec_s = time.time() - exec_t0
        join_t0 = time.time()
        if th is not None:
            th.join()
        join_wait_s = time.time() - join_t0

        if "err" in box:
            raise box["err"]
        next_actions = box["actions"]

        # How many steps elapsed since obs was captured:
        #   steps_since_obs: steps actually executed after obs capture
        #   + any stall time if infer ran long (join_wait_s > 0 means we
        #     finished the chunk and still had to wait — shouldn't happen
        #     with early-break, but kept as a safety net)
        elapsed_steps = steps_since_obs + int(join_wait_s * args.control_hz)
        blend_count = min(elapsed_steps, next_actions.shape[0] - 1)

        rec = _timing_rec(box.get("timing", {}), box.get("server_ms"))
        infer_recs.append(rec)
        chunk_recs.append({"exec_s": exec_s, "join_wait_s": join_wait_s})
        log.info(f"[chunk {c}] execute={exec_s:.2f}s join_wait={join_wait_s*1e3:.0f}ms "
                 f"blend={blend_count} | "
                 f"infer wall={rec['wall_ms']:.0f}ms pack={rec['pack_ms']:.0f} "
                 f"send={rec['send_ms']:.0f} wait_recv={rec['wait_recv_ms']:.0f} "
                 f"unpack={rec['unpack_ms']:.0f}"
                 + (f" server={rec['server_ms']:.0f}" if rec['server_ms'] is not None else ""))
        log_chunk_ranges(c, next_actions)
        actions = next_actions

    _summarize_timing(infer_recs, chunk_recs, args)


# ---------- pipeline stages (kept identical in spirit to main.py) ----------

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
    """Release every resource. disable_arm_sdk MUST run — it returns arm
    authority to the locomotion service. Each step isolated so a second Ctrl+C
    cannot skip later steps."""
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
    """--test mode: fake arm/gripper/camera, real policy server connection."""
    log.info("*** TEST MODE — no robot hardware will be used ***")
    arm = _FakeArm()
    arm.start()
    grip = _FakeGripper()
    grip.start()
    cam = _FakeCamera()
    policy = None
    # Import PolicyClient here so --test works even without g1_client installed.
    from g1_client.policy_client import PolicyClient
    loop = _run_inference_loop_sync if args.sync else _run_inference_loop_async
    log.info(f"Loop mode: {'sync' if args.sync else 'async (overlapped)'}")
    try:
        log.info(f"Connecting to policy server {args.server_host}:{args.server_port}")
        policy = PolicyClient(host=args.server_host, port=args.server_port)
        loop(arm, grip, cam, policy, args)
    finally:
        _cleanup(arm, grip, cam, policy)


def _run_real(args) -> None:
    """Normal mode: real robot hardware."""
    global GRIPPER_MIN, GRIPPER_MAX
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from g1_client.arm_controller import ArmController, INIT_POSE_READY
    from g1_client.gripper_controller import GripperController
    from g1_client.gripper_controller import GRIPPER_MIN as _GMIN, GRIPPER_MAX as _GMAX
    from g1_client.camera_client import CameraClient
    from g1_client.policy_client import PolicyClient
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
        _initialize_pose(arm, grip, args, INIT_POSE_READY)
        _wait_for_operator(args)
        log.info(f"Switching arm kp to inference value: {args.inference_kp_arm}")
        arm.set_arm_kp(args.inference_kp_arm)
        policy = PolicyClient(host=args.server_host, port=args.server_port)
        loop = _run_inference_loop_sync if args.sync else _run_inference_loop_async
        log.info(f"Loop mode: {'sync' if args.sync else 'async (overlapped)'}")
        loop(arm, grip, cam, policy, args)
        _initialize_pose(arm, grip, args, INIT_POSE_READY)
    finally:
        _cleanup(arm, grip, cam, policy)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--test", action="store_true",
                   help="Use fake camera/arm/gripper stubs instead of real hardware. "
                        "Tests the full inference loop against a live server without "
                        "a robot. --iface is not required in this mode.")
    p.add_argument("--iface", default=None,
                   help="Network interface to robot, e.g. enp0s31f6 (required without --test)")
    p.add_argument("--server-host", required=True, help="openpi serve_policy.py host or IP")
    p.add_argument("--server-port", type=int, default=8000, help="openpi server port (default 8000)")
    p.add_argument("--image-server", default="192.168.123.164",
                   help="G1 PC2 image-server host (default 192.168.123.164)")
    p.add_argument("--prompt", default="pick the red bottle")
    p.add_argument("--send-jpeg", action="store_true",
                   help="Send compressed JPEG bytes instead of decoded RGB arrays "
                        "(~12x smaller upload — fixes a network-bound wait_recv). "
                        "REQUIRES the server to imdecode + BGR->RGB these image keys.")
    p.add_argument("--max-chunks", type=int, default=300,
                   help="How many action chunks to run before stopping")
    p.add_argument("--control-hz", type=float, default=30.0,
                   help="Per-step dispatch rate; match your LeRobot recording fps (default 30)")
    p.add_argument("--exec-steps", type=int, default=50,
                   help="Steps to execute per chunk before re-querying; 0 = full horizon. "
                        "Set smaller (e.g. 25) for tighter closed-loop reactivity.")
    p.add_argument("--prefetch-lead", type=int, default=10,
                   help="Fire the next inference when this many steps remain in the "
                        "current chunk (default 10). Controls obs freshness: obs is "
                        "captured lead/control_hz seconds before the chunk boundary, "
                        "so the model plans from a near-current state. Raise if "
                        "join_wait > 0 (infer slower than the lead window). "
                        "Only used in async mode.")
    p.add_argument("--blend-steps", type=int, default=5,
                   help="Cross-fade the first N steps of each new chunk from the last "
                        "commanded pose so a chunk swap ramps in smoothly instead of "
                        "snapping (default 5; 0 disables blending)")
    p.add_argument("--sync", action="store_true",
                   help="Run inference synchronously: execute chunk fully, then infer "
                        "next chunk (serialized, robot pauses at every boundary). "
                        "Default is async (overlapped): infer fires at the start of "
                        "each chunk so GPU time is hidden behind execution.")
    # ---- Safety / motion limits (same as main.py) ----
    p.add_argument("--velocity-limit", type=float, default=8.0,
                   help="rad/s velocity cap on the per-tick motion clamp (default 8.0)")
    p.add_argument("--inference-kp-arm", type=float, default=80.0,
                   help="kp for shoulder/elbow once inference starts (default 80)")
    p.add_argument("--init-duration", type=float, default=2.0)
    p.add_argument("--gripper-init-duration", type=float, default=1.0)
    p.add_argument("--settle-duration", type=float, default=1.0)
    p.add_argument("--init-gripper-left", type=float, default=5.0)
    p.add_argument("--init-gripper-right", type=float, default=5.0)
    p.add_argument("--auto-start", action="store_true",
                   help="Skip the post-init Enter prompt and start immediately.")
    args = p.parse_args()

    if not args.test and args.iface is None:
        p.error("--iface is required without --test")

    run(args)


if __name__ == "__main__":
    main()