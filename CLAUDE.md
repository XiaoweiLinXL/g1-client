# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A standalone runtime inference client that drives a Unitree G1 humanoid's arms and grippers from action chunks produced by a remote policy server. Two server protocols are supported:

- **LingBot-VA** (`main.py`) — stateful, FDM-grounded, autoregressive: `reset → cold_start → async_step × N`. Server port 29056.
- **openpi `serve_policy.py`** (`main_openpi.py`, `main_openpi_sync.py`) — stateless, receding-horizon: repeated `infer(obs) → {"actions": [H,16]}`. Server port 8000.

- **Isaac-GR00T** (`main_groot.py`) — stateless, receding-horizon: `infer(obs) → {"actions": [H,17]}` via ZMQ (port 5555). All action channels are **ABSOLUTE** joint targets (server's `StateActionProcessor` converts training-time relative deltas using the obs-time state we send); channel 16 is waist yaw.

Both paths share the same robot control code (`g1_client/` package). Only the policy data layer changes.

## Commands

Two prerequisites are installed separately (not on PyPI under these names):
- **`unitree_sdk2py`** — the Unitree Python SDK. Clone https://github.com/unitreerobotics/unitree_sdk2_python and `pip3 install -e .` there. If the build can't find cyclonedds, build it and `export CYCLONEDDS_HOME=~/cyclonedds/install` (see that repo's README).
- **`teleimager`** — the G1 image-server client, from the `xr_teleoperate` repo.

The remaining deps are pip-installable:

```bash
pip install -r requirements.txt
```

Run the LingBot-VA pipeline (from the repo root):

```bash
python main.py \
    --iface enp0s31f6 \
    --server-host <cloud-ip> \
    --server-port 29056 \
    --prompt "pick up the pink object and place it on the blue cross mark"
```

Run the openpi pipeline:

```bash
python main_openpi.py \
    --iface enp0s31f6 \
    --server-host <openpi-ip> \
    --server-port 8000 \
    --prompt "pick up the pink object"
```

`main_openpi_sync.py` has a `--test` mode for driving a real openpi server with fake hardware (no robot, no DDS):

```bash
python main_openpi_sync.py \
    --test \
    --server-host <openpi-ip> \
    --server-port 8000 \
    --prompt "pick up the pink object"
```

All entry-point scripts (`main.py`, `main_openpi.py`, `main_openpi_sync.py`) import the `g1_client` package and must be run directly from the repo root — Python finds the package because the script's own directory is on `sys.path`. Running them as `python -m ...` will NOT work.

Contract/safety tests (no robot, no DDS required):

```bash
# Wire-contract check against a running LingBot-VA server.
python smoke_test.py --server-host <cloud-ip> --server-port 29056

# Offline check of the async loop's wire schedule using fakes.
python test_async_loop.py

# Offline safety tests: server timeout, camera fail-closed, --repeat flow, reset key, --sync mode.
# Requires unitree_sdk2py installed (imported transitively) but touches no DDS.
python test_async_safety.py

Run the GR00T pipeline (see scripts/ for Azure VM setup):

```bash
python main_groot.py \
    --iface enp0s31f6 \
    --server-host a10-pi05-embodyx.southcentralus.cloudapp.azure.com \
    --server-port 5555 \
    --prompt "Load the bottle water to the shelf"

# Test connectivity without a robot:
python main_groot.py --test \
    --server-host a10-pi05-embodyx.southcentralus.cloudapp.azure.com \
    --prompt "Load the bottle water to the shelf"
```

# Latency profiling for openpi servers — drives main_openpi's real loop with
# synthetic obs. No robot motion. Requires unitree_sdk2py (transitively).
python test_policy_server.py --server-host <openpi-ip> --server-port 8000 --max-chunks 10

# Validate a deployed checkpoint against a HuggingFace LeRobot dataset.
# Sends training frames to a running policy server and compares predictions
# against ground truth. Requires lerobot installed.
python test_checkpoint_on_dataset.py \
    --repo-id XiaoweiLinXL/pi05-unitree-g1-put-away-tools-v2.1 \
    --server-host <openpi-ip> --server-port 8000 --num-samples 10

# Capture one frame from each robot camera and save as JPEG.
# No DDS/arm control. Use to visually compare camera setup against training data.
python capture_frames.py --image-server 192.168.123.164 --out-dir frames/
```

There are no linters or build configs in this repo.

## Architecture

### GR00T path (`main_groot.py`)

Stateless receding-horizon loop: `infer(obs) → {"actions": [H=16, 16]}` over ZMQ REQ/REP. No server state; every call is independent.

**Key protocol differences from the openpi path:**

| Dimension | openpi | GR00T |
|---|---|---|
| Transport | WebSocket | ZMQ REQ/REP (port 5555) |
| Obs format | flat dict, state (16,) | nested `video/state/language`, state split into 4 keys each `(1,1,D)` |
| Image format | decoded RGB uint8 HxWx3 | decoded RGB uint8 `(1,1,H,W,3)` (no JPEG bypass) |
| Action format | `[H,16]` absolute rad | 4-key dict; arm=RELATIVE deltas, gripper=ABSOLUTE |
| Server package | openpi JAX | Isaac-GR00T PyTorch on Azure A10 |

**Obs conversion** (`g1_client/groot_policy.py`): JPEG bytes → RGB → `(1,1,H,W,3)`; state `(16,)` → split into `left_arm (1,1,7)`, `right_arm (1,1,7)`, `left_gripper (1,1,1)`, `right_gripper (1,1,1)`.

**Action integration**: arm deltas are RELATIVE to the arm_q snapshot taken at obs capture time, not cumulative. `main_groot.py` does `arm_target[t] = obs_arm_q_snapshot + actions[t, :14]` for every step in the chunk.

**Wire format** (`g1_client/groot_policy.py`): `msgpack.packb` with `msgpack_numpy.encode` as default; `msgpack.unpackb` with `msgpack_numpy.decode` as object_hook — compatible with GR00T's `MsgSerializer`. Does NOT use the vendored `g1_client/msgpack_numpy.py` (which is the LingBot-VA WebSocket format).

### LingBot-VA path (`main.py`)

The inference pipeline overlaps the policy-server request with on-robot chunk execution (Algorithm 2 in `main.py`, "async FDM-grounded loop"):

```
Cloud LingBot-VA server  ──(WebSocket + msgpack_numpy)──►  PolicyClient
        ▲                                                     │
        │ async_step (daemon thread)                          ▼ action [16,F,S]
        │  obs=K_{n-1}, state=a_{n-1},                    execute_chunk()
        │  executing_action=C_n                           ├─► ArmController     ──(DDS rt/arm_sdk @ 50 Hz)──► G1 arms+body lock
        │                                                ├─► GripperController ──(DDS rt/dex1/{l,r}/cmd @ 200 Hz)──► Dex1 grippers
        └────────────────────── keyframes (JPEG bytes) ◄─┴─► CameraClient.get_obs_images() ◄──(ZMQ teleimager)── G1 PC2 image server
```

**Action tensor layout.** Each chunk is shape `(16, F, S)` — 16 channels, `F` latent frames, `S` sub-steps/frame. `F` and `S` are read from the returned tensor (`action.shape[1]`/`[2]`), **not assumed**: the original g1 model is `F=2, S=16`; `g1_500step` is `F=4, S=16`. The named constants at the top of `main.py`:
- `ARM_CHANNELS = slice(0, 14)` — 14 arm joints, ordered L{pitch,roll,yaw,elbow,wristR,wristP,wristY} then R{same}. This order is hard-coded in `g1_client.arm_controller.ARM_JOINTS` and the model is trained to match it — do not reorder.
- `LEFT_GRIPPER_CHANNEL = 14`, `RIGHT_GRIPPER_CHANNEL = 15` — gripper q in `[GRIPPER_MIN=0, GRIPPER_MAX=5.4]` rad.

**Chunk execution cadence.** 30 Hz sub-step dispatch → one chunk = `F × S` sub-steps (≈1.07 s at F=2, ≈2.13 s at F=4). Keyframes are captured every `CAPTURE_EVERY=4` sub-steps (within each executed frame the captures fire at `f=3,7,11,15`) and are sent back in the **next** `async_step` request as the `obs` field. Counts:
- Chunk 0 runs with `is_first_chunk=True` → frame 0 skipped → `(F-1)×S/CAPTURE_EVERY` keyframes (**4 at F=2**).
- Chunks 1+ run all frames → `F×S/CAPTURE_EVERY` keyframes (**8 at F=2**).
- Cycle 0's `async_step` carries no `obs`/`state` — the server grounds `z_0` from its own init pose.

**Wire schedule.** `reset(prompt)` → `cold_start(obs=<single dict>)` → repeated `async_step` until `--max-chunks` is hit. `compute_kv_cache` is **not** used on the async path; grounding folds into `async_step`. `cold_start` returns a single chunk `resp["action"]` (one chunk, not two). `test_async_loop.py` and `smoke_test.py` pin this schedule exactly.

**Contract: keyframe counts and identity must match the server's expectation.** The first grounding cycle (cycle 1) must carry exactly 4 keyframes (not 8) because chunk 0 skipped frame 0; steady cycles carry 8. `state` and `executing_action` are passed as the **same ndarray objects** the server returned — no copy/reshape/renormalize on the client side. Too few/many keyframes, or a quietly reshaped `state`, desync the server's autoregressive context with no error — the same silent-failure class as the camera color-order contract below.

**Branch A / Branch B overlap.** Inside `_run_task_once`, each steady cycle dispatches Branch B (`_async_step_worker` on a daemon thread) **before** running Branch A (`execute_chunk` on the main thread). The daemon thread does the blocking `policy.infer(...)`; the main thread streams the current chunk to DDS at 30 Hz. Only after `execute_chunk` returns does the main thread `out_q.get()` on Branch B's result. Errors raised inside the daemon are surfaced via the queue (`("err", e)`) so a network failure aborts the loop.

**`--repeat` / `--sync` / `--server-timeout`.** Key `main.py` flags not visible in older docs:
- `--repeat`: after each task completes, return arm to ready pose, wait for Enter (init then standby), and run again. Server KV cache is cleared via a fresh `reset` each iteration. `--auto-start --repeat` loops forever unattended.
- `--sync`: run `async_step` on the main thread *after* `execute_chunk` instead of overlapping on a daemon. Same wire contract, no overlap. Useful for latency debugging.
- `--server-timeout` (default 60 s): `out_q.get` timeout on the daemon; prevents an infinite hang when the server dies while the robot is arm_sdk latched.

**Reset key ('r').** While a task is running, `_make_reset_watcher` puts stdin in cbreak mode (single-keystroke, no Enter) and starts a daemon thread monitoring for `'r'`/`'R'`. When pressed, it sets a `reset_event` that `_run_task_once` checks at each chunk boundary, raising `ResetRequested`. This triggers: `_initialize_pose` → `_wait_for_operator` (Enter) → fresh `reset`+`cold_start`. The reset key is always active — not gated on `--repeat`. Terminal is restored to canonical mode before any `input()` call (critical: otherwise the next `input()` echoes nothing).

### openpi path (`main_openpi.py` / `main_openpi_sync.py`)

Stateless receding-horizon loop: `infer(obs) → {"actions": [H, 16]}` on every tick. No `cold_start`, `reset`, or server state. Obs keys must match the checkpoint's `RepackTransform`/`DataConfig`:

```
observation.images.cam_left_high    uint8 HxWx3 RGB   (main_openpi.py key)
observation.images.cam_left_wrist   uint8 HxWx3 RGB
observation.images.cam_right_wrist  uint8 HxWx3 RGB
observation.state                   float32 (16,) = [14 arm q | L grip | R grip]
prompt                              str
```

> **Key difference between the two openpi entry points:** `main_openpi.py` sends obs keys with dots (`observation.images.cam_left_high`), while `main_openpi_sync.py` maps to slash-separated keys (`observation/image`, `observation/left_wrist_image`, `observation/right_wrist_image`). Keep the obs key format in lockstep with your server checkpoint's `RepackTransform`.

**One-chunk prefetch with boundary smoothing.** `_run_inference_loop` in `main_openpi.py` executes the current chunk step-by-step at `--control-hz` (default 15 Hz). When `--prefetch-lead` steps remain, it snapshots a fresh obs and fires the next infer on a daemon thread so it overlaps the chunk tail. Two anti-jitter measures at chunk boundaries:
- **Time-alignment** (`--chunk-align`, on by default): skip the leading `(n-1-i)` steps of a newly received chunk that already "elapsed" during inference, so the arm doesn't jump back then forward.
- **Cross-fade** (`--blend-steps`, default 5): ramp the first N steps from the last commanded pose into the new chunk linearly.

**`--send-jpeg` flag.** By default, images are decoded to RGB arrays before sending (~720 KiB/obs). With `--send-jpeg`, the raw JPEG bytes from the camera (BGR-encoded, q90) are sent directly (~60 KiB/obs, ~12× smaller). The server must then call `cv2.imdecode` and `cv2.COLOR_BGR2RGB` on these keys — a silent channel-swap if mismatched.

**`g1_client/openpi_policy.py`** — thin wrapper over `openpi_client.websocket_client_policy.WebsocketClientPolicy`. This requires `openpi_client` installed (not in `requirements.txt`; comes with the openpi package). Used only by code that directly instantiates `OpenPIPolicy`; `main_openpi.py` uses `g1_client.policy_client.PolicyClient` instead (the vendored msgpack client, no extra dep).

### Shared robot control layer

**ArmController (`g1_client/arm_controller.py`).** Publishes a `LowCmd_` on `rt/arm_sdk` at 50 Hz. To take control of arm joints from the locomotion service, it sets `motor_cmd[29].q = 1.0` (the `kNotUsedJoint` slot doubles as the arm_sdk handover weight). The same `LowCmd_` also pins legs (`LEG_JOINTS` 0–11) and waist (`WAIST_JOINTS` 12–14) at their startup pose with `kp_body_lock=300`, so locomotion stays balanced while arms are driven. Per-tick velocity is clamped via `_clip_target` against `velocity_limit`; `move_to_pose` temporarily overrides it for the duration of the ramp and restores it on exit, but the inference loop never touches it — so the clamp is effectively a single fixed value during model-driven motion. `set_arm_target` also clips every command to per-joint position limits (`ARM_JOINT_MIN`/`ARM_JOINT_MAX`, defined at `g1_client/arm_controller.py:100-107`) before it ever reaches the wire, a second safety layer beyond the velocity clamp. These limits are deliberately *tighter than the hardware limits* to keep a margin — do not widen them to match the spec sheet without a specific reason. On exit, `disable_arm_sdk()` ramps the weight back to 0 to hand control to the loco service. **kp switching pattern**: `kp_arm` defaults to **150** (stiffer hold, less gravity sag during the standby wait); `arm.set_arm_kp(80)` is called once Enter is pressed (just before connecting to the policy server) to drop into the softer kp the model was trained against. `set_arm_kp` only touches the 8 shoulder/elbow joints — the 6 wrist joints stay at `kp_wrist=40` for the whole run, so any future kp tuning that should also affect the wrists has to be added explicitly. Required precondition: robot is in **ai** motion mode and standing — `main.py` does not switch modes, operator sets it via the Unitree app.

**Publish-thread fault detector.** `ArmController` sets a `_faulted` flag if its publish or subscribe loop crashes on an unhandled exception, and exposes `faulted()`. `execute_chunk` polls `arm.faulted()` every sub-step and raises — otherwise a dead publish thread would silently stop `arm_sdk` frames while `motor_cmd[29].q` is still latched at 1, and the loco service would not regain authority.

**GripperController (`g1_client/gripper_controller.py`).** Publishes `MotorCmds_` on `rt/dex1/left/cmd` and `rt/dex1/right/cmd` at 200 Hz with a `DELTA_GRIPPER_CMD=0.18 rad` per-tick rate cap (the publish thread, not the user, enforces this). State is read back from `rt/dex1/{left,right}/state`.

**CameraClient (`g1_client/camera_client.py`).** Uses `teleimager.ImageClient` (host defaults to `192.168.123.164`, the G1 PC2). The head camera may be binocular; if so, only its **left half** is taken to match the training format. All three streams are resized to **256×320** (H×W) and **JPEG-encoded at q90 in cv2's native BGR order** — the wire format is JPEG `bytes`, not a NumPy array.

**Color-order contract.** Because frames are encoded straight from BGR, the **server is responsible for the BGR→RGB conversion** after `cv2.imdecode` (which returns BGR). The client no longer does this. Keep client encode and server decode in lockstep — getting the order wrong silently feeds the model channel-swapped images.

`get_obs(prompt)` returns the cold-start payload (3 cams + `"task"` key); `get_obs_images()` returns the camera-only dict used for keyframes inside `execute_chunk`.

**PolicyClient (`g1_client/policy_client.py`).** Synchronous `websockets.sync.client` connection with `ping_interval=None` so long inference calls (multi-second GPU work) don't trip the keepalive. The wire format is msgpack with NumPy support (`g1_client/msgpack_numpy.py`, mirrors the server-side module byte-for-byte — keep them in sync). The client itself is **protocol-agnostic** — `infer(payload)` packs whatever dict you give it; `reset(prompt)` is the only helper. `last_timing` exposes a pack/send/wait_recv/unpack breakdown of the most recent call. `_wait_for_server` retries forever on 5-second intervals — intentional, since the operator may start the policy server *after* the client. Do not "fix" this to a bounded retry without also revisiting the operator workflow.

**SDK channel choice.** G1 uses `unitree_sdk2py.idl.unitree_hg` (NOT `unitree_go`, which is for Go2/B2/H1). The Dex1 grippers in `g1_client/gripper_controller.py` are an exception — they use the older `unitree_go` IDL because dex1 is a separate accessory; do not "fix" this to match the arms.

### Startup sequence

`arm.start()` → `move_to_pose(INIT_POSE_READY)` (5 s ramp at the operating vlim) → grippers do a `close→open` sequence so the operator can see the dex1 has joined the SDK → `--settle-duration` pause → `input("")` standby (skipped with `--auto-start`) → `arm.set_arm_kp(args.inference_kp_arm)` (default 80) → connect to policy server.

### Shutdown safety

A single `try`/`finally` block in `run()` wraps everything from gripper init onwards. On any exit path (normal completion, exception, or Ctrl+C), the finally calls `_cleanup`, which runs each release step in its own `try/except BaseException` so any single failure (including a second Ctrl+C landing mid-cleanup) cannot skip the others. Steps:
1. `arm.stop()` — joins the publish thread.
2. `arm.disable_arm_sdk()` — ramps `motor_cmd[29].q` 1→0 over 1 s, with its own internal `try/finally` that **guarantees a terminal `q=0` write** even if the ramp is interrupted mid-sleep.
3. `grip.stop()` — parks the gripper target at the current measured position before signaling stop.
4. `cam.close()`, `policy.close()` — each may be `None` if the failure happened before that resource was constructed.

**`except BaseException`, not `except Exception`.** `KeyboardInterrupt` is `BaseException`, so a plain `except Exception` would let a second Ctrl+C escape and skip the rest of cleanup. The pattern in `_cleanup` is deliberate; do not "tighten" it.

## Test scripts

**`test_async_loop.py`** — offline check of the LingBot-VA wire schedule using fakes. Pins the exact `reset → cold_start → async_step` sequence, keyframe counts (4 for cycle 0, 8 for cycles 1+), and verbatim identity on `state`/`executing_action`. Requires `unitree_sdk2py` at import time even though no DDS channel is opened. Run after any change to `_run_task_once` or `execute_chunk`.

**`smoke_test.py`** — wire-contract check against a running LingBot-VA server (no robot/cameras needed). Validates the full cloud round-trip with the exact protocol `main.py` uses.

**`test_async_safety.py`** — offline safety tests for two latched-robot failure modes and multi-task flows. Tests:
- **#17** non-responding server → `TimeoutError` in `--server-timeout` seconds (not infinite hang with arm_sdk latched)
- **#2** camera capture failure mid-chunk → `RuntimeError` raised loud (no silent short keyframe list that desyncs server grounding)
- `--repeat` between-task flow: Enter#1 → INIT → Enter#2 → fresh `reset`+`cold_start`
- Reset key mid-task: 'r' aborts at chunk boundary → INIT → Enter → re-run (resets server KV cache)
- `--sync` mode: all `async_step` calls run on the main thread (verifies no daemon spawned)

**`test_policy_server.py`** — latency profiler for openpi servers. Drives `main_openpi._run_inference_loop` with self-generated synthetic obs (no robot motion). Reports: in-loop FPS vs target `control_hz` (stalls show as `fps < control_hz`), per-infer latency breakdown (pack/send/wait_recv/unpack), action contract checks (NaN/inf, out-of-range, unit heuristics). Distinguishes GPU compute time from network RTT when the server reports `server_timing`.

**`test_checkpoint_on_dataset.py`** — validates a deployed checkpoint against a HuggingFace LeRobot dataset. Loads frames from the dataset, sends them to a running policy server, and compares the first predicted action against ground truth. Low L2 error on training samples confirms the model learned the data; high error indicates a config mismatch (wrong norm stats, wrong obs key format, etc). Uses `main_openpi_sync.py`-style obs keys (`observation/image`, `observation/left_wrist_image`, `observation/right_wrist_image`). Requires `lerobot` installed.

**`capture_frames.py`** — diagnostic: connects to the G1 image server, captures one frame from each camera, and saves them as RGB JPEGs. No DDS or arm control. Use to visually compare the actual camera setup against training dataset frames to diagnose mounting/angle mismatches before a run.

## `scripts/` — Azure VM deployment

**`scripts/setup_azure_groot.sh`** — one-time setup for the Azure A10 VM (`fabricio@a10-pi05-embodyx.southcentralus.cloudapp.azure.com`, key at `~/azure_files/a10-1.5-inference-fabricio_key.pem`). Installs system deps (git-lfs, ffmpeg), clones Isaac-GR00T from GitHub, installs uv, and runs `uv sync --python 3.10`.

**`scripts/launch_groot_server.sh`** — downloads the chosen GR00T checkpoint from HuggingFace (if not already local) and starts the ZMQ policy server on port 5555. Usage: `bash launch_groot_server.sh [9000step|18000step|30000step]`. Available checkpoints: `XiaoweiLinXL/unitree-GR00T-load-bottle-water-{9000,18000,30000}step`. Port 5555 must be open in the Azure NSG.

## `openpi-fintune/` directory

A copy of the openpi training framework (JAX + PyTorch model implementations, training configs, data loaders, policy server). Of direct relevance to this client:
- `src/openpi/policies/unitree_policy.py` — the G1 `DataConfig` and `RepackTransform` that define the exact obs key names and action normalization the server expects. If obs keys or action units seem wrong on a new checkpoint, check this file first.
- `scripts/serve_policy.py` — the openpi policy server that `main_openpi.py` talks to.
- `packages/openpi-client/` — the `openpi_client` package used by `g1_client/openpi_policy.py`.

The `openpi-fintune/` code is a separate codebase; edits there are independent from the client code above.

## Wire-contract status

**Current client protocol (main.py + tests, all consistent):**
- `cold_start` returns one chunk: `C0 = resp["action"]` (no `"action1"`).
- Cycle 0's `async_step` sends only `executing_action` — no `obs`/`state`.
- Cycles 1+ send `obs`/`state`/`executing_action`.

All client-side code (`main.py`, `smoke_test.py`, `test_async_loop.py`, `test_async_safety.py`) agrees on this single-chunk protocol. **Server-side alignment should be re-verified** before making protocol changes — the deployed server at `/home/nuwm-1/Workspace/dev/lingbot-va/wan_va/wan_va_server.py` was previously documented as expecting two chunks back from `cold_start`, but the client has since been updated to the single-chunk path.

**Color-order contract** — camera frames are JPEG-encoded in BGR; the server decodes and swaps to RGB after `cv2.imdecode`. Getting this wrong silently feeds channel-swapped images. Keep client encode and server decode in lockstep.

**Keyframe-count contract** — sending the wrong count (e.g. due to a misconfigured `CAPTURE_EVERY` or a swallowed camera exception) desyncs the server's grounding with no error. `execute_chunk` defends this with an explicit assert after the loop and fails loud on any camera exception rather than continuing with a short list.
