# Session Notes — 2026-08-20

Combined notes from both the hardware/infrastructure session and the code-changes session.

---

## Hardware & infrastructure

### Camera setup on PC2 (Jetson NX)

RealSense head camera replaced with a plain RGB USB camera. **Always identify cameras by serial number** — `/dev/videoX` numbers shift on every reboot.

| Camera | Serial | Type | Note |
|--------|--------|------|------|
| Head (Arducam 1080P Low Light) | `UC684` | `uvc` | Must use libusb — V4L2 bandwidth contention |
| Left wrist (JSK-WDR 0001) | `0001` | `opencv` | |
| Right wrist (JSK-WDR 0002) | `0002` | `opencv` | |

The Arducam must use `type: uvc` (libusb) not `type: opencv` (V4L2). Three USB 2.0 cameras on one hub cause V4L2 bandwidth contention; the third can't produce frames. Side effect: `/dev/video4+` disappears from sysfs while the service runs (expected).

Config: `/home/unitree/xr_teleoperate/teleop/teleimager/cam_config_server.yaml`

### Five bug fixes in image_server.py

File: `/home/unitree/xr_teleoperate/teleop/teleimager/src/teleimager/image_server.py`

- **a)** `reload_uvc_driver()` — skip reload if driver already loaded (avoids ~5s delay on every startup)
- **b)** Sleep 1s after `uvc.device_list()` — prevents race where libusb briefly claims devices, blocking OpenCV
- **c)** `_is_like_rgb()` — retry 10× with 0.2s delay for cameras that need multiple frames to stabilize
- **d)** `_can_read_frame()` — non-fatal on USB bandwidth contention: retry 15× then warn and return `True`
- **e)** `get_uid_by_sn()` — fallback to `uid_map` when V4L2 driver is detached (Arducam disappears from `uvc_rgb_cameras` after a prior UVC open)

### Teleimager service & SSL certs

Service runs as root; certs must be at `/root/.config/xr_teleoperate/`.

```bash
sudo cp /home/unitree/.config/xr_teleoperate/{cert,key}.pem /root/.config/xr_teleoperate/

sudo systemctl restart teleimager
sudo journalctl -u teleimager -f

# If cameras disappear after a crash:
sudo modprobe -r uvcvideo && sleep 1 && sudo modprobe uvcvideo
```

WebRTC streams: `https://192.168.10.62:60001/` (head), `:60002/` (left wrist), `:60003/` (right wrist).

### openpi policy server on EmbodyX-server

Checkpoint: `pi05_unitree_load_bottle_water_20k` — **17-dim state/action (with waist)**.

```bash
cd ~/xiaowei/openpi-finetuning
nohup ~/.local/bin/uv run python scripts/serve_policy.py \
    --port 8000 \
    policy:checkpoint \
    --policy.config pi05_unitree_g1_load_bottle_water \
    --policy.dir checkpoints/pi05_unitree_load_bottle_water_20k \
    > /tmp/openpi_server.log 2>&1 &

tail -f /tmp/openpi_server.log
```

### pinocchio installed on PC2 venv

```bash
pip install pin   # installs pinocchio 4.1.0
```

numpy upgraded to 2.2.6 — teleimager imports cleanly. Kinematics confirmed working.

---

## Code changes

### main_openpi.py — obs wire keys (fix)

Keys renamed to match the server's `RepackTransform`:

| Before | After |
|--------|-------|
| `observation/image` | `observation/cam_left_high` |
| `observation/left_wrist_image` | `observation/cam_left_wrist` |
| `observation/right_wrist_image` | `observation/cam_right_wrist` |

Dot-format keys (`observation.images.*`) are the internal `CameraClient` dict format only.

### main_openpi.py — gravity feedforward compensation (new)

- `--tauff-scale` flag (default `1.0`)
- Lazily loads `G1DualArmKinematics` in `run()` when `tauff_scale > 0`
- Calls `arm.set_arm_tauff(kin.gravity_torque(a[ARM_CHANNELS], args.tauff_scale))` after every `set_arm_target`
- Torques zeroed on loop exit
- Matches collection-time dynamics (`sol_tauff`) the model was trained against

### main_openpi.py — waist joint support (new)

- `WAIST_CHANNEL = 16`
- `build_obs(with_waist=False)` — appends `arm.get_waist_yaw()` → 17-dim obs
- `arm.set_waist_yaw_target(float(a[WAIST_CHANNEL]))` in step loop
- `--with-waist` flag

### main_openpi_sync.py — chunk-boundary jump-back fix

In async mode, obs is captured mid-chunk; the new chunk's leading actions predict positions from that past moment, causing the arm to snap back at chunk boundaries.

**Old**: blend stale steps from `last_cmd` — softened but didn't eliminate the snap.  
**New**: skip stale steps, then apply a fixed `--blend-steps` cross-fade on the first non-stale steps.

```python
elapsed_steps = steps_since_obs + int(join_wait_s * args.control_hz)
skip          = min(elapsed_steps, next_actions.shape[0] - 1)
blend_count   = args.blend_steps          # fixed, not dynamic
actions       = next_actions[skip:]       # skip stale leading steps
```

Log now shows `skip=N blend=M` at each chunk boundary.

### CLAUDE.md fixes

- **GR00T actions**: corrected RELATIVE → ABSOLUTE (server's `StateActionProcessor` converts; client applies directly)
- **GR00T run command**: SSH tunnel (`127.0.0.1:5555`), `--no-waist` for 16-dim checkpoint, put-away-tools prompt
- **Added**: camera hardware section, teleimager commands, EmbodyX-server start command, PC2 run command
- **Added**: `test_groot_on_dataset.py` to test scripts

---

## Run command — bottle water task

```bash
ssh unitree@192.168.10.62   # password: 123
cd ~/g1-client && source .venv/bin/activate

python main_openpi_sync.py \
    --iface enP8p1s0 \
    --server-host EmbodyX-server.local \
    --server-port 8000 \
    --with-waist \
    --prompt "Load the bottle water to the shelf" \
    --tauff-scale 1.0
```
