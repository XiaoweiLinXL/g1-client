"""GR00T policy client for the G1 inference loop.

Wraps the Isaac-GR00T ZMQ REQ/REP server without requiring Isaac-GR00T installed
on the robot — only pyzmq and msgpack-numpy are needed (both in requirements.txt).

Wire format: msgpack + msgpack_numpy (compatible with GR00T's MsgSerializer).

The server is launched on the Azure VM via:
    uv run python gr00t/eval/run_gr00t_server.py \\
        --model-path checkpoints/g1-30000step \\
        --embodiment-tag new_embodiment --device cuda:0 --host 0.0.0.0 --port 5555

Action layout returned by infer():
    With waist (use_waist=True):   result["actions"]  shape [H, 17]
    Without waist (use_waist=False): result["actions"] shape [H, 16]

    [:, 0:7]   left  arm joint ABSOLUTE targets (rad)
    [:, 7:14]  right arm joint ABSOLUTE targets (rad)
    [:, 14]    left  gripper absolute target  (rad, [GRIPPER_MIN, GRIPPER_MAX])
    [:, 15]    right gripper absolute target
    [:, 16]    waist yaw ABSOLUTE target (rad)  — only when use_waist=True

NOTE: Even though the model was trained with use_relative_action=True, the server's
StateActionProcessor.unapply_action() already converts relative deltas → absolute positions
using the obs-time arm_q we send in the state observation. Do NOT add obs_arm_q to the
returned arm actions — they are already absolute joint position targets.
"""

import time

import cv2
import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq


def _enc(obj):
    if isinstance(obj, np.ndarray) and obj.dtype.kind == "O":
        raise TypeError(f"Object-dtype ndarray rejected (shape={obj.shape})")
    return mnp.encode(obj)


def _pack(data) -> bytes:
    return msgpack.packb(data, default=_enc)


def _unpack(data: bytes):
    return msgpack.unpackb(data, object_hook=mnp.decode, raw=False)


def _jpeg_bgr_to_rgb_uint8(jpeg_bytes: bytes) -> np.ndarray:
    """Decode a BGR JPEG (as produced by CameraClient) to an RGB uint8 HxWx3 array."""
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("cv2.imdecode failed on camera frame")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class GR00TPolicy:
    """Thin ZMQ client that translates between g1-client obs/action conventions
    and the GR00T server's nested observation / split-action dict protocol."""

    def __init__(self, host: str, port: int = 5555, timeout_ms: int = 60_000,
                 use_waist: bool = True, image_hw: tuple[int, int] = (256, 342)):
        self._context = zmq.Context()
        self._host = host
        self._port = port
        self._timeout_ms = timeout_ms
        self._use_waist = use_waist
        self._image_h, self._image_w = image_hw
        self._socket = self._make_socket()
        self.last_timing: dict = {}

    def _make_socket(self):
        sock = self._context.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        sock.connect(f"tcp://{self._host}:{self._port}")
        return sock

    def _call(self, endpoint: str, data: dict | None = None):
        req = {"endpoint": endpoint}
        if data is not None:
            req["data"] = data
        self._socket.send(_pack(req))
        raw = self._socket.recv()
        resp = _unpack(raw)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"GR00T server error: {resp['error']}")
        return resp

    def ping(self) -> bool:
        try:
            self._call("ping", data=None)
            return True
        except zmq.error.ZMQError:
            self._socket = self._make_socket()
            return False

    def infer(self, obs: dict) -> dict:
        """Convert g1-client flat obs dict → GR00T nested obs, call server,
        return {"actions": [H, 17]}.

        obs keys consumed:
            observation.images.cam_left_high    JPEG bytes (BGR q90); resized to 256×342
            observation.images.cam_left_wrist   JPEG bytes (BGR q90); resized to 256×342
            observation.images.cam_right_wrist  JPEG bytes (BGR q90); resized to 256×342
            observation.state                   float32 (17,) [14 arm q | L grip | R grip | waist_yaw]
            prompt                              str — must match training task exactly
        """
        t0 = time.time()

        # ---- build GR00T nested observation ----
        def _img(key):
            rgb = _jpeg_bgr_to_rgb_uint8(obs[key])  # (H, W, 3)
            if rgb.shape[0] != self._image_h or rgb.shape[1] != self._image_w:
                rgb = cv2.resize(rgb, (self._image_w, self._image_h))  # cv2: (width, height)
            return rgb[np.newaxis, np.newaxis]        # (1, 1, H, W, 3)

        state = np.asarray(obs["observation.state"], dtype=np.float32)  # (16,) or (17,)
        state_dict = {
            "left_arm":      state[0:7].reshape(1, 1, 7),
            "right_arm":     state[7:14].reshape(1, 1, 7),
            "left_gripper":  state[14:15].reshape(1, 1, 1),
            "right_gripper": state[15:16].reshape(1, 1, 1),
        }
        if self._use_waist:
            waist_q = state[16:17] if len(state) > 16 else np.zeros(1, dtype=np.float32)
            state_dict["waist"] = waist_q.reshape(1, 1, 1)
        groot_obs = {
            "video": {
                "cam_left_high":   _img("observation.images.cam_left_high"),
                "cam_left_wrist":  _img("observation.images.cam_left_wrist"),
                "cam_right_wrist": _img("observation.images.cam_right_wrist"),
            },
            "state": state_dict,
            "language": {
                "annotation.human.task_description": [[obs["prompt"]]],
            },
        }

        # ---- call server ----
        t_send = time.time()
        response = self._call("get_action", {"observation": groot_obs, "options": None})
        t_recv = time.time()

        # response is a list [action_dict, info_dict] (msgpack list → tuple by server)
        chunk = response[0] if isinstance(response, (list, tuple)) else response
        # chunk keys: left_arm (1,H,7), right_arm (1,H,7), left_gripper (1,H,1),
        #             right_gripper (1,H,1), waist (1,H,1) — waist absent on no-waist models
        H = np.asarray(chunk["left_arm"]).shape[1]
        n_cols = 17 if "waist" in chunk else 16
        actions = np.zeros((H, n_cols), dtype=np.float32)
        actions[:, 0:7]  = np.asarray(chunk["left_arm"])[0]
        actions[:, 7:14] = np.asarray(chunk["right_arm"])[0]
        actions[:, 14]   = np.asarray(chunk["left_gripper"])[0, :, 0]
        actions[:, 15]   = np.asarray(chunk["right_gripper"])[0, :, 0]
        if "waist" in chunk:
            actions[:, 16] = np.asarray(chunk["waist"])[0, :, 0]

        self.last_timing = {"wall_ms": (t_recv - t0) * 1e3, "infer_ms": (t_recv - t_send) * 1e3}
        return {"actions": actions}

    def close(self) -> None:
        try:
            self._socket.close(linger=0)
        except Exception:
            pass
        try:
            self._context.term()
        except Exception:
            pass
