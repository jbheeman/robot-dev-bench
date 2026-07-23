"""Pure, testable core for the 2D->3D lifting pipeline.

Ported from rt-pose-atao/src/lift_core.py.

No cv2, no torch model, no I/O. Everything here is numpy in / numpy out so it can
be unit-tested in isolation. The stateful pieces (sliding window, per-joint hold)
are small classes with obvious semantics.

Joint conventions
-----------------
COCO-17 (what YOLO11-pose / ViTPose emits), index -> name:
    0 nose        1 l_eye     2 r_eye     3 l_ear      4 r_ear
    5 l_shoulder  6 r_shoulder 7 l_elbow  8 r_elbow    9 l_wrist  10 r_wrist
    11 l_hip      12 r_hip    13 l_knee   14 r_knee    15 l_ankle 16 r_ankle

H36M-17 (what MotionBERT / MotionAGFormer expect), index -> name:
    0 pelvis   1 r_hip   2 r_knee  3 r_ankle
    4 l_hip    5 l_knee  6 l_ankle
    7 spine    8 thorax  9 neck/nose  10 head
    11 l_shoulder 12 l_elbow 13 l_wrist
    14 r_shoulder 15 r_elbow 16 r_wrist
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

# --- COCO indices (readability) ---
(NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHO, R_SHO, L_ELB, R_ELB,
 L_WRI, R_WRI, L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK) = range(17)

# H36M skeleton edges (parent, child) used for both 3D rendering and structure.
H36M_EDGES = [
    (0, 1), (1, 2), (2, 3),        # right leg
    (0, 4), (4, 5), (5, 6),        # left leg
    (0, 7), (7, 8), (8, 9), (9, 10),  # spine -> head
    (8, 11), (11, 12), (12, 13),   # left arm
    (8, 14), (14, 15), (15, 16),   # right arm
]

# Which H36M joints belong to the legs -- these are the ones that vanish when the
# lower body leaves frame, and the ones we must never let crash the pipeline.
H36M_LEG_JOINTS = (1, 2, 3, 4, 5, 6)


def coco2h36m(kp: np.ndarray) -> np.ndarray:
    """Remap COCO-17 keypoints to H36M-17.

    kp: (..., 17, C) where C is 2 (x,y) or 3 (x,y,conf). Composite H36M joints
    (pelvis, spine, thorax, head) are synthesized by averaging COCO joints; when
    a confidence channel is present the composite confidence is the *min* of its
    parents (a joint is only as trustworthy as its weakest input).
    """
    kp = np.asarray(kp, dtype=np.float32)
    if kp.shape[-2] != 17:
        raise ValueError(f"expected 17 COCO joints, got shape {kp.shape}")
    C = kp.shape[-1]
    out = np.zeros_like(kp)

    def avg(*idxs):
        # mean of xy; min of conf (if present)
        pts = [kp[..., i, :] for i in idxs]
        m = np.mean(pts, axis=0)
        if C == 3:
            m[..., 2] = np.min([p[..., 2] for p in pts], axis=0)
        return m

    out[..., 0, :] = avg(L_HIP, R_HIP)          # pelvis
    out[..., 1, :] = kp[..., R_HIP, :]
    out[..., 2, :] = kp[..., R_KNE, :]
    out[..., 3, :] = kp[..., R_ANK, :]
    out[..., 4, :] = kp[..., L_HIP, :]
    out[..., 5, :] = kp[..., L_KNE, :]
    out[..., 6, :] = kp[..., L_ANK, :]
    out[..., 8, :] = avg(L_SHO, R_SHO)          # thorax = mid-shoulder
    # spine = mid(pelvis, thorax); recompute conf as min of those two
    out[..., 7, :] = (out[..., 0, :] + out[..., 8, :]) * 0.5
    if C == 3:
        out[..., 7, 2] = np.minimum(out[..., 0, 2], out[..., 8, 2])
    out[..., 9, :] = kp[..., NOSE, :]           # neck/nose
    out[..., 10, :] = avg(L_EYE, R_EYE)         # head

    # Synthesize neck and head using the torso vector (thorax - pelvis)
    # This prevents the 3D lifter from contorting if face points are missing/0-confidence
    # (like on a humanoid robot). We fallback to synthetic if conf < 0.1
    torso_vec = out[..., 8, :2] - out[..., 0, :2]
    synth_neck = out[..., 8, :2] + torso_vec * 0.4
    synth_head = out[..., 8, :2] + torso_vec * 0.8

    if C == 3:
        # Use synthetic if confidence is too low
        nose_ok = out[..., 9, 2:3] > 0.1
        head_ok = out[..., 10, 2:3] > 0.1

        out[..., 9, :2] = np.where(nose_ok, out[..., 9, :2], synth_neck)
        out[..., 9, 2] = np.where(nose_ok[..., 0], out[..., 9, 2], out[..., 8, 2])

        out[..., 10, :2] = np.where(head_ok, out[..., 10, :2], synth_head)
        out[..., 10, 2] = np.where(head_ok[..., 0], out[..., 10, 2], out[..., 8, 2])
    else:
        out[..., 9, :2] = synth_neck
        out[..., 10, :2] = synth_head
    out[..., 11, :] = kp[..., L_SHO, :]
    out[..., 12, :] = kp[..., L_ELB, :]
    out[..., 13, :] = kp[..., L_WRI, :]
    out[..., 14, :] = kp[..., R_SHO, :]
    out[..., 15, :] = kp[..., R_ELB, :]
    out[..., 16, :] = kp[..., R_WRI, :]
    return out


def normalize_screen(kp_xy: np.ndarray, w: int, h: int) -> np.ndarray:
    """Normalize pixel coords to the MotionBERT/AGFormer convention.

    x -> 2x/w - 1,  y -> 2y/w - h/w. Both axes are scaled by width so aspect
    ratio is preserved and x lands in [-1, 1]. Operates on the xy of the last
    channel-dim; a confidence channel (index 2) is passed through untouched.
    """
    kp = np.asarray(kp_xy, dtype=np.float32).copy()
    kp[..., 0] = kp[..., 0] / w * 2.0 - 1.0
    kp[..., 1] = kp[..., 1] / w * 2.0 - (h / w)
    return kp


class JointHold:
    """Per-joint last-valid-value hold, keyed by confidence.

    This is the missing-legs safety valve: when a joint's confidence drops below
    `thresh`, its position is replaced by the last frame where it *was* confident
    (or the pelvis, if it has never been seen), so the tensor fed to the lifter
    is always finite and free of teleporting garbage. It also reports a boolean
    reliability mask so the renderer can dim/skip untrusted joints.
    """

    def __init__(self, n_joints: int = 17, thresh: float = 0.3):
        self.n = n_joints
        self.thresh = thresh
        self._last_rel: Optional[np.ndarray] = None  # (n, 2) last confident (xy - pelvis)
        self._last_pelvis: Optional[np.ndarray] = None # (2,) last confident pelvis

    def reset(self) -> None:
        self._last_rel = None
        self._last_pelvis = None

    def update(self, kp: np.ndarray):
        """kp: (17, 3) = (x, y, conf) in any coordinate space.

        Returns (held_xy_conf, reliable_mask):
          held_xy_conf: (17, 3) with untrusted joints replaced by last-valid xy
                        (conf preserved as-is so downstream can still see it low)
          reliable_mask: (17,) bool, True where conf >= thresh this frame
        """
        kp = np.asarray(kp, dtype=np.float32)
        if kp.shape != (self.n, 3):
            raise ValueError(f"expected ({self.n}, 3), got {kp.shape}")
        reliable = kp[:, 2] >= self.thresh

        if self._last_rel is None:
            # First ever frame: seed unseen joints at the pelvis (joint 0) so the
            # skeleton is at least connected rather than at the origin.
            self._last_rel = np.zeros((self.n, 2), dtype=np.float32)
            self._last_pelvis = kp[0, :2].copy()
            self._last_rel[reliable] = kp[reliable, :2] - self._last_pelvis

        # Joint 0 is pelvis in H36M. Use its current position if reliable,
        # otherwise fall back to the last known pelvis.
        if reliable[0]:
            curr_pelvis = kp[0, :2].copy()
            self._last_pelvis = curr_pelvis
        else:
            curr_pelvis = self._last_pelvis

        held = kp.copy()

        # Trusted joints update their relative offset from the current pelvis.
        self._last_rel[reliable] = kp[reliable, :2] - curr_pelvis

        # Untrusted joints are rebuilt from their last known relative offset
        # plus the current pelvis, so they track the body's movement.
        held[~reliable, :2] = curr_pelvis + self._last_rel[~reliable]
        return held, reliable


class SlidingWindow:
    """Causal frame buffer for temporal lifters.

    Holds the last `size` frames of H36M 2D keypoints. Until it fills, it pads by
    repeating the earliest frame so a (1, T, 17, C) tensor is available from the
    very first frame (warmup quality is lower but the demo is interactive
    immediately). The lifter output for the *last* frame is what we render.
    """

    def __init__(self, size: int, n_joints: int = 17, channels: int = 3):
        if size < 1:
            raise ValueError("window size must be >= 1")
        self.size = size
        self.n = n_joints
        self.c = channels
        self._buf: deque = deque(maxlen=size)

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def warm(self) -> bool:
        """True once the buffer has seen `size` real frames."""
        return len(self._buf) >= self.size

    def reset(self) -> None:
        self._buf.clear()

    def push(self, frame: np.ndarray) -> np.ndarray:
        """Append a (17, C) frame; return the (1, size, 17, C) padded stack."""
        frame = np.asarray(frame, dtype=np.float32)
        if frame.shape != (self.n, self.c):
            raise ValueError(f"expected ({self.n}, {self.c}), got {frame.shape}")
        self._buf.append(frame)
        frames = list(self._buf)
        if len(frames) < self.size:
            pad = [frames[0]] * (self.size - len(frames))
            frames = pad + frames
        return np.stack(frames, axis=0)[None, ...]  # (1, size, 17, C)


# Fixed H36M camera->world quaternion used by the MotionAGFormer/MHFormer demos.
# The lifter outputs poses in camera coordinates; rotating by this puts the body
# upright with +Z as the up axis (matching the repo's visualization).
H36M_CAM_ROT = np.array(
    [0.1407056450843811, -0.1500701755285263, -0.755240797996521,
     0.6223280429840088], dtype=np.float32)


def _qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors v (N,3) by quaternion q (4,) in (w,x,y,z) order."""
    qvec = np.broadcast_to(q[1:], v.shape)
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2.0 * (q[0] * uv + uuv)


def camera_to_world(pose3d: np.ndarray, q: np.ndarray = H36M_CAM_ROT) -> np.ndarray:
    """Rotate a (17,3) camera-frame pose into the world frame (Z up)."""
    return _qrot(np.asarray(q, dtype=np.float32),
                 np.asarray(pose3d, dtype=np.float32))
