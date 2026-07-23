"""MotionAGFormer-S 2D->3D lifter, wrapped behind a small, swappable interface.

The rest of the app only needs `Lifter.lift(window) -> (17,3)`. This module owns
the messy details: vendoring the MotionAGFormer source from third_party/, the
exact -S hyperparameters, stripping the DataParallel `module.` prefix, and
picking a working torch device.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, "..", "..", "third_party", "MotionAGFormer")
_CKPT = os.path.join(_HERE, "..", "..", "checkpoints", "motionagformer-s-h36m.pth.tr")

# MotionAGFormer-S (H36M), from configs/h36m/MotionAGFormer-small.yaml
S_CONFIG = dict(
    n_layers=26, dim_in=3, dim_feat=64, dim_rep=512, dim_out=3, mlp_ratio=4,
    num_heads=8, n_frames=81, use_temporal_similarity=True, neighbour_num=2,
    use_tcn=False, graph_only=False, hierarchical=False,
    use_adaptive_fusion=True, use_layer_scale=True,
    layer_scale_init_value=1e-5, qkv_bias=False,
)
N_FRAMES = S_CONFIG["n_frames"]


def resolve_device(prefer: str = "auto") -> torch.device:
    """Resolve a device string to a working torch.device.

    'auto' picks the best available (cuda > mps > cpu), so the same code runs on
    an NVIDIA Ubuntu box, an Apple Silicon Mac, or a plain CPU without edits. An
    explicit choice is honored when available and otherwise falls back to CPU.
    """
    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available()
    if prefer == "auto":
        if has_cuda:
            return torch.device("cuda")
        if has_mps:
            return torch.device("mps")
        return torch.device("cpu")
    if prefer == "cuda" and has_cuda:
        return torch.device("cuda")
    if prefer == "mps" and has_mps:
        return torch.device("mps")
    return torch.device("cpu")


class Lifter:
    """Loads MotionAGFormer-S and lifts a window of 2D H36M keypoints to 3D."""

    def __init__(self, checkpoint: str = _CKPT, device: str = "auto"):
        if _REPO not in sys.path:
            sys.path.insert(0, _REPO)
        # Allow MPS to fall back to CPU for any unimplemented op rather than error.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

        from model.MotionAGFormer import MotionAGFormer  # noqa: E402

        self.device = resolve_device(device)
        self.model = MotionAGFormer(**S_CONFIG)
        self.load_checkpoint(checkpoint)
        
    def load_checkpoint(self, checkpoint: str) -> None:
        if hasattr(self, "_current_ckpt") and self._current_ckpt == checkpoint:
            return
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        ckpt_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        sd = ckpt_data["model"] if isinstance(ckpt_data, dict) and "model" in ckpt_data else ckpt_data
        # Strip the DataParallel prefix the released weights were saved with.
        sd = {k[len("module."):] if k.startswith("module.") else k: v
              for k, v in sd.items()}
        self.model.load_state_dict(sd, strict=True)
        self.model.eval().to(self.device)
        self._current_ckpt = checkpoint

    @property
    def n_frames(self) -> int:
        return N_FRAMES

    @torch.no_grad()
    def lift(self, window: np.ndarray) -> np.ndarray:
        """window: (1, T, 17, 3) normalized 2D+conf -> (17, 3) root-relative 3D.

        Returns the 3D pose for the *last* (most recent) frame in the window,
        which is what makes this a causal, realtime-friendly lifter.
        """
        window = np.asarray(window, dtype=np.float32)
        if window.ndim != 4 or window.shape[2:] != (17, 3):
            raise ValueError(f"expected (1,T,17,3), got {window.shape}")
        x = torch.from_numpy(window).to(self.device)
        out = self.model(x)                      # (1, T, 17, 3)
        
        # Scale back to meters
        out_m = out * (1000 / 2) / 1000.0  # which is just out * 0.5
        
        pose = out_m[0, -1].cpu().numpy()         # last frame
        pose = pose - pose[0]                     # ensure root-relative
        return pose.astype(np.float32)


def _selftest():
    lifter = Lifter()
    print(f"Lifter loaded MotionAGFormer-S on {lifter.device}, T={lifter.n_frames}")
    win = np.zeros((1, N_FRAMES, 17, 3), dtype=np.float32)
    win[..., 2] = 1.0  # full confidence
    # a tiny bit of structure so it isn't degenerate
    win[..., 0, :2] = 0.0
    win[..., 10, 1] = 0.5   # head up
    pose = lifter.lift(win)
    print("output pose shape:", pose.shape, "| finite:", np.isfinite(pose).all())
    print("pelvis (should be ~0):", np.round(pose[0], 4))
    print("head z-range sample:", np.round(pose[10], 4))


if __name__ == "__main__":
    _selftest()
