import sys
import types
import importlib.machinery
if 'mmcv._ext' not in sys.modules:
    # mmdet/mmpose's import chain pulls in several compiled mmcv ops
    # (active_rotated_filter, assign_score_withk, roi_align, ...) purely as a
    # side effect of module-level imports in code paths we never call (e.g.
    # DetInferencer's eval/metrics machinery) -- none of them are reachable
    # by the actual RTMDet + ViTPose++ top-down inference this module runs,
    # so a no-op is fine for those. `nms` is the one exception: RTMDet's
    # detection head genuinely calls it for post-processing, so it needs a
    # real implementation -- backed by torchvision's compiled NMS op rather
    # than mmcv's, since we use mmcv-lite (no compiled C++/CUDA ops) to
    # avoid needing a build toolchain.
    def _stub_nms(bboxes, scores, iou_threshold, offset=0, **kwargs):
        import torchvision
        if offset:
            bboxes = bboxes.clone()
            bboxes[:, 2:] += offset
        return torchvision.ops.nms(bboxes, scores, float(iou_threshold))

    def _mmcv_ext_getattr(name):
        # Let dunder lookups (__file__, __path__, __spec__, ...) raise
        # normally -- returning a fake value for those confuses stdlib
        # `inspect`/importlib introspection (e.g. mmengine's Registry
        # scans sys.modules and calls inspect.getabsfile() on each one).
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        if name == 'nms':
            return _stub_nms
        return lambda *args, **kwargs: None

    _mmcv_ext_stub = types.ModuleType('mmcv._ext')
    _mmcv_ext_stub.__getattr__ = _mmcv_ext_getattr
    # importlib.util.find_spec raises ValueError for an already-imported
    # module whose __spec__ is None (mmengine's mmcv_full_available() hits
    # this via pkgutil.find_loader) -- give it a real, inert spec instead.
    _mmcv_ext_stub.__spec__ = importlib.machinery.ModuleSpec('mmcv._ext', loader=None)
    sys.modules['mmcv._ext'] = _mmcv_ext_stub
"""
2D Pose Estimation Module

Provides an interface to run 2D human/robot pose estimation on video frames
using MMPose with a ViTPose++ backbone.

The module:
1. Initialises MMPose with a ViTPose++ model (downloaded on first use).
2. Runs a top-down pose estimation pipeline:
   - Detect the subject with an MMDet detector (RTMDet).
   - Estimate 2D keypoints for each detected instance.
3. Returns per-frame keypoints and confidence scores as numpy arrays.

The COCO-WholeBody keypoint format is used (133 keypoints), but the
caller can select a subset via the `keypoint_indices` parameter.

Standard COCO body keypoints (17):
    0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
    5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
    9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
    13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
"""

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Number of COCO body keypoints (the standard 17 used for body pose)
COCO_BODY_KEYPOINTS = 17

# Human-readable names for the 17 COCO body keypoints
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Skeleton connections for visualisation
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # upper body
    (5, 11), (6, 12), (11, 12),                 # torso
    (11, 13), (13, 15), (12, 14), (14, 16),     # lower body
]


@dataclass
class PoseResult:
    """Result of pose estimation on a single video."""
    keypoints: np.ndarray       # (T, J, 2) — 2D coordinates per frame
    confidence: np.ndarray      # (T, J) — confidence scores
    num_frames: int = 0
    num_joints: int = COCO_BODY_KEYPOINTS
    fps: float = 0.0
    frame_width: int = 0
    frame_height: int = 0


def _try_import_mmpose():
    """Lazily import MMPose components. Returns None if unavailable."""
    try:
        from mmpose.apis import init_model, inference_topdown
        from mmpose.utils import register_all_modules as register_pose_modules
        return init_model, inference_topdown, register_pose_modules
    except ImportError:
        return None, None, None


def _try_import_mmdet():
    """Lazily import MMDet components. Returns None if unavailable."""
    try:
        from mmdet.apis import init_detector, inference_detector
        from mmdet.utils import register_all_modules as register_det_modules
        return init_detector, inference_detector, register_det_modules
    except ImportError:
        return None, None, None


# ── Model configuration ─────────────────────────────────────────────────────

# Default ViTPose++ config and checkpoint (downloaded on first use)
# Using the small variant for reasonable CPU inference speed.
_DEFAULT_POSE_CONFIG = "td-hm_ViTPose-small_8xb64-210e_coco-256x192"
_DEFAULT_DET_CONFIG = "rtmdet_m_8xb32-300e_coco"

# Model cache directory
_MODEL_CACHE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "models")


class PoseEstimator:
    """
    Top-down 2D pose estimator using MMPose + MMDet.

    Usage:
        estimator = PoseEstimator()
        result = estimator.estimate_from_video("video.mp4")
    """

    def __init__(
        self,
        device: str = "cpu",
        det_score_threshold: float = 0.5,
        keypoint_indices: Optional[List[int]] = None,
    ):
        """
        Args:
            device: "cpu" or "cuda:0".
            det_score_threshold: Minimum detection confidence to run pose on.
            keypoint_indices: Which keypoint indices to keep from the model output.
                              None means keep the first COCO_BODY_KEYPOINTS (17).
        """
        self.device = device
        self.det_score_threshold = det_score_threshold
        self.keypoint_indices = keypoint_indices

        self._detector = None
        self._pose_model = None
        self._initialised = False

    def _lazy_init(self):
        """Initialise models on first use to avoid loading at import time."""
        if self._initialised:
            return

        init_det, self._infer_det, register_det = _try_import_mmdet()
        init_pose, self._infer_pose, register_pose = _try_import_mmpose()

        if init_det is None or init_pose is None:
            raise ImportError(
                "MMPose and/or MMDet are not installed. "
                "Install with: pip install mmpose mmdet mmengine mmcv-lite"
            )

        register_det()
        register_pose()

        os.makedirs(_MODEL_CACHE, exist_ok=True)

        # Initialise detector
        logger.info("Loading RTMDet detector …")
        try:
            from mmdet.utils import get_test_pipeline_cfg
        except ImportError:
            pass

        det_config = self._resolve_config("mmdet", _DEFAULT_DET_CONFIG)
        det_checkpoint = self._resolve_checkpoint("mmdet", _DEFAULT_DET_CONFIG)
        self._detector = init_det(det_config, det_checkpoint, device=self.device)

        # Initialise pose model
        logger.info("Loading ViTPose++ pose model …")
        pose_config = self._resolve_config("mmpose", _DEFAULT_POSE_CONFIG)
        pose_checkpoint = self._resolve_checkpoint("mmpose", _DEFAULT_POSE_CONFIG)
        self._pose_model = init_pose(pose_config, pose_checkpoint, device=self.device)

        self._initialised = True
        logger.info("Pose estimation models loaded successfully.")

    @staticmethod
    def _resolve_config(package: str, model_name: str) -> str:
        """Resolve model config path from the installed package."""
        try:
            import importlib
            mod = importlib.import_module(package)
            pkg_dir = os.path.dirname(mod.__file__)
            # MMPose/MMDet store configs in .mim/configs/
            config_dir = os.path.join(pkg_dir, ".mim", "configs")
            if not os.path.isdir(config_dir):
                config_dir = os.path.join(pkg_dir, "configs")

            # Search for the config file
            for root, dirs, files in os.walk(config_dir):
                for f in files:
                    if model_name in f and f.endswith(".py"):
                        return os.path.join(root, f)

            raise FileNotFoundError(f"Config for {model_name} not found in {config_dir}")
        except Exception as e:
            raise FileNotFoundError(f"Cannot resolve config for {package}/{model_name}: {e}")

    @staticmethod
    def _resolve_checkpoint(package: str, model_name: str) -> str:
        """
        Resolve or download the model checkpoint.
        For now, returns the model zoo URL — MMPose/MMDet will download it
        automatically on first use.
        """
        # The checkpoint URLs for our default models
        checkpoints = {
            "rtmdet_m_8xb32-300e_coco": (
                "https://download.openmmlab.com/mmdetection/v3.0/"
                "rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth"
            ),
            "td-hm_ViTPose-small_8xb64-210e_coco-256x192": (
                "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/"
                "coco/td-hm_ViTPose-small_8xb64-210e_coco-256x192-62d7a712_20230314.pth"
            ),
        }
        return checkpoints.get(model_name, "")

    def estimate_from_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        skip_frames: int = 0,
    ) -> PoseResult:
        """
        Run 2D pose estimation on every frame of a video.

        Args:
            video_path: Path to an MP4 video file.
            max_frames: Optional cap on number of frames to process.
            skip_frames: Number of initial frames to skip (for synchronization).

        Returns:
            PoseResult with keypoints (T, J, 2) and confidence (T, J).
        """
        self._lazy_init()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if max_frames:
            total_frames = min(total_frames - skip_frames, max_frames)
        else:
            total_frames = total_frames - skip_frames

        if skip_frames > 0:
            logger.info("Sync: Skipping first %d frames of video", skip_frames)
            for _ in range(skip_frames):
                cap.read()

        n_joints = len(self.keypoint_indices) if self.keypoint_indices else COCO_BODY_KEYPOINTS

        all_keypoints = []
        all_confidence = []

        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            from mmengine.registry import DefaultScope

            # Step 1: Detect subjects
            with DefaultScope.overwrite_default_scope('mmdet'):
                det_result = self._infer_det(self._detector, frame)
            det_instances = det_result.pred_instances

            # Filter to "person" class (class 0 in COCO) and above threshold
            person_mask = (det_instances.labels == 0) & (det_instances.scores >= self.det_score_threshold)
            bboxes = det_instances.bboxes[person_mask].cpu().numpy()

            if len(bboxes) == 0:
                # No detection: fill with zeros
                all_keypoints.append(np.zeros((n_joints, 2), dtype=np.float32))
                all_confidence.append(np.zeros(n_joints, dtype=np.float32))
                continue

            # Step 2: Run pose estimation (take the highest-confidence detection)
            scores = det_instances.scores[person_mask].cpu().numpy()
            best_idx = np.argmax(scores)
            best_bbox = bboxes[best_idx]

            with DefaultScope.overwrite_default_scope('mmpose'):
                pose_results = self._infer_pose(
                    self._pose_model,
                    frame,
                    bboxes=best_bbox[None],  # Pass as (1, 4) numpy array
                )

            if pose_results and len(pose_results) > 0:
                pred = pose_results[0].pred_instances
                kpts = pred.keypoints[0]                   # (K, 2)
                scores = pred.keypoint_scores[0]           # (K,)

                # Select keypoint subset
                if self.keypoint_indices:
                    kpts = kpts[self.keypoint_indices]
                    scores = scores[self.keypoint_indices]
                else:
                    kpts = kpts[:COCO_BODY_KEYPOINTS]
                    scores = scores[:COCO_BODY_KEYPOINTS]

                all_keypoints.append(kpts.astype(np.float32))
                all_confidence.append(scores.astype(np.float32))
            else:
                all_keypoints.append(np.zeros((n_joints, 2), dtype=np.float32))
                all_confidence.append(np.zeros(n_joints, dtype=np.float32))

            if (frame_idx + 1) % 50 == 0:
                logger.info("Processed %d / %d frames", frame_idx + 1, total_frames)

        cap.release()

        keypoints_arr = np.stack(all_keypoints)  # (T, J, 2)
        confidence_arr = np.stack(all_confidence) # (T, J)

        logger.info(
            "Pose estimation complete: %d frames, %d joints, video=%.1f fps",
            len(all_keypoints), n_joints, fps,
        )

        return PoseResult(
            keypoints=keypoints_arr,
            confidence=confidence_arr,
            num_frames=len(all_keypoints),
            num_joints=n_joints,
            fps=fps,
            frame_width=width,
            frame_height=height,
        )


def estimate_stereo_poses(
    left_video_path: str,
    right_video_path: str,
    device: str = "cpu",
    max_frames: Optional[int] = None,
) -> Tuple[PoseResult, PoseResult]:
    """
    Convenience function to run 2D pose estimation on both cameras of a
    stereo pair. Auto-synchronizes using audio cross-correlation.

    Returns (left_result, right_result).
    """
    from src.processing.sync import get_video_offset
    
    estimator = PoseEstimator(device=device)

    # Compute offset
    cap_l = cv2.VideoCapture(left_video_path)
    cap_r = cv2.VideoCapture(right_video_path)
    fps_l = cap_l.get(cv2.CAP_PROP_FPS) or 30.0
    fps_r = cap_r.get(cv2.CAP_PROP_FPS) or 30.0
    cap_l.release()
    cap_r.release()

    offset_seconds = get_video_offset(left_video_path, right_video_path)
    skip_l = 0
    skip_r = 0
    if offset_seconds > 0:
        skip_l = int(offset_seconds * fps_l)
    elif offset_seconds < 0:
        skip_r = int(abs(offset_seconds) * fps_r)

    logger.info("Estimating poses on LEFT camera …")
    left_result = estimator.estimate_from_video(left_video_path, max_frames, skip_frames=skip_l)

    logger.info("Estimating poses on RIGHT camera …")
    right_result = estimator.estimate_from_video(right_video_path, max_frames, skip_frames=skip_r)

    # Align the sequences to be the same length (clip the longer one from the end)
    min_frames = min(left_result.num_frames, right_result.num_frames)
    if left_result.num_frames > min_frames:
        left_result.keypoints = left_result.keypoints[:min_frames]
        left_result.confidence = left_result.confidence[:min_frames]
        left_result.num_frames = min_frames
    if right_result.num_frames > min_frames:
        right_result.keypoints = right_result.keypoints[:min_frames]
        right_result.confidence = right_result.confidence[:min_frames]
        right_result.num_frames = min_frames

    return left_result, right_result
