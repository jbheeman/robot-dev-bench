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
from typing import List, Optional, Tuple, Callable

import cv2
import numpy as np

from . import stereo_core as sc
from .lifter import Lifter

G1_BONE_LENGTHS = {
    1: (0, 0.1212), 2: (1, 0.3409), 3: (2, 0.3000), # left leg
    4: (0, 0.1212), 5: (4, 0.3409), 6: (5, 0.3000), # right leg
    7: (0, 0.1459), 8: (7, 0.1459), 9: (8, 0.1459), 10: (9, 0.1459), # spine -> head
    11: (8, 0.1002), 12: (11, 0.1929), 13: (12, 0.1929), # left arm
    14: (8, 0.1002), 15: (14, 0.1929), 16: (15, 0.1929), # right arm
}

def apply_g1_morphology(pose: np.ndarray) -> np.ndarray:
    """
    Given a (17, 3) pose, rescale the bones to match Unitree G1 proportions
    while preserving the joint angles predicted by the model.
    """
    new_pose = pose.copy()
    
    # Traverse kinematic tree from root (0) to leaves (1..16 are topologically sorted)
    for child in range(1, 17):
        parent, length = G1_BONE_LENGTHS[child]
        
        # Calculate direction vector from ORIGINAL pose
        original_vec = pose[child] - pose[parent]
        norm = np.linalg.norm(original_vec)
        
        if norm > 1e-6:
            scaled_vec = original_vec / norm * length
        else:
            scaled_vec = np.zeros(3)
            
        # Attach the scaled vector to the NEW parent position
        new_pose[child] = new_pose[parent] + scaled_vec
        
    return new_pose

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
class StereoConfig:
    """Configuration for stereo camera triangulation."""
    baseline_mm: float = 60.0       # Distance between cameras in mm
    focal_length_px: float = 800.0  # Focal length in pixels


@dataclass
class PoseResult:
    """Result of pose estimation on a single video."""
    keypoints: np.ndarray       # (T, J, 2) — 2D coordinates per frame
    confidence: np.ndarray      # (T, J) — confidence scores
    poses_3d: Optional[np.ndarray] = None # (T, J, 3) — 3D coordinates per frame
    num_frames: int = 0
    num_joints: int = COCO_BODY_KEYPOINTS
    fps: float = 0.0
    frame_width: int = 0
    frame_height: int = 0
    stereo_used: bool = False  # Whether stereo triangulation was applied


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
# Using the huge variant for maximum accuracy on non-humanoid morphologies.
_DEFAULT_POSE_CONFIG = "td-hm_ViTPose-huge_8xb64-210e_coco-256x192"
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
        self._lifter = None
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
        pose_config_name = _DEFAULT_POSE_CONFIG
        pose_checkpoint = self._resolve_checkpoint("mmpose", pose_config_name)

        pose_config = self._resolve_config("mmpose", pose_config_name)
        self._pose_model = init_pose(
            pose_config, 
            pose_checkpoint, 
            device=self.device,
            cfg_options={'model.backbone.init_cfg': None}
        )

        # Apply TTA (Test-Time Augmentation) for pose model if supported
        if hasattr(self._pose_model, 'cfg'):
            if 'test_dataloader' in self._pose_model.cfg:
                dataset_cfg = self._pose_model.cfg.test_dataloader.get('dataset', {})
                if 'pipeline' in dataset_cfg:
                    # Enabling flip_test implicitly if possible, but usually MMPose 1.x
                    # requires it to be set in the config. We will rely on model default or manually set it.
                    pass
        # MMPose 1.x handles TTA via model.cfg.model.test_cfg.flip_test
        try:
            if hasattr(self._pose_model.cfg, 'model') and hasattr(self._pose_model.cfg.model, 'test_cfg'):
                self._pose_model.cfg.model.test_cfg.flip_test = True
                logger.info("Enabled flip_test (TTA) for ViTPose++")
        except Exception:
            pass

        # Initialise lifter model
        logger.info("Loading MotionAGFormer lifter model …")
        from .lifter import Lifter
        self._lifter = Lifter(device=self.device)

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
            "td-hm_ViTPose-base_8xb64-210e_coco-256x192": (
                "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/"
                "coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192-216eae50_20230314.pth"
            ),
            "td-hm_ViTPose-huge_8xb64-210e_coco-256x192": (
                "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/"
                "coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth"
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
        progress_callback: Optional[Callable[[float, str], None]] = None,
        stereo: bool = False,
        stereo_config: Optional[StereoConfig] = None,
        morphology: str = "g1",
        pipeline_mode: str = "3d",
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
        
        from .lifter import _HERE
        if morphology == "g1_retrained":
            ckpt = os.path.join(_HERE, "..", "..", "checkpoints", "motionagformer-s-g1.pth.tr")
        elif morphology == "g1":
            ckpt = os.path.join(_HERE, "..", "..", "checkpoints", "motionagformer-b-g1", "best_epoch.pth.tr")
        else:
            ckpt = os.path.join(_HERE, "..", "..", "checkpoints", "motionagformer-s-h36m.pth.tr")
        self._lifter.load_checkpoint(ckpt)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        raw_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # In stereo mode the raw frame is side-by-side; each half is the real width.
        if stereo:
            width = raw_width // 2
            logger.info("Stereo mode enabled: splitting %dx%d into 2x %dx%d",
                        raw_width, height, width, height)
        else:
            width = raw_width

        if stereo_config is None:
            stereo_config = StereoConfig()

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
        
        # State for causal 3D lifting
        if pipeline_mode == "3d":
            from .lift_core import SlidingWindow, JointHold, coco2h36m, normalize_screen
            window = SlidingWindow(size=self._lifter.n_frames)
            joint_hold = JointHold()
        all_poses_3d = []
        
        # Per-frame stereo pelvis depths (mm). Used to scale the 3D output.
        stereo_pelvis_depths: list[Optional[float]] = []

        for frame_idx in range(total_frames):
            ret, raw_frame = cap.read()
            if not ret:
                break

            # ── Stereo split ──
            if stereo:
                frame_left, frame_right = sc.split_stereo_frame(raw_frame)
                frame = frame_left  # left view is the "main" view for 2D
            else:
                frame = raw_frame
                frame_right = None

            if progress_callback:
                progress_callback(frame_idx / max(total_frames, 1), f"Processing frame {frame_idx}/{total_frames} (2D pose)")

            from mmengine.registry import DefaultScope

            # Step 1: Detect subjects
            with DefaultScope.overwrite_default_scope('mmdet'):
                det_result = self._infer_det(self._detector, frame)
            det_instances = det_result.pred_instances

            # Humanoid robots often yield very low object detection confidence or are misclassified.
            # We first look for ANY 'person' detection, even with extremely low confidence.
            person_mask = (det_instances.labels == 0) & (det_instances.scores >= 0.15)
            bboxes = det_instances.bboxes[person_mask].cpu().numpy()
            scores = det_instances.scores[person_mask].cpu().numpy()

            # Merge adjacent or overlapping boxes. Because the humanoid might not look perfectly human,
            # RTMDet sometimes splits it into a "top half" and "bottom half" person detection.
            # We union any boxes that are close to each other.
            if len(bboxes) > 1:
                merged = []
                used = np.zeros(len(bboxes), dtype=bool)
                for i in range(len(bboxes)):
                    if used[i]: continue
                    c_box = bboxes[i].copy()
                    used[i] = True
                    while True:
                        added = False
                        for j in range(len(bboxes)):
                            if used[j]: continue
                            # Expand box2 slightly (e.g., 50px) to merge near-touches
                            eb = [bboxes[j][0]-50, bboxes[j][1]-50, bboxes[j][2]+50, bboxes[j][3]+50]
                            if max(c_box[0], eb[0]) <= min(c_box[2], eb[2]) and max(c_box[1], eb[1]) <= min(c_box[3], eb[3]):
                                c_box[0] = min(c_box[0], bboxes[j][0])
                                c_box[1] = min(c_box[1], bboxes[j][1])
                                c_box[2] = max(c_box[2], bboxes[j][2])
                                c_box[3] = max(c_box[3], bboxes[j][3])
                                used[j] = True
                                added = True
                        if not added: break
                    merged.append(c_box)
                bboxes = np.array(merged)
                # Re-compute scores (just take max for now since we merge)
                scores = np.array([1.0] * len(bboxes), dtype=np.float32)

            if len(bboxes) == 0:
                # If no 'person' is found, look at all detections >= 0.15 as a fallback
                any_mask = (det_instances.scores >= 0.15)
                candidate_bboxes = det_instances.bboxes[any_mask].cpu().numpy()
                
                if len(candidate_bboxes) > 0:
                    # Pick the largest bounding box by area, as the robot is the main subject
                    # (Avoids picking high-confidence small background objects like a cup or chair)
                    areas = (candidate_bboxes[:, 2] - candidate_bboxes[:, 0]) * (candidate_bboxes[:, 3] - candidate_bboxes[:, 1])
                    best_idx = np.argmax(areas)
                    bboxes = candidate_bboxes[best_idx:best_idx+1]
                    scores = np.array([0.5], dtype=np.float32) # fake score to pass through

            if len(bboxes) > 0:
                h_img, w_img = frame.shape[:2]
                for i in range(len(bboxes)):
                    bx1, by1, bx2, by2 = bboxes[i]
                    bh = by2 - by1
                    bw = bx2 - bx1
                    
                    # Aggressively expand the bounding box vertically!
                    # Expand 60% upwards, 120% downwards, and 20% horizontally
                    new_y1 = max(0, by1 - 0.6 * bh)
                    new_y2 = min(h_img, by2 + 1.2 * bh)
                    new_x1 = max(0, bx1 - 0.2 * bw)
                    new_x2 = min(w_img, bx2 + 0.2 * bw)
                    
                    bboxes[i] = [new_x1, new_y1, new_x2, new_y2]

            if len(bboxes) == 0:
                # Still no detection: fill with zeros
                all_keypoints.append(np.zeros((n_joints, 2), dtype=np.float32))
                all_confidence.append(np.zeros(n_joints, dtype=np.float32))
                all_poses_3d.append(np.zeros((COCO_BODY_KEYPOINTS, 3), dtype=np.float32))
                stereo_pelvis_depths.append(None)
                continue

            # Step 2: Run pose estimation (take the highest-confidence detection)
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

                # ── Stereo: run 2D on the right frame & triangulate pelvis ──
                pelvis_depth_mm: Optional[float] = None
                if stereo and frame_right is not None and pipeline_mode == "3d":
                    # Reuse left frame's bounding box to skip duplicate object detection
                    with DefaultScope.overwrite_default_scope('mmpose'):
                        pose_r = self._infer_pose(
                            self._pose_model,
                            frame_right,
                            bboxes=best_bbox[None],
                        )
                    if pose_r and len(pose_r) > 0:
                            pred_r = pose_r[0].pred_instances
                            kpts_r = pred_r.keypoints[0][:COCO_BODY_KEYPOINTS]
                            scores_kp_r = pred_r.keypoint_scores[0][:COCO_BODY_KEYPOINTS]
                            # Build COCO-format (17,3) arrays for match_pelvis
                            coco_left_17 = np.concatenate(
                                [kpts.astype(np.float32),
                                 scores[:COCO_BODY_KEYPOINTS, None].astype(np.float32)],
                                axis=1)
                            coco_right_17 = np.concatenate(
                                [kpts_r.astype(np.float32),
                                 scores_kp_r[:, None].astype(np.float32)],
                                axis=1)
                            pelvis_l, pelvis_r_pt, pelvis_ok = sc.match_pelvis(
                                coco_left_17, coco_right_17, conf_thresh=0.3)
                            if pelvis_ok:
                                d = sc.triangulate_depth(
                                    pelvis_l[0], pelvis_r_pt[0],
                                    stereo_config.focal_length_px,
                                    stereo_config.baseline_mm)
                                if d != float('inf'):
                                    pelvis_depth_mm = d

                stereo_pelvis_depths.append(pelvis_depth_mm)

                # ── Convert COCO keypoints to H36M for the pose lifter ──
                if pipeline_mode == "3d":
                    coco_conf = scores[:COCO_BODY_KEYPOINTS, None]
                    coco_17 = np.concatenate([kpts[:COCO_BODY_KEYPOINTS], coco_conf], axis=-1)
                    
                    # Convert format and apply JointHold memory
                    h36m_17 = coco2h36m(coco_17)
                    held_h36m, _ = joint_hold.update(h36m_17)
                    
                    # Normalize and push to causal sliding window
                    norm_h36m = normalize_screen(held_h36m, width, height)
                    win_tensor = window.push(norm_h36m)
                    
                    # Run the lifter
                    pose_3d = self._lifter.lift(win_tensor)
                    
                    if frame_idx == 0:
                        logger.info(f"First 3D pose (H36M): {pose_3d}")
                    all_poses_3d.append(pose_3d)
            else:
                all_keypoints.append(np.zeros((n_joints, 2), dtype=np.float32))
                all_confidence.append(np.zeros(n_joints, dtype=np.float32))
                stereo_pelvis_depths.append(None)
                if pipeline_mode == "3d":
                    all_poses_3d.append(np.zeros((COCO_BODY_KEYPOINTS, 3), dtype=np.float32))

            if (frame_idx + 1) % 50 == 0:
                logger.info("Processed %d / %d frames", frame_idx + 1, total_frames)

        cap.release()

        if len(all_keypoints) == 0:
            logger.warning("No valid frames or 3D poses extracted from video: %s", video_path)
            return PoseResult(
                keypoints=np.zeros((0, n_joints, 2), dtype=np.float32),
                confidence=np.zeros((0, n_joints), dtype=np.float32),
                poses_3d=np.zeros((0, COCO_BODY_KEYPOINTS, 3), dtype=np.float32),
                num_frames=0,
                num_joints=n_joints,
                fps=fps,
                frame_width=width,
                frame_height=height,
                stereo_used=stereo,
            )

        if pipeline_mode == "2d":
            # In 2D mode we completely skip 3D lifting and smoothing, returning empty 3D structures.
            return PoseResult(
                keypoints=np.stack(all_keypoints),
                confidence=np.stack(all_confidence),
                poses_3d=np.zeros((len(all_keypoints), COCO_BODY_KEYPOINTS, 3), dtype=np.float32),
                num_frames=len(all_keypoints),
                num_joints=n_joints,
                fps=fps,
                frame_width=width,
                frame_height=height,
                stereo_used=stereo,
            )

        poses_3d_arr = np.stack(all_poses_3d) # (T, 17, 3) in H36M format

        # ── Unproject X and Y using stereo depth or dynamic estimation ──
        # The AI output has X and Y as tangent angles, and Z as root-relative depth in meters.
        # We need absolute depth to unproject X and Y into true metric space.
        if stereo and any(d is not None for d in stereo_pelvis_depths):
            valid_depths = [d for d in stereo_pelvis_depths if d is not None]
            median_depth_mm = float(np.median(valid_depths))
            logger.info("Stereo fusion: anchoring 3D poses to triangulated median depth: %.1f mm", median_depth_mm)
            depth_m = median_depth_mm / 1000.0
            
            poses_3d_arr[:, :, 0] *= depth_m
            poses_3d_arr[:, :, 1] *= depth_m
        else:
            # Monocular fallback: estimate depth dynamically based on torso length
            logger.info("Monocular mode: dynamically estimating depth from torso length.")
            # H36M pelvis is 0, neck is 8.
            # Use 0.5m for human torso, 0.2918m for G1 retrained torso
            assumed_torso_m = 0.2918 if morphology == "g1_retrained" else 0.5
            
            for t in range(poses_3d_arr.shape[0]):
                torso_tangent = np.linalg.norm(poses_3d_arr[t, 8, :2] - poses_3d_arr[t, 0, :2])
                dynamic_depth = assumed_torso_m / max(torso_tangent, 1e-4)
                poses_3d_arr[t, :, 0] *= dynamic_depth
                poses_3d_arr[t, :, 1] *= dynamic_depth
                
            # Set a generic absolute depth for visualization
            median_depth_mm = 3000.0

        if morphology == "g1":
            logger.info("Applying Unitree G1 morphology (rescaling limb lengths)...")
            for t in range(poses_3d_arr.shape[0]):
                poses_3d_arr[t] = apply_g1_morphology(poses_3d_arr[t])

        # ── Post-processing Smoothing ──
        try:
            from .filter import TelemetryFilter
            logger.info("Applying Butterworth low-pass filter to 3D trajectories (cutoff=5Hz)...")
            # Create a 5Hz cutoff lowpass filter for smoothing the 3D coordinates.
            # Assuming typical walk cycle frequencies are ~1-2Hz, 5Hz keeps the shape while removing micro-jitter.
            b_filter = TelemetryFilter(sample_rate=fps, cutoff_freq=5.0, order=4)
            # poses_3d_arr is (T, 17, 3). We flatten the spatial dims to filter along T.
            T, J, D = poses_3d_arr.shape
            flat_poses = poses_3d_arr.reshape(T, J * D)
            filtered_flat = b_filter.filter_array(flat_poses)
            poses_3d_arr = filtered_flat.reshape(T, J, D)
        except Exception as e:
            logger.warning(f"Failed to apply Butterworth filter: {e}")

        # Convert from meters to millimeters for the rest of the pipeline
        poses_3d_arr *= 1000.0
        
        # Push the skeleton away from the camera by the absolute depth
        poses_3d_arr[:, :, 2] += median_depth_mm

        # Apply zero-phase Butterworth filter to smooth out 3D jitter
        from .filter import TelemetryFilter
        filter_fps = fps if fps > 0 else 30.0
        pose_filter = TelemetryFilter(sample_rate=filter_fps, cutoff_freq=5.0)
        poses_3d_arr = pose_filter.filter_array(poses_3d_arr)

        # ── Map MotionBERT H36M coordinates to ThreeJS ──
        # MotionBERT outputs 3D poses in H36M space:
        #   X = right (positive)
        #   Y = up (negative is up in H36M raw, we flip it)
        #   Z = forward (positive away from camera)
        #
        # ThreeJS expects: X=right, Y=up, Z=towards-camera
        converted = np.empty_like(poses_3d_arr)
        converted[:, :, 0] = poses_3d_arr[:, :, 0]   # X (left-right)
        converted[:, :, 1] = -poses_3d_arr[:, :, 1]  # Y (flip up)
        converted[:, :, 2] = -poses_3d_arr[:, :, 2]  # Z (flip depth)
        
        # MotionAGFormer native output and our stereo scaling are in millimeters.
        # ThreeJS expects meters (the camera is at z=3, grid is 5x5).
        # We must divide by 1000, otherwise the giant skeleton clips through the camera.
        poses_3d_arr = converted / 1000.0

        # Map H36M (17 joints) → COCO (17 joints)
        coco_poses_3d = np.full((len(all_poses_3d), COCO_BODY_KEYPOINTS, 3), np.nan, dtype=np.float32)
        coco_poses_3d[:, 0] = poses_3d_arr[:, 10]  # Nose ← head (H36M 10, closest match)
        # Joints 1-4 (eyes, ears) have no H36M equivalent — left as NaN
        coco_poses_3d[:, 5] = poses_3d_arr[:, 11]  # L Shoulder
        coco_poses_3d[:, 6] = poses_3d_arr[:, 14]  # R Shoulder
        coco_poses_3d[:, 7] = poses_3d_arr[:, 12]  # L Elbow
        coco_poses_3d[:, 8] = poses_3d_arr[:, 15]  # R Elbow
        coco_poses_3d[:, 9] = poses_3d_arr[:, 13]  # L Wrist
        coco_poses_3d[:, 10] = poses_3d_arr[:, 16] # R Wrist
        coco_poses_3d[:, 11] = poses_3d_arr[:, 4]  # L Hip
        coco_poses_3d[:, 12] = poses_3d_arr[:, 1]  # R Hip
        coco_poses_3d[:, 13] = poses_3d_arr[:, 5]  # L Knee
        coco_poses_3d[:, 14] = poses_3d_arr[:, 2]  # R Knee
        coco_poses_3d[:, 15] = poses_3d_arr[:, 6]  # L Ankle
        coco_poses_3d[:, 16] = poses_3d_arr[:, 3]  # R Ankle

        keypoints_arr = np.stack(all_keypoints)  # (T, J, 2)
        confidence_arr = np.stack(all_confidence) # (T, J)

        logger.info(
            "Pose estimation complete: %d frames, %d joints, video=%.1f fps",
            len(all_keypoints), n_joints, fps,
        )

        return PoseResult(
            keypoints=keypoints_arr,
            confidence=confidence_arr,
            poses_3d=coco_poses_3d,
            num_frames=len(all_keypoints),
            num_joints=n_joints,
            fps=fps,
            frame_width=width,
            frame_height=height,
            stereo_used=stereo,
        )


