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
from typing import List, Optional, Tuple, Callable, Dict

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
    objects: Optional[Dict[str, np.ndarray]] = None


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
        task: str = "general",
        manual_bbox: Optional[list] = None,
        max_jump_ratio: float = 0.15,
    ) -> PoseResult:
        """
        Run 2D pose estimation on every frame of a video.

        Args:
            video_path: Path to an MP4 video file.
            max_frames: Optional cap on number of frames to process.
            skip_frames: Number of initial frames to skip (for synchronization).
            task: Task context for altering pose heuristics (e.g. "manipulation").

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
        red_block_traj = []
        white_block_traj = []
        
        last_valid_kpts = np.zeros((n_joints, 2), dtype=np.float32)
        last_valid_scores = np.zeros(n_joints, dtype=np.float32)
        jump_threshold = max(width, height) * max_jump_ratio
        
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
                
            if progress_callback:
                progress_callback(frame_idx / max(total_frames, 1), f"Processing frame {frame_idx}/{total_frames} (2D pose)")

            from mmengine.registry import DefaultScope

            if manual_bbox is not None:
                # Bypass object detection entirely if a manual bounding box is provided
                bboxes = np.array([manual_bbox], dtype=np.float32)
                scores = np.array([1.0], dtype=np.float32)
            else:
                # Step 1: Detect subjects
                with DefaultScope.overwrite_default_scope('mmdet'):
                    det_result = self._infer_det(self._detector, frame)
                det_instances = det_result.pred_instances
    
                # Humanoid robots often yield very low object detection confidence or are misclassified.
                # We first look for ANY 'person' detection, even with extremely low confidence.
                person_mask = (det_instances.labels == 0) & (det_instances.scores >= 0.15)
                bboxes = det_instances.bboxes[person_mask].cpu().numpy()
                scores = det_instances.scores[person_mask].cpu().numpy()
    
                # Merge adjacent or overlapping boxes.
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
                    scores = np.array([1.0] * len(bboxes), dtype=np.float32)
    
                if len(bboxes) == 0:
                    # If the robot is heavily occluded (e.g. behind a table), it may not be detected as a 'person'.
                    # Instead of falling back to the largest background object, fallback to the entire frame.
                    # The pose estimator will then search the whole frame and naturally latch onto the humanoid.
                    h_img, w_img = frame.shape[:2]
                    bboxes = np.array([[0, 0, w_img, h_img]], dtype=np.float32)
                    scores = np.array([1.0], dtype=np.float32)

            if len(bboxes) > 0:
                h_img, w_img = frame.shape[:2]
                for i in range(len(bboxes)):
                    bx1, by1, bx2, by2 = bboxes[i]
                    # Only expand the box if it wasn't manually provided
                    if manual_bbox is None:
                        bh = by2 - by1
                        bw = bx2 - bx1
                        
                        new_y1 = max(0, by1 - 0.6 * bh)
                        new_y2 = min(h_img, by2 + 1.2 * bh)
                        new_x1 = max(0, bx1 - 0.2 * bw)
                        new_x2 = min(w_img, bx2 + 0.2 * bw)
                        
                        bboxes[i] = [new_x1, new_y1, new_x2, new_y2]

            if len(bboxes) == 0:
                all_keypoints.append(np.zeros((n_joints, 2), dtype=np.float32))
                all_confidence.append(np.zeros(n_joints, dtype=np.float32))
                continue

            # Step 2: Run pose estimation (take the highest-confidence detection)
            best_idx = np.argmax(scores)
            best_bbox = bboxes[best_idx]

            with DefaultScope.overwrite_default_scope('mmpose'):
                pose_results = self._infer_pose(
                    self._pose_model,
                    frame,
                    bboxes=best_bbox[None],
                )

            if pose_results and len(pose_results) > 0:
                pred = pose_results[0].pred_instances
                kpts = pred.keypoints[0]                   # (K, 2)
                scores = pred.keypoint_scores[0]           # (K,)

                if self.keypoint_indices:
                    kpts = kpts[self.keypoint_indices]
                    scores = scores[self.keypoint_indices]
                else:
                    kpts = kpts[:COCO_BODY_KEYPOINTS]
                    scores = scores[:COCO_BODY_KEYPOINTS]
                    
                if manual_bbox is not None:
                    mx1, my1, mx2, my2 = manual_bbox
                    for j_idx in range(len(kpts)):
                        kx, ky = kpts[j_idx]
                        if kx < mx1 or kx > mx2 or ky < my1 or ky > my2:
                            scores[j_idx] = 0.0

                if task == "manipulation":
                    scores[11:17] = 0.0

                # Filter out points that jump too far from their last valid position
                for j_idx in range(n_joints):
                    if scores[j_idx] > 0.05:
                        if last_valid_scores[j_idx] > 0:
                            dist = np.linalg.norm(kpts[j_idx] - last_valid_kpts[j_idx])
                            if dist > jump_threshold:
                                scores[j_idx] = 0.0
                                continue
                        
                        last_valid_kpts[j_idx] = kpts[j_idx]
                        last_valid_scores[j_idx] = scores[j_idx]

                all_keypoints.append(kpts.astype(np.float32))
                all_confidence.append(scores.astype(np.float32))
            else:
                all_keypoints.append(np.zeros((n_joints, 2), dtype=np.float32))
                all_confidence.append(np.zeros(n_joints, dtype=np.float32))

            if task == "manipulation":
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
                mask2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
                red_mask = cv2.bitwise_or(mask1, mask2)
                
                # Green mask for the dot on the white block
                green_mask = cv2.inRange(hsv, np.array([30, 40, 40]), np.array([90, 255, 255]))
                
                if manual_bbox is not None:
                    mx1, my1, mx2, my2 = [int(v) for v in manual_bbox]
                    bbox_mask = np.zeros_like(red_mask)
                    h_m, w_m = red_mask.shape
                    my1, my2 = max(0, my1), min(h_m, my2)
                    mx1, mx2 = max(0, mx1), min(w_m, mx2)
                    bbox_mask[my1:my2, mx1:mx2] = 255
                    red_mask = cv2.bitwise_and(red_mask, bbox_mask)
                    green_mask = cv2.bitwise_and(green_mask, bbox_mask)
                
                def get_largest_centroid(mask, min_area=10):
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest = max(contours, key=cv2.contourArea)
                        if cv2.contourArea(largest) > min_area:
                            M = cv2.moments(largest)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                return [cx, cy]
                    return [np.nan, np.nan]
                    
                red_centroid = get_largest_centroid(red_mask, min_area=30)
                white_centroid = get_largest_centroid(green_mask, min_area=3)

                red_block_traj.append(red_centroid)
                white_block_traj.append(white_centroid)

            if (frame_idx + 1) % 50 == 0:
                logger.info("Processed %d / %d frames", frame_idx + 1, total_frames)

        cap.release()

        if len(all_keypoints) == 0:
            logger.warning("No valid frames extracted from video: %s", video_path)
            return PoseResult(
                keypoints=np.zeros((0, n_joints, 2), dtype=np.float32),
                confidence=np.zeros((0, n_joints), dtype=np.float32),
                num_frames=0,
                num_joints=n_joints,
                fps=fps,
                frame_width=width,
                frame_height=height,
            )

        keypoints_arr = np.stack(all_keypoints)  # (T, J, 2)
        confidence_arr = np.stack(all_confidence) # (T, J)
        
        objects = None
        if task == "manipulation":
            objects = {
                "red_block": np.array(red_block_traj, dtype=np.float32),
                "white_block": np.array(white_block_traj, dtype=np.float32)
            }
            
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
            objects=objects
        )


