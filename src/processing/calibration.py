"""
Stereo Camera Calibration Module

Provides a complete OpenCV-based stereo calibration pipeline:
1. Extract checkerboard corners from synchronised left/right video pairs.
2. Calibrate each camera individually (intrinsics).
3. Run stereo calibration to recover the rotation (R) and translation (T)
   between the two cameras.
4. Compute stereo rectification transforms.
5. Persist / load calibration results as JSON.
"""

import json
import logging
import os
import cv2
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import Tuple, List, Optional, Callable
from src.processing.sync import get_video_offset

logger = logging.getLogger(__name__)


# ── Data containers ─────────────────────────────────────────────────────────

@dataclass
class CalibrationResult:
    """Holds the full stereo calibration output."""

    # Per-camera intrinsics
    K_left: np.ndarray = field(default_factory=lambda: np.eye(3))
    dist_left: np.ndarray = field(default_factory=lambda: np.zeros(5))
    K_right: np.ndarray = field(default_factory=lambda: np.eye(3))
    dist_right: np.ndarray = field(default_factory=lambda: np.zeros(5))

    # Stereo extrinsics
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    T: np.ndarray = field(default_factory=lambda: np.zeros((3, 1)))

    # Rectification transforms (populated after stereo_rectify)
    R1: Optional[np.ndarray] = None
    R2: Optional[np.ndarray] = None
    P1: Optional[np.ndarray] = None
    P2: Optional[np.ndarray] = None
    Q: Optional[np.ndarray] = None

    # Quality metrics
    rms_stereo: float = 0.0
    rms_left: float = 0.0
    rms_right: float = 0.0
    mean_reprojection_error: float = 0.0

    image_size: Tuple[int, int] = (1280, 720)
    board_size: Tuple[int, int] = (10, 7)
    square_size: float = 0.025
    marker_size: float = 0.015
    num_valid_pairs: int = 0

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dictionary."""
        def _arr(a):
            return a.tolist() if a is not None else None

        return {
            "K_left": _arr(self.K_left),
            "dist_left": _arr(self.dist_left),
            "K_right": _arr(self.K_right),
            "dist_right": _arr(self.dist_right),
            "R": _arr(self.R),
            "T": _arr(self.T),
            "R1": _arr(self.R1),
            "R2": _arr(self.R2),
            "P1": _arr(self.P1),
            "P2": _arr(self.P2),
            "Q": _arr(self.Q),
            "rms_stereo": self.rms_stereo,
            "rms_left": self.rms_left,
            "rms_right": self.rms_right,
            "mean_reprojection_error": self.mean_reprojection_error,
            "image_size": list(self.image_size),
            "board_size": list(self.board_size),
            "square_size": self.square_size,
            "marker_size": self.marker_size,
            "num_valid_pairs": self.num_valid_pairs,
        }

    def save(self, path: str) -> None:
        """Write the calibration to a JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Calibration saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "CalibrationResult":
        """Load a previously saved calibration from JSON."""
        with open(path) as f:
            d = json.load(f)

        def _np(v, shape=None):
            if v is None:
                return None
            a = np.array(v, dtype=np.float64)
            return a.reshape(shape) if shape else a

        result = cls()
        result.K_left = _np(d["K_left"], (3, 3))
        result.dist_left = _np(d["dist_left"])
        result.K_right = _np(d["K_right"], (3, 3))
        result.dist_right = _np(d["dist_right"])
        result.R = _np(d["R"], (3, 3))
        result.T = _np(d["T"], (3, 1))
        result.R1 = _np(d.get("R1"))
        result.R2 = _np(d.get("R2"))
        result.P1 = _np(d.get("P1"))
        result.P2 = _np(d.get("P2"))
        result.Q = _np(d.get("Q"))
        result.rms_stereo = d.get("rms_stereo", 0.0)
        result.rms_left = d.get("rms_left", 0.0)
        result.rms_right = d.get("rms_right", 0.0)
        result.mean_reprojection_error = d.get("mean_reprojection_error", 0.0)
        result.image_size = tuple(d.get("image_size", [1280, 720]))
        result.board_size = tuple(d.get("board_size", [10, 7]))
        result.square_size = d.get("square_size", 0.025)
        result.marker_size = d.get("marker_size", 0.015)
        result.num_valid_pairs = d.get("num_valid_pairs", 0)
        return result


# ── Corner extraction ───────────────────────────────────────────────────────

def _get_charuco_board(board_size, square_size, marker_size):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard(board_size, square_size, marker_size, dictionary)
    return board, dictionary

def extract_corners_from_video_pair(
    left_path: str,
    right_path: str,
    board_size: Tuple[int, int] = (10, 7),
    square_size: float = 0.025,
    marker_size: float = 0.015,
    frame_skip: int = 1,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], Tuple[int, int]]:
    cap_l = cv2.VideoCapture(left_path)
    cap_r = cv2.VideoCapture(right_path)

    # Automatically sync the two videos based on audio cross-correlation
    fps_l = cap_l.get(cv2.CAP_PROP_FPS) or 30.0
    fps_r = cap_r.get(cv2.CAP_PROP_FPS) or 30.0
    offset_seconds = get_video_offset(left_path, right_path)
    
    if offset_seconds > 0:
        # Left video started earlier, so we skip frames in left video
        skip_frames = int(offset_seconds * fps_l)
        logger.info(f"Sync: Skipping {skip_frames} frames in left video")
        for _ in range(skip_frames):
            cap_l.read()
    elif offset_seconds < 0:
        # Right video started earlier
        skip_frames = int(abs(offset_seconds) * fps_r)
        logger.info(f"Sync: Skipping {skip_frames} frames in right video")
        for _ in range(skip_frames):
            cap_r.read()

    if not cap_l.isOpened():
        raise FileNotFoundError(f"Cannot open left video: {left_path}")
    if not cap_r.isOpened():
        raise FileNotFoundError(f"Cannot open right video: {right_path}")

    left_corners: List[np.ndarray] = []
    left_ids: List[np.ndarray] = []
    right_corners: List[np.ndarray] = []
    right_ids: List[np.ndarray] = []
    image_size = None
    frame_idx = 0

    board, dictionary = _get_charuco_board(board_size, square_size, marker_size)
    detectorParams = cv2.aruco.DetectorParameters()
    charucoDetector = cv2.aruco.CharucoDetector(board)

    total_frames = int(cap_l.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()
        if not ret_l or not ret_r:
            break
        
        if progress_callback and total_frames > 0 and frame_idx % 30 == 0:
            progress_callback(min(1.0, frame_idx / total_frames))

        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        if image_size is None:
            image_size = (frame_l.shape[1], frame_l.shape[0])

        grey_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        grey_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

        charucoCorners_l, charucoIds_l, _, _ = charucoDetector.detectBoard(grey_l)
        charucoCorners_r, charucoIds_r, _, _ = charucoDetector.detectBoard(grey_r)

        # Ensure enough corners were found in both
        if charucoCorners_l is not None and charucoCorners_r is not None and len(charucoCorners_l) > 6 and len(charucoCorners_r) > 6:
            # Flatten IDs to handle varying OpenCV versions
            flat_ids_l = charucoIds_l.flatten()
            flat_ids_r = charucoIds_r.flatten()
            
            # Find the intersection of detected IDs
            common_ids = np.intersect1d(flat_ids_l, flat_ids_r)
            if len(common_ids) > 6:
                # Filter to only keep common corners
                filt_corners_l = []
                filt_ids_l = []
                for i, id_val in enumerate(flat_ids_l):
                    if id_val in common_ids:
                        filt_corners_l.append(charucoCorners_l[i])
                        filt_ids_l.append([id_val])
                
                filt_corners_r = []
                filt_ids_r = []
                for i, id_val in enumerate(flat_ids_r):
                    if id_val in common_ids:
                        filt_corners_r.append(charucoCorners_r[i])
                        filt_ids_r.append([id_val])
                
                # Sort them so they align
                sorted_l = sorted(zip(filt_ids_l, filt_corners_l), key=lambda x: x[0][0])
                sorted_r = sorted(zip(filt_ids_r, filt_corners_r), key=lambda x: x[0][0])

                left_corners.append(np.array([x[1] for x in sorted_l]))
                left_ids.append(np.array([x[0] for x in sorted_l]))
                right_corners.append(np.array([x[1] for x in sorted_r]))
                right_ids.append(np.array([x[0] for x in sorted_r]))

        frame_idx += 1

    if progress_callback:
        progress_callback(1.0)

    cap_l.release()
    cap_r.release()

    logger.info(
        "Found %d valid stereo pairs out of %d frames",
        len(left_corners), frame_idx,
    )
    return left_corners, left_ids, right_corners, right_ids, image_size or (1280, 720)


# ── Calibration ─────────────────────────────────────────────────────────────

def calibrate_stereo(
    left_video_path: str,
    right_video_path: str,
    board_size: Tuple[int, int] = (10, 7),
    square_size: float = 0.025,
    marker_size: float = 0.015,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> CalibrationResult:
    result = CalibrationResult(
        board_size=board_size,
        square_size=square_size,
        marker_size=marker_size,
    )

    # Step 1: extract corners
    left_corners, left_ids, right_corners, right_ids, img_size = extract_corners_from_video_pair(
        left_video_path, right_video_path, board_size, square_size, marker_size, progress_callback=progress_callback
    )
    result.image_size = img_size
    result.num_valid_pairs = len(left_corners)

    if len(left_corners) < 5:
        raise ValueError(
            f"Only {len(left_corners)} valid stereo pairs found — need at least 5 "
            "for a reliable calibration. Ensure the ChArUco board is visible in "
            "both cameras simultaneously."
        )

    board, dictionary = _get_charuco_board(board_size, square_size, marker_size)

    # Convert ChArUco corners and IDs into standard object/image point lists for calibrateCamera
    object_points = []
    left_corners_list = []
    right_corners_list = []

    board_obj_pts = board.getChessboardCorners()
    for i in range(len(left_corners)):
        obj_pts_frame = np.array([board_obj_pts[id[0]] for id in left_ids[i]], dtype=np.float32)
        object_points.append(obj_pts_frame)
        left_corners_list.append(left_corners[i].astype(np.float32))
        right_corners_list.append(right_corners[i].astype(np.float32))

    # Step 2: individual camera calibration
    logger.info("Calibrating left camera …")
    result.rms_left, result.K_left, result.dist_left, _, _ = cv2.calibrateCamera(
        object_points, left_corners_list, img_size, None, None
    )
    logger.info("Left camera RMS: %.4f", result.rms_left)

    logger.info("Calibrating right camera …")
    result.rms_right, result.K_right, result.dist_right, _, _ = cv2.calibrateCamera(
        object_points, right_corners_list, img_size, None, None
    )
    logger.info("Right camera RMS: %.4f", result.rms_right)

    # Step 3: stereo calibration
    logger.info("Running stereo calibration …")
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    result.rms_stereo, _, _, _, _, result.R, result.T, _, _ = cv2.stereoCalibrate(
        object_points,
        left_corners_list,
        right_corners_list,
        result.K_left,
        result.dist_left,
        result.K_right,
        result.dist_right,
        img_size,
        criteria=criteria,
        flags=flags,
    )
    logger.info("Stereo RMS: %.4f", result.rms_stereo)

    # Step 4: stereo rectification
    logger.info("Computing stereo rectification …")
    result.R1, result.R2, result.P1, result.P2, result.Q, _, _ = cv2.stereoRectify(
        result.K_left, result.dist_left,
        result.K_right, result.dist_right,
        img_size, result.R, result.T,
        alpha=0,
    )

    result.mean_reprojection_error = result.rms_stereo

    logger.info("Calibration complete. %d pairs, stereo RMS=%.4f",
                result.num_valid_pairs, result.rms_stereo)

    return result
