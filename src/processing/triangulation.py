"""
Stereo Triangulation Module

Given matched 2D joint detections from a calibrated stereo camera pair,
triangulates each joint into 3D world coordinates using the Direct Linear
Transform (DLT) method via OpenCV's triangulatePoints.

Input:
    - 2D keypoints from left camera: (N_frames, N_joints, 2)
    - 2D keypoints from right camera: (N_frames, N_joints, 2)
    - Projection matrices P1, P2 from stereo rectification (3x4 each)

Output:
    - 3D joint positions: (N_frames, N_joints, 3) in world coordinates
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from src.processing.calibration import CalibrationResult

logger = logging.getLogger(__name__)


def triangulate_points(
    pts_left: np.ndarray,
    pts_right: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
) -> np.ndarray:
    """
    Triangulate a set of corresponding 2D points into 3D.

    Args:
        pts_left:  (N, 2) array of 2D points in the left image.
        pts_right: (N, 2) array of 2D points in the right image.
        P1: (3, 4) projection matrix for the left camera.
        P2: (3, 4) projection matrix for the right camera.

    Returns:
        (N, 3) array of 3D points in the rectified coordinate frame.
    """
    # OpenCV wants (2, N) float64
    pts_l = pts_left.T.astype(np.float64)   # (2, N)
    pts_r = pts_right.T.astype(np.float64)  # (2, N)

    # triangulatePoints returns (4, N) homogeneous coordinates
    pts_4d = cv2.triangulatePoints(P1, P2, pts_l, pts_r)

    # Convert from homogeneous to 3D
    pts_3d = (pts_4d[:3] / pts_4d[3:]).T  # (N, 3)

    return pts_3d


def undistort_points(
    points: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    R: Optional[np.ndarray] = None,
    P: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Undistort and optionally rectify 2D keypoints.

    Args:
        points: (N, 2) array of distorted 2D points.
        K:      (3, 3) camera intrinsic matrix.
        dist:   distortion coefficients.
        R:      (3, 3) rectification rotation (from stereoRectify).
        P:      (3, 4) new projection matrix (from stereoRectify).

    Returns:
        (N, 2) array of undistorted (and optionally rectified) points.
    """
    # cv2.undistortPoints expects (N, 1, 2)
    pts = points.reshape(-1, 1, 2).astype(np.float64)
    undistorted = cv2.undistortPoints(pts, K, dist, R=R, P=P)
    return undistorted.reshape(-1, 2)


def triangulate_pose_sequence(
    keypoints_left: np.ndarray,
    keypoints_right: np.ndarray,
    calibration: CalibrationResult,
    confidence_left: Optional[np.ndarray] = None,
    confidence_right: Optional[np.ndarray] = None,
    min_confidence: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Triangulate a full sequence of 2D pose detections into 3D.

    Args:
        keypoints_left:  (T, J, 2) — 2D keypoints per frame from left camera.
        keypoints_right: (T, J, 2) — 2D keypoints per frame from right camera.
        calibration:     CalibrationResult with K, dist, R1/R2, P1/P2.
        confidence_left: (T, J) — optional confidence scores for left detections.
        confidence_right:(T, J) — optional confidence scores for right detections.
        min_confidence:  Minimum confidence threshold. Joints below this in
                         either camera are set to NaN in the 3D output.

    Returns:
        poses_3d:   (T, J, 3) array of 3D joint positions.
                    Joints with insufficient confidence are NaN.
        valid_mask: (T, J) boolean array indicating which joints were
                    successfully triangulated.
    """
    T, J, _ = keypoints_left.shape
    assert keypoints_right.shape == (T, J, 2), (
        f"Shape mismatch: left={keypoints_left.shape}, right={keypoints_right.shape}"
    )

    # Ensure calibration has rectification matrices
    if calibration.P1 is None or calibration.P2 is None:
        raise ValueError(
            "Calibration is missing rectification matrices (P1/P2). "
            "Run stereo_rectify first."
        )

    P1 = calibration.P1.astype(np.float64)
    P2 = calibration.P2.astype(np.float64)

    poses_3d = np.full((T, J, 3), np.nan, dtype=np.float64)
    valid_mask = np.zeros((T, J), dtype=bool)

    for t in range(T):
        # Determine which joints are valid in this frame
        if confidence_left is not None and confidence_right is not None:
            joint_valid = (
                (confidence_left[t] >= min_confidence) &
                (confidence_right[t] >= min_confidence)
            )
        else:
            # If no confidence provided, assume all joints valid if coords are non-zero
            joint_valid = (
                (np.abs(keypoints_left[t]).sum(axis=1) > 0) &
                (np.abs(keypoints_right[t]).sum(axis=1) > 0)
            )

        valid_indices = np.where(joint_valid)[0]
        if len(valid_indices) == 0:
            continue

        # Undistort and rectify the 2D points
        pts_l = undistort_points(
            keypoints_left[t, valid_indices],
            calibration.K_left, calibration.dist_left,
            R=calibration.R1, P=P1,
        )
        pts_r = undistort_points(
            keypoints_right[t, valid_indices],
            calibration.K_right, calibration.dist_right,
            R=calibration.R2, P=P2,
        )

        # Triangulate
        pts_3d = triangulate_points(pts_l, pts_r, P1, P2)

        poses_3d[t, valid_indices] = pts_3d
        valid_mask[t, valid_indices] = True

    n_valid = np.sum(valid_mask)
    n_total = T * J
    logger.info(
        "Triangulated %d / %d joint-frames (%.1f%%)",
        n_valid, n_total, 100.0 * n_valid / max(n_total, 1),
    )

    return poses_3d, valid_mask


def postprocess_3d_poses(
    poses_3d: np.ndarray,
    valid_mask: np.ndarray,
    jump_threshold: float = 0.5,
    alpha: float = 0.3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Post-processes triangulated 3D poses to fix orientation, reject outliers, and smooth.
    
    Args:
        poses_3d: (T, J, 3) array of triangulated 3D joint positions.
        valid_mask: (T, J) boolean array of valid joints.
        jump_threshold: Maximum allowed Euclidean distance (meters) between valid frames.
        alpha: Smoothing factor for Exponential Moving Average (0 = no update, 1 = no smoothing).
    
    Returns:
        poses_3d_filtered: The processed (T, J, 3) array.
        valid_mask_filtered: The updated (T, J) boolean mask.
    """
    T, J, _ = poses_3d.shape
    filtered_poses = poses_3d.copy()
    filtered_mask = valid_mask.copy()
    
    # 1. Axis Inversion (OpenCV Y points down, flip to Y-up for rendering/biomechanics)
    filtered_poses[:, :, 1] *= -1

    # 2. Outlier Rejection & Smoothing per-joint
    for j in range(J):
        last_valid_pos = None
        
        for t in range(T):
            if not filtered_mask[t, j]:
                continue
                
            curr_pos = filtered_poses[t, j]
            
            if last_valid_pos is not None:
                dist = np.linalg.norm(curr_pos - last_valid_pos)
                if dist > jump_threshold:
                    # Reject as outlier
                    filtered_poses[t, j] = np.nan
                    filtered_mask[t, j] = False
                    continue
                    
                # Exponential Moving Average Smoothing
                smoothed_pos = (alpha * curr_pos) + ((1 - alpha) * last_valid_pos)
                filtered_poses[t, j] = smoothed_pos
                last_valid_pos = smoothed_pos
            else:
                last_valid_pos = curr_pos

    return filtered_poses, filtered_mask
