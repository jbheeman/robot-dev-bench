"""Pure stereo triangulation math for the sensor-fusion pipeline.

Ported from rt-pose-atao/src/stereo_core.py.

No cv2, no torch, no I/O.  Everything is numpy in / numpy out so it can be
unit-tested in isolation.

The key formula is the classic pinhole stereo depth equation:
    Z = (focal_length_px * baseline_mm) / disparity_px

where disparity = x_left - x_right for a rectified, side-by-side stereo pair.
"""

from __future__ import annotations

import numpy as np


def split_stereo_frame(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Slice a side-by-side stereo frame into left and right halves.

    frame: (H, W, C) where W is the combined width (left | right).
    Returns (left, right), each (H, W//2, C).
    """
    if frame.ndim != 3:
        raise ValueError(f"expected 3-dim image (H, W, C), got ndim={frame.ndim}")
    h, w, c = frame.shape
    mid = w // 2
    return frame[:, :mid, :].copy(), frame[:, mid:mid + mid, :].copy()


def triangulate_depth(x_left: float, x_right: float,
                      focal_length_px: float, baseline_mm: float) -> float:
    """Compute the depth (Z) of a point from its horizontal pixel coordinates
    in a rectified stereo pair.

    x_left:  horizontal pixel coordinate in the left image.
    x_right: horizontal pixel coordinate in the right image.
    focal_length_px: focal length expressed in pixels.
    baseline_mm: distance between the two cameras in millimeters.

    Returns depth Z in millimeters.  If the disparity is zero or negative
    (invalid), returns +inf.
    """
    disparity = x_left - x_right
    if disparity <= 0:
        return float("inf")
    return (focal_length_px * baseline_mm) / disparity


def triangulate_point_3d(pt_left: np.ndarray, pt_right: np.ndarray,
                         focal_length_px: float, baseline_mm: float,
                         image_width: int, image_height: int) -> np.ndarray:
    """Compute the full (X, Y, Z) position of a point in millimeters.

    Uses the pinhole camera model:
        X = (x_left - cx) * Z / f
        Y = (y_left - cy) * Z / f
        Z = f * baseline / disparity

    pt_left, pt_right: (2,) or (3,) arrays; only the first two elements
        (pixel x, y) are used.
    image_width, image_height: dimensions of a *single* (not side-by-side) frame,
        used to compute the principal point (cx, cy) = (W/2, H/2).

    Returns (3,) float32 array [X, Y, Z] in millimeters.
    """
    xl, yl = float(pt_left[0]), float(pt_left[1])
    xr = float(pt_right[0])

    z = triangulate_depth(xl, xr, focal_length_px, baseline_mm)

    cx = image_width / 2.0
    cy = image_height / 2.0
    x = (xl - cx) * z / focal_length_px
    y = (yl - cy) * z / focal_length_px
    return np.array([x, y, z], dtype=np.float32)


def match_pelvis(coco_left: np.ndarray, coco_right: np.ndarray,
                 conf_thresh: float = 0.3) -> tuple[np.ndarray, np.ndarray, bool]:
    """Extract the pelvis (midpoint of hips) from COCO-17 keypoints in both views.

    coco_left, coco_right: (17, 3) arrays [x, y, conf].
    Returns (pelvis_left, pelvis_right, valid) where each pelvis is (2,) pixel
    coords.  `valid` is True only when both hips are above `conf_thresh` in
    *both* views.
    """
    # COCO indices: 11 = left hip, 12 = right hip
    L_HIP, R_HIP = 11, 12

    left_ok = (coco_left[L_HIP, 2] >= conf_thresh and
               coco_left[R_HIP, 2] >= conf_thresh)
    right_ok = (coco_right[L_HIP, 2] >= conf_thresh and
                coco_right[R_HIP, 2] >= conf_thresh)

    pelvis_left = (coco_left[L_HIP, :2] + coco_left[R_HIP, :2]) / 2.0
    pelvis_right = (coco_right[L_HIP, :2] + coco_right[R_HIP, :2]) / 2.0

    return pelvis_left, pelvis_right, bool(left_ok and right_ok)
