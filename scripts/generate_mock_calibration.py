"""
Mock Stereo Checkerboard Video Generator

Generates synthetic left/right camera videos of a checkerboard pattern
being moved through a calibration volume, suitable for testing the
stereo calibration pipeline without real hardware.

The generator:
1. Defines two virtual cameras with known intrinsics and a known R/T relationship.
2. For each frame, places a 3D checkerboard at a random pose in the scene.
3. Renders the full checkerboard (including the outer border that OpenCV needs)
   by warping a canonical board image into the camera view via a homography.
4. Encodes both streams into MP4 files.

Because we know the ground-truth R and T, we can validate the calibration
pipeline's output against these values.
"""

import argparse
import json
import os

import cv2
import numpy as np


# ── Ground-truth camera parameters ──────────────────────────────────────────

# Image dimensions
IMG_W, IMG_H = 1280, 720

# Checkerboard: inner corners are (BOARD_COLS x BOARD_ROWS).
# The full board has (BOARD_COLS+1) x (BOARD_ROWS+1) squares.
BOARD_COLS, BOARD_ROWS = 9, 6
SQUARE_SIZE = 0.025  # 25 mm squares in metres

# Number of full squares in each dimension (one more than inner corners)
FULL_COLS = BOARD_COLS + 1
FULL_ROWS = BOARD_ROWS + 1


def _make_intrinsics(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Build a 3x3 camera intrinsic matrix."""
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=np.float64)


def _make_ground_truth() -> dict:
    """
    Return the ground-truth stereo rig geometry.

    The left camera is at the world origin.
    The right camera is translated 0.12 m to the right (along X) and
    rotated 0° relative to the left — a simple parallel stereo pair.
    """
    K = _make_intrinsics(fx=800.0, fy=800.0, cx=IMG_W / 2, cy=IMG_H / 2)
    dist = np.zeros(5, dtype=np.float64)

    R = np.eye(3, dtype=np.float64)
    T = np.array([[0.12], [0.0], [0.0]], dtype=np.float64)

    return {
        "K_left": K,
        "dist_left": dist,
        "K_right": K.copy(),
        "dist_right": dist.copy(),
        "R": R,
        "T": T,
    }


def _make_object_points() -> np.ndarray:
    """
    Canonical 3-D coordinates of the inner checkerboard corners in the
    board's own coordinate frame (Z = 0 plane).
    """
    objp = np.zeros((BOARD_ROWS * BOARD_COLS, 3), dtype=np.float64)
    objp[:, :2] = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2) * SQUARE_SIZE
    return objp


def _create_canonical_board_image(px_per_square: int = 40) -> tuple:
    """
    Create a clean checkerboard image in pixel space with a white border.
    Returns (image, corner_positions_in_px).
    
    The board has FULL_COLS x FULL_ROWS squares.  OpenCV's findChessboardCorners
    expects the squares around each inner corner to alternate colours, plus
    a visible border of at least one square around the entire pattern.
    """
    # Add 1-square border on each side
    border = 1
    total_cols = FULL_COLS + 2 * border
    total_rows = FULL_ROWS + 2 * border
    
    w = total_cols * px_per_square
    h = total_rows * px_per_square
    
    img = np.ones((h, w), dtype=np.uint8) * 255  # white background
    
    for r in range(total_rows):
        for c in range(total_cols):
            if (r + c) % 2 == 1:
                x0 = c * px_per_square
                y0 = r * px_per_square
                img[y0:y0 + px_per_square, x0:x0 + px_per_square] = 0
    
    # The inner corners (in the canonical image pixel coords)
    # Inner corners start at (border, border) squares offset and span
    # BOARD_COLS x BOARD_ROWS corners.
    corner_pts = []
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            px_x = (border + c + 1) * px_per_square  # +1 because corners are between squares
            px_y = (border + r + 1) * px_per_square
            corner_pts.append([px_x, px_y])
    
    return img, np.array(corner_pts, dtype=np.float32)


def _random_board_pose(rng: np.random.Generator) -> tuple:
    """
    Generate a random 3-D rigid-body transform for the checkerboard.
    Returns (rvec, tvec) suitable for cv2.projectPoints.
    """
    angles = rng.uniform([-0.3, -0.3, -0.15], [0.3, 0.3, 0.15])
    rvec = angles.astype(np.float64).reshape(3, 1)

    tx = rng.uniform(-0.08, 0.08)
    ty = rng.uniform(-0.06, 0.06)
    tz = rng.uniform(0.45, 0.85)
    tvec = np.array([[tx], [ty], [tz]], dtype=np.float64)

    return rvec, tvec


def _render_frame(
    board_img: np.ndarray,
    board_corner_px: np.ndarray,
    obj_pts_3d: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray:
    """
    Render a camera frame by warping the canonical board image onto the
    camera view via homography.
    """
    # Project the 3D inner-corner positions into the camera image
    img_pts_2d, _ = cv2.projectPoints(obj_pts_3d, rvec, tvec, K, dist)
    img_pts_2d = img_pts_2d.reshape(-1, 2).astype(np.float32)
    
    # Check all projected points are inside the frame with margin
    margin = 20
    if (np.any(img_pts_2d[:, 0] < margin) or np.any(img_pts_2d[:, 0] > IMG_W - margin) or
        np.any(img_pts_2d[:, 1] < margin) or np.any(img_pts_2d[:, 1] > IMG_H - margin)):
        # Board is partially outside the frame — return grey background only
        return np.full((IMG_H, IMG_W, 3), 128, dtype=np.uint8)
    
    # Compute homography from canonical board corners to projected 2D points
    # We need at least 4 point correspondences; we have BOARD_ROWS*BOARD_COLS
    H, _ = cv2.findHomography(board_corner_px, img_pts_2d)
    
    if H is None:
        return np.full((IMG_H, IMG_W, 3), 128, dtype=np.uint8)
    
    # Warp the canonical board image
    board_bgr = cv2.cvtColor(board_img, cv2.COLOR_GRAY2BGR)
    frame = np.full((IMG_H, IMG_W, 3), 128, dtype=np.uint8)
    warped = cv2.warpPerspective(board_bgr, H, (IMG_W, IMG_H),
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(128, 128, 128))
    
    # Composite: use the warped board where it's not the border colour
    mask = np.any(warped != 128, axis=2)
    frame[mask] = warped[mask]
    
    return frame


def _board_pose_for_right_camera(
    rvec_board: np.ndarray,
    tvec_board: np.ndarray,
    R_stereo: np.ndarray,
    T_stereo: np.ndarray,
) -> tuple:
    """
    Convert a board pose (expressed in the left-camera frame) into
    the equivalent pose in the right-camera frame.
    """
    R_board, _ = cv2.Rodrigues(rvec_board)
    R_right = R_stereo @ R_board
    t_right = R_stereo @ tvec_board + T_stereo
    rvec_right, _ = cv2.Rodrigues(R_right)
    return rvec_right, t_right


def generate(
    output_dir: str,
    num_frames: int = 60,
    fps: int = 15,
    seed: int = 42,
) -> dict:
    """
    Generate a pair of synthetic stereo calibration MP4 videos and save
    the ground-truth parameters to a JSON sidecar.

    Returns the ground-truth dict for programmatic use.
    """
    os.makedirs(output_dir, exist_ok=True)

    gt = _make_ground_truth()
    obj_pts = _make_object_points()
    rng = np.random.default_rng(seed)
    
    # Create the canonical board image + pixel coords of inner corners
    board_img, board_corner_px = _create_canonical_board_image(px_per_square=40)

    left_path = os.path.join(output_dir, "calibration_left.mp4")
    right_path = os.path.join(output_dir, "calibration_right.mp4")
    gt_path = os.path.join(output_dir, "ground_truth.json")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    left_writer = cv2.VideoWriter(left_path, fourcc, fps, (IMG_W, IMG_H))
    right_writer = cv2.VideoWriter(right_path, fourcc, fps, (IMG_W, IMG_H))

    valid_frames = 0
    for _ in range(num_frames):
        rvec_board, tvec_board = _random_board_pose(rng)

        # Left camera image
        left_img = _render_frame(
            board_img, board_corner_px, obj_pts,
            rvec_board, tvec_board, gt["K_left"], gt["dist_left"],
        )

        # Right camera image
        rvec_r, tvec_r = _board_pose_for_right_camera(
            rvec_board, tvec_board, gt["R"], gt["T"]
        )
        right_img = _render_frame(
            board_img, board_corner_px, obj_pts,
            rvec_r, tvec_r, gt["K_right"], gt["dist_right"],
        )

        left_writer.write(left_img)
        right_writer.write(right_img)
        valid_frames += 1

    left_writer.release()
    right_writer.release()

    # Serialise ground truth
    gt_serialisable = {
        "K_left": gt["K_left"].tolist(),
        "dist_left": gt["dist_left"].tolist(),
        "K_right": gt["K_right"].tolist(),
        "dist_right": gt["dist_right"].tolist(),
        "R": gt["R"].tolist(),
        "T": gt["T"].tolist(),
        "image_size": [IMG_W, IMG_H],
        "board_size": [BOARD_COLS, BOARD_ROWS],
        "square_size": SQUARE_SIZE,
    }
    with open(gt_path, "w") as f:
        json.dump(gt_serialisable, f, indent=2)

    print(f"✓ Left  video: {left_path} ({valid_frames} frames)")
    print(f"✓ Right video: {right_path} ({valid_frames} frames)")
    print(f"✓ Ground truth: {gt_path}")

    return gt_serialisable


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mock stereo calibration videos")
    parser.add_argument("--output-dir", default="data/mock_calibration",
                        help="Directory to write the MP4 files and ground truth JSON")
    parser.add_argument("--frames", type=int, default=60,
                        help="Number of frames per video")
    parser.add_argument("--fps", type=int, default=15,
                        help="Frames per second in the output videos")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    generate(args.output_dir, args.frames, args.fps, args.seed)
