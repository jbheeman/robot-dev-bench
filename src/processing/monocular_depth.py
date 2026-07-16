"""
Monocular Depth Inference Module

Estimates 3D joint positions from a SINGLE camera's 2D pose sequence,
in place of src.processing.triangulation which requires two calibrated
cameras.

How it works
------------
A second camera view lets triangulation recover depth geometrically, with
no assumptions about the subject. With one camera, depth is fundamentally
ambiguous from geometry alone -- so instead this module uses a *physical*
constraint: the real-world length of each skeletal segment (bone) is
assumed known. Starting from a root joint, it walks the skeleton's
kinematic tree outward and, for each parent -> child bone, solves for the
child's depth as the value that makes the 3D distance between parent and
child equal to the bone's known length, given the child's fixed 2D
viewing ray.

This is a much better fit for the G1 than it would be for an arbitrary
human subject: the robot's link lengths are exact and fixed (see the
URDFs under src/web/static/assets/g1_description/), so the constraint
used here is exact rather than an averaged anthropometric guess.
BONE_LENGTHS_M defaults to G1_BONE_LENGTHS_M, computed via forward
kinematics directly from g1_29dof_rev_1_0.urdf (see that dict for
details/caveats). Swap to HUMAN_BONE_LENGTHS_M if pointing the camera at
a person instead of the robot.

Math
----
Let a joint's undistorted, normalised image ray be r = (x, y, 1) so that
its 3D position is P = Z * r for some depth Z. Given the parent's already
solved 3D position P_parent and a known bone length L to the child (ray
r_child), we need:

    || Z_child * r_child - P_parent ||^2 = L^2

which expands to a quadratic in Z_child:

    a = r_child . r_child
    b = -2 * r_child . P_parent
    c = P_parent . P_parent - L^2

Solved analytically below, picking the root closer to the parent's depth
(the physically plausible one; the other root mirrors the joint behind
the camera-side solution).

Limitations vs. stereo triangulation
-------------------------------------
- Requires known, per-subject bone lengths -- wrong lengths bias depth.
- No cross-view check, so 2D detection noise maps directly into depth
  error, especially for bones seen nearly end-on (foreshortened).
- A bone can be geometrically inconsistent with its 2D projection (noisy
  detections, wrong length) giving a negative discriminant; when that
  happens this falls back to placing the child at the parent's depth.
- Errors accumulate outward along the kinematic chain from the root.
"""

import logging
from typing import Dict, List, Optional, Tuple, Callable

import cv2
import numpy as np

logger = logging.getLogger(__name__)

from src.processing.pose_estimation import COCO_KEYPOINT_NAMES

# ── Skeleton definition ─────────────────────────────────────────────────────
#
# A spanning tree over 13 of the 17 COCO body keypoints: (parent_idx, child_idx).
# Every joint except the root (11, left_hip) appears exactly once as a child,
# so depth can be propagated outward with no cycles.
# Eyes/ears (1, 2, 3, 4) are intentionally excluded: the G1 has no
# corresponding rigid features, so there's no known bone length to anchor
# them to. Nose (0) is kept as a rough head-position proxy via the
# shoulder->nose "neck + head" bone.
SKELETON_TREE: List[Tuple[int, int]] = [
    (11, 12),  # left_hip -> right_hip      (pelvis width)
    (11, 5),   # left_hip -> left_shoulder  (left torso)
    (12, 14),  # right_hip -> right_knee    (right thigh)
    (14, 16),  # right_knee -> right_ankle  (right shank)
    (11, 13),  # left_hip -> left_knee      (left thigh)
    (13, 15),  # left_knee -> left_ankle    (left shank)
    (5, 6),    # left_shoulder -> right_shoulder (shoulder width)
    (5, 7),    # left_shoulder -> left_elbow     (left upper arm)
    (7, 9),    # left_elbow -> left_wrist        (left forearm)
    (6, 8),    # right_shoulder -> right_elbow   (right upper arm)
    (8, 10),   # right_elbow -> right_wrist      (right forearm)
    (5, 0),    # left_shoulder -> nose      (neck + head, approximate)
]

ROOT_JOINT = 11  # left_hip

# Segment lengths in metres for the Unitree G1 (29-DOF), computed via forward
# kinematics on src/web/static/assets/g1_description/g1_29dof_rev_1_0.urdf at
# the URDF's zero-joint-angle (neutral) pose, mapping each COCO body joint to
# the nearest corresponding G1 joint origin:
#   hip -> *_hip_pitch_joint, shoulder -> *_shoulder_pitch_joint,
#   elbow -> *_elbow_joint, wrist -> *_wrist_roll_joint,
#   knee -> *_knee_joint, ankle -> *_ankle_pitch_joint.
# Note: the hip/shoulder are each a 3-DOF sub-chain (pitch/roll/yaw) with a
# few cm of offset between them, so these lengths (measured joint-origin to
# joint-origin at neutral pose) are very close to, but not perfectly
# invariant across, the robot's full range of joint angles.
G1_BONE_LENGTHS_M: Dict[Tuple[int, int], float] = {
    (11, 12): 0.1289,  # pelvis width (left_hip_pitch <-> right_hip_pitch)
    (11, 5): 0.3961,   # torso, left hip -> left shoulder
    (12, 14): 0.3409,  # right thigh (hip_pitch -> knee)
    (14, 16): 0.3000,  # right shank (knee -> ankle_pitch)
    (11, 13): 0.3409,  # left thigh
    (13, 15): 0.3000,  # left shank
    (5, 6): 0.2004,    # shoulder width
    (5, 7): 0.1929,    # left upper arm (shoulder_pitch -> elbow)
    (7, 9): 0.1005,    # left forearm (elbow -> wrist_roll)
    (6, 8): 0.1929,    # right upper arm
    (8, 10): 0.1005,   # right forearm
    # The G1's head is a single rigid link with no separate nose/eye/ear
    # geometry, so this isn't derivable from the URDF the way the limbs
    # above are. head_joint's own origin sits at the neck mount point
    # (near torso height, not at the front of the head), so it's a poor
    # stand-in for "nose" -- left as a rough placeholder. Eyes/ears (1-4)
    # are dropped from SKELETON_TREE entirely rather than guessed.
    (5, 0): 0.15,    # neck + head (approximate)
}

# Rough adult-human proportions, kept for use if pointing the camera at a
# human operator instead of the G1 itself.
HUMAN_BONE_LENGTHS_M: Dict[Tuple[int, int], float] = {
    (11, 12): 0.25,  # pelvis width
    (11, 5): 0.50,   # torso (hip-shoulder)
    (12, 14): 0.45,  # right thigh
    (14, 16): 0.40,  # right shank
    (11, 13): 0.45,  # left thigh
    (13, 15): 0.40,  # left shank
    (5, 6): 0.35,    # shoulder width
    (5, 7): 0.30,    # left upper arm
    (7, 9): 0.25,    # left forearm
    (6, 8): 0.30,    # right upper arm
    (8, 10): 0.25,   # right forearm
    (5, 0): 0.25,    # neck + head
}

BONE_LENGTHS_M = G1_BONE_LENGTHS_M


def _undistort_to_rays(points: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """
    Convert pixel coordinates to normalised camera rays (x, y, 1).

    Args:
        points: (N, 2) pixel coordinates.
        K:      (3, 3) camera intrinsic matrix.
        dist:   distortion coefficients.

    Returns:
        (N, 3) array of rays in camera space (undistorted, unit focal length).
    """
    pts = points.reshape(-1, 1, 2).astype(np.float64)
    normalised = cv2.undistortPoints(pts, K, dist)  # (N, 1, 2), no R/P -> normalised coords
    rays = np.ones((points.shape[0], 3), dtype=np.float64)
    rays[:, :2] = normalised.reshape(-1, 2)
    return rays


def _solve_child_depth(ray_child: np.ndarray, p_parent: np.ndarray, bone_length: float, z_parent: float) -> float:
    """
    Solve for the child joint's depth given its ray, the parent's solved 3D
    position, and the known bone length between them.
    """
    a = float(np.dot(ray_child, ray_child))
    b = -2.0 * float(np.dot(ray_child, p_parent))
    c = float(np.dot(p_parent, p_parent) - bone_length ** 2)

    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        # 2D bone length is geometrically inconsistent with the known 3D
        # length (noise or wrong bone_length) -- fall back to parent's depth.
        logger.debug("Negative discriminant solving child depth; falling back to parent depth.")
        return z_parent

    sqrt_disc = np.sqrt(discriminant)
    z1 = (-b + sqrt_disc) / (2 * a)
    z2 = (-b - sqrt_disc) / (2 * a)

    # Pick the physically plausible root: positive depth, closest to the parent.
    candidates = [z for z in (z1, z2) if z > 0]
    if not candidates:
        return z_parent
    return min(candidates, key=lambda z: abs(z - z_parent))


def _estimate_root_depth(
    ray_root: np.ndarray,
    ray_ref: np.ndarray,
    pt_root: np.ndarray,
    pt_ref: np.ndarray,
    K: np.ndarray,
    true_length: float,
    fallback_depth: float,
) -> float:
    """
    Seed the root joint's absolute depth using similar triangles against a
    reference bone of known length (assumes the bone is roughly
    fronto-parallel; this is only an initial estimate, refined by the
    subsequent kinematic-chain solve).
    """
    pixel_distance = float(np.linalg.norm(pt_root - pt_ref))
    if pixel_distance < 1e-6:
        return fallback_depth
    focal = float((K[0, 0] + K[1, 1]) / 2.0)
    return focal * true_length / pixel_distance


def estimate_depth_monocular(
    keypoints: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    bone_lengths: Optional[Dict[Tuple[int, int], float]] = None,
    skeleton_tree: Optional[List[Tuple[int, int]]] = None,
    root_joint: int = ROOT_JOINT,
    min_confidence: float = 0.3,
    default_root_depth: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lift a single camera's 2D pose sequence into 3D using known bone
    lengths, in place of stereo triangulation.

    Args:
        keypoints:    (T, J, 2) 2D keypoints per frame from one camera.
        K:            (3, 3) camera intrinsic matrix for that camera.
        dist:         distortion coefficients for that camera.
        confidence:   (T, J) optional per-joint confidence scores.
        bone_lengths: {(parent, child): length_m} overrides for BONE_LENGTHS_M.
        skeleton_tree: overrides SKELETON_TREE (must be a tree over all J joints).
        root_joint:   index of the joint to anchor absolute depth to.
        min_confidence: joints (and their subtrees) below this are marked invalid.
        default_root_depth: fallback root depth (m) if no reference bone is
                             visible and no prior frame gave a valid root depth.

    Returns:
        poses_3d:   (T, J, 3) 3D joint positions in camera space (metres).
                    Invalid joints are NaN.
        valid_mask: (T, J) boolean array of which joints were solved.
    """
    bone_lengths = bone_lengths or BONE_LENGTHS_M
    skeleton_tree = skeleton_tree or SKELETON_TREE

    T, J, _ = keypoints.shape
    poses_3d = np.full((T, J, 3), np.nan, dtype=np.float64)
    valid_mask = np.zeros((T, J), dtype=bool)

    # The root's first tree edge doubles as the reference bone for its
    # initial absolute-depth estimate.
    root_ref_joint, root_ref_length = next(
        ((c, L) for (p, c), L in bone_lengths.items() if p == root_joint),
        (None, None),
    )

    last_valid_root_depth = default_root_depth

    for t in range(T):
        conf_t = confidence[t] if confidence is not None else np.ones(J)
        rays = _undistort_to_rays(keypoints[t], K, dist)

        if conf_t[root_joint] < min_confidence:
            # Can't anchor this frame's absolute depth; skip entirely.
            continue

        z_root = last_valid_root_depth
        if root_ref_joint is not None and conf_t[root_ref_joint] >= min_confidence:
            z_root = _estimate_root_depth(
                rays[root_joint], rays[root_ref_joint],
                keypoints[t, root_joint], keypoints[t, root_ref_joint],
                K, root_ref_length, last_valid_root_depth,
            )

        p_root = rays[root_joint] * z_root
        depths = {root_joint: z_root}
        positions = {root_joint: p_root}
        valid_mask[t, root_joint] = True
        poses_3d[t, root_joint] = p_root
        last_valid_root_depth = z_root

        for parent, child in skeleton_tree:
            if parent not in positions:
                continue  # parent's own subtree was invalidated
            if conf_t[child] < min_confidence:
                continue  # this joint (and anything below it) stays invalid

            length = bone_lengths.get((parent, child))
            if length is None:
                logger.warning("No bone length for edge (%d, %d); skipping.", parent, child)
                continue

            z_child = _solve_child_depth(rays[child], positions[parent], length, depths[parent])
            p_child = rays[child] * z_child

            depths[child] = z_child
            positions[child] = p_child
            poses_3d[t, child] = p_child
            valid_mask[t, child] = True

    n_valid = np.sum(valid_mask)
    n_total = T * J
    logger.info(
        "Monocular depth estimation complete: %d / %d joint-frames solved (%.1f%%)",
        n_valid, n_total, 100.0 * n_valid / max(n_total, 1),
    )

    return poses_3d, valid_mask


def estimate_monocular_pose_and_depth(
    video_path: str,
    K: np.ndarray,
    dist: np.ndarray,
    device: str = "cpu",
    max_frames: Optional[int] = None,
    bone_lengths: Optional[Dict[Tuple[int, int], float]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    estimator=None,
):
    """
    Convenience end-to-end helper: run 2D pose estimation on a single video,
    then lift it to 3D via known bone lengths. Mirrors the shape of
    src.processing.pose_estimation.estimate_stereo_poses +
    src.processing.triangulation.triangulate_pose_sequence, but for one camera.

    Returns (pose_result, poses_3d, valid_mask).
    """
    from src.processing.pose_estimation import PoseEstimator

    if estimator is None:
        estimator = PoseEstimator(device=device)
    pose_result = estimator.estimate_from_video(
        video_path, max_frames=max_frames, progress_callback=progress_callback
    )

    poses_3d, valid_mask = estimate_depth_monocular(
        pose_result.keypoints, K, dist,
        confidence=pose_result.confidence,
        bone_lengths=bone_lengths,
    )
    return pose_result, poses_3d, valid_mask


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Estimate 3D pose from a single camera video.")
    parser.add_argument("video", help="Path to an .mp4 video file.")
    parser.add_argument("--calibration", help="Path to a stereo_calibration.json (uses K_left/dist_left).")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--out", default=None, help="Path to save poses_3d as .npy")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.calibration:
        from src.processing.calibration import CalibrationResult
        cal = CalibrationResult.load(args.calibration)
        K, dist = cal.K_left, cal.dist_left
    else:
        logger.warning("No --calibration given; using a rough default intrinsic guess.")
        K = np.array([[1000.0, 0, 640.0], [0, 1000.0, 360.0], [0, 0, 1.0]])
        dist = np.zeros(5)

    pose_result, poses_3d, valid_mask = estimate_monocular_pose_and_depth(
        args.video, K, dist, device=args.device, max_frames=args.max_frames,
    )

    print(f"Frames: {pose_result.num_frames}, joints: {pose_result.num_joints}")
    print(f"Valid joint-frames: {valid_mask.sum()} / {valid_mask.size}")
    for j, name in enumerate(COCO_KEYPOINT_NAMES):
        valid_frac = valid_mask[:, j].mean()
        print(f"  {name:16s} valid={valid_frac:.0%}")

    if args.out:
        np.save(args.out, poses_3d)
        print(f"Saved poses_3d {poses_3d.shape} to {args.out}")
