"""
Biomechanics Feature Extraction — 3D Pose Array API

Computes kinematic features from triangulated 3D pose sequences:
    - Smoothness (Log Dimensionless Jerk — LDLJ)
    - Spectral Arc Length (SPARC)
    - Symmetry Index
    - Periodicity / gait regularity
    - Range of Motion

All functions accept a (T, J, 3) pose array and a scalar fps value
instead of the legacy DataFrame-based API.
"""

import numpy as np
from typing import Dict, Any
from scipy.signal import correlate, find_peaks

# np.trapezoid was added in NumPy 2.0 as the replacement for the deprecated
# np.trapz; support both so this module works on either NumPy major version.
_trapezoid = getattr(np, 'trapezoid', None) or np.trapz


def compute_smoothness_3d(
    poses_3d: np.ndarray,
    fps: float,
    valid_mask: np.ndarray | None = None,
) -> Dict[str, Any]:
    """
    Computes Log Dimensionless Jerk (LDLJ) on 3D joint trajectories.
    Scale-invariant. Higher (less negative) = smoother.

    Args:
        poses_3d:   (T, J, 3) array.
        fps:        Frames per second.
        valid_mask: (T, J) boolean mask. Joints marked False are interpolated.

    Returns:
        Dict with mean_ldlj and per-joint values.
    """
    T, J, _ = poses_3d.shape
    if T < 4 or fps <= 0:
        return {"mean_ldlj": 0.0, "status": "insufficient_data"}

    dt = 1.0 / fps
    poses = _interpolate_nans(poses_3d.copy())

    ldlj_per_joint = []
    for j in range(J):
        traj = poses[:, j, :]  # (T, 3)

        # Velocity, acceleration, jerk (central differences)
        v = np.gradient(traj, dt, axis=0)  # (T, 3)
        a = np.gradient(v, dt, axis=0)
        jerk = np.gradient(a, dt, axis=0)

        speed = np.linalg.norm(v, axis=1)
        v_peak = np.max(speed)

        if v_peak < 1e-4:
            ldlj_per_joint.append(0.0)
            continue

        duration = T * dt
        jerk_sq = np.sum(jerk ** 2, axis=1)  # scalar jerk magnitude squared
        jerk_integral = _trapezoid(jerk_sq, dx=dt)

        term = (duration ** 3 / v_peak ** 2) * jerk_integral
        ldlj = -np.log(max(term, 1e-12))
        ldlj_per_joint.append(float(ldlj))

    mean_ldlj = float(np.mean(ldlj_per_joint)) if ldlj_per_joint else 0.0
    return {"mean_ldlj": mean_ldlj, "ldlj_per_joint": ldlj_per_joint}


def compute_sparc_3d(
    poses_3d: np.ndarray,
    fps: float,
) -> Dict[str, Any]:
    """
    Computes Spectral Arc Length (SPARC) on 3D velocity profiles.
    Shorter arc length (closer to 0) = smoother.

    Args:
        poses_3d: (T, J, 3) array.
        fps:      Frames per second.

    Returns:
        Dict with mean_sparc and per-joint values.
    """
    T, J, _ = poses_3d.shape
    if T < 4 or fps <= 0:
        return {"mean_sparc": 0.0, "status": "insufficient_data"}

    dt = 1.0 / fps
    poses = _interpolate_nans(poses_3d.copy())

    sparc_per_joint = []
    for j in range(J):
        v = np.gradient(poses[:, j, :], dt, axis=0)
        speed = np.linalg.norm(v, axis=1)  # (T,)

        if np.max(speed) < 1e-4:
            sparc_per_joint.append(0.0)
            continue

        f_mag = np.abs(np.fft.rfft(speed))
        if f_mag[0] == 0:
            f_mag[0] = 1e-6
        f_norm = f_mag / f_mag[0]

        df = np.diff(f_norm)
        dw = 1.0 / len(f_norm)
        arc_len = np.sum(np.sqrt(dw ** 2 + df ** 2))
        sparc_per_joint.append(-float(arc_len))

    mean_sparc = float(np.mean(sparc_per_joint)) if sparc_per_joint else 0.0
    return {"mean_sparc": mean_sparc, "sparc_per_joint": sparc_per_joint}


def compute_symmetry_3d(
    poses_3d: np.ndarray,
    left_joints: list | None = None,
    right_joints: list | None = None,
) -> Dict[str, Any]:
    """
    Computes Symmetry Index (SI) between left and right joint pairs.

    Default uses COCO keypoint pairing:
        L/R shoulder (5,6), L/R elbow (7,8), L/R wrist (9,10),
        L/R hip (11,12), L/R knee (13,14), L/R ankle (15,16).

    Returns SI in percentage (0% = perfect symmetry).
    """
    if left_joints is None:
        left_joints = [5, 7, 9, 11, 13, 15]
    if right_joints is None:
        right_joints = [6, 8, 10, 12, 14, 16]

    poses = _interpolate_nans(poses_3d.copy())
    si_values = []

    for lj, rj in zip(left_joints, right_joints):
        l_rom = np.ptp(poses[:, lj, :], axis=0)  # (3,) ROM per axis
        r_rom = np.ptp(poses[:, rj, :], axis=0)

        l_total = np.linalg.norm(l_rom)
        r_total = np.linalg.norm(r_rom)
        denom = max(l_total, r_total)

        if denom < 1e-4:
            continue

        si = (abs(l_total - r_total) / denom) * 100.0
        si_values.append(si)

    mean_si = float(np.mean(si_values)) if si_values else 0.0
    return {"mean_symmetry_index": mean_si, "si_per_pair": si_values}


def compute_periodicity_3d(
    poses_3d: np.ndarray,
    fps: float,
) -> Dict[str, Any]:
    """
    Autocorrelation-based periodicity analysis on 3D joint trajectories.
    Returns a regularity score (height of first autocorrelation peak).
    """
    T, J, _ = poses_3d.shape
    if T < 20 or fps <= 0:
        return {"regularity_score": 0.0, "status": "insufficient_data"}

    poses = _interpolate_nans(poses_3d.copy())

    # Compute speed profile averaged across all joints
    dt = 1.0 / fps
    v = np.gradient(poses, dt, axis=0)  # (T, J, 3)
    speeds = np.linalg.norm(v, axis=2)  # (T, J)
    global_speed = np.mean(speeds, axis=1)  # (T,)

    if np.var(global_speed) < 1e-8:
        return {"regularity_score": 0.0}

    signal = global_speed - np.mean(global_speed)
    autocorr = correlate(signal, signal, mode="full")
    autocorr = autocorr[len(autocorr) // 2:]
    if autocorr[0] > 0:
        autocorr /= autocorr[0]

    min_dist = max(5, int(fps * 0.3))
    peaks, _ = find_peaks(autocorr, distance=min_dist, height=0.1)

    if len(peaks) > 0:
        return {"regularity_score": float(autocorr[peaks[0]])}
    return {"regularity_score": 0.0}


def compute_rom_3d(poses_3d: np.ndarray) -> Dict[str, Any]:
    """
    Range of Motion for each joint in 3D space (metres).
    """
    poses = _interpolate_nans(poses_3d.copy())
    T, J, _ = poses.shape

    rom_per_joint = []
    for j in range(J):
        rom = np.max(np.ptp(poses[:, j, :], axis=0))
        rom_per_joint.append(float(rom))

    mean_rom = float(np.mean(rom_per_joint)) if rom_per_joint else 0.0
    return {"mean_rom": mean_rom, "rom_per_joint": rom_per_joint}


def compute_jumping_metrics_3d(
    poses_3d: np.ndarray,
    fps: float,
) -> Dict[str, Any]:
    """
    Heuristic estimation of jumping metrics using 3D joint trajectories.
    Assumes jumping is a full-body movement involving rapid vertical acceleration.
    """
    T, J, _ = poses_3d.shape
    if T < 4 or fps <= 0:
        return {"flight_time": 0.0, "peak_z_accel": 0.0, "landing_jerk": 0.0}

    dt = 1.0 / fps
    poses = _interpolate_nans(poses_3d.copy())

    # We use the vertical axis (Y or Z, depending on convention).
    # Assuming Y is up, which is index 1.
    vertical_poses = poses[:, :, 1] # (T, J)
    global_vertical = np.mean(vertical_poses, axis=1) # (T,)

    v = np.gradient(global_vertical, dt)
    a = np.gradient(v, dt)
    j = np.gradient(a, dt)

    abs_a = np.abs(a)
    abs_j = np.abs(j)

    peak_z_accel = float(np.max(abs_a))
    landing_jerk = float(np.max(abs_j))

    # Flight time: period where velocity is very low AFTER a huge acceleration peak
    push_off_idx = np.argmax(abs_a)
    landing_jerk_idx = np.argmax(abs_j[push_off_idx:]) + push_off_idx

    if landing_jerk_idx > push_off_idx:
        flight_time = float((landing_jerk_idx - push_off_idx) * dt)
    else:
        flight_time = 0.0

    return {
        "flight_time": max(0.0, flight_time),
        "peak_z_accel": peak_z_accel,
        "landing_jerk": landing_jerk
    }


def compute_transition_metrics_3d(
    poses_3d: np.ndarray,
    fps: float,
) -> Dict[str, Any]:
    """
    Heuristic estimation for Stand <-> Sit transitions from 3D joint trajectories.
    """
    T, J, _ = poses_3d.shape
    if T < 4 or fps <= 0:
        return {"com_oscillation": 0.0, "transition_time": 0.0}

    dt = 1.0 / fps
    poses = _interpolate_nans(poses_3d.copy())

    # Use vertical axis
    vertical_poses = poses[:, :, 1]
    global_vertical = np.mean(vertical_poses, axis=1)

    v = np.gradient(global_vertical, dt)
    abs_v = np.abs(v)

    threshold = np.max(abs_v) * 0.15
    active_indices = np.where(abs_v > threshold)[0]

    if len(active_indices) > 0:
        transition_time = float((active_indices[-1] - active_indices[0]) * dt)
    else:
        transition_time = 0.0

    if len(active_indices) > 2:
        active_v = v[active_indices]
        com_oscillation = float(np.var(active_v))
    else:
        com_oscillation = 0.0

    return {
        "com_oscillation": com_oscillation,
        "transition_time": max(0.0, transition_time)
    }


# ── Utility ─────────────────────────────────────────────────────────────────

def _interpolate_nans(arr: np.ndarray) -> np.ndarray:
    """
    Linearly interpolate NaN values along the time axis (axis 0).
    Handles the (T, J, 3) shape by flattening to (T, J*3).
    """
    T = arr.shape[0]
    flat = arr.reshape(T, -1)

    for col in range(flat.shape[1]):
        series = flat[:, col]
        nans = np.isnan(series)
        if nans.all() or not nans.any():
            continue
        not_nan = ~nans
        indices = np.arange(T)
        series[nans] = np.interp(indices[nans], indices[not_nan], series[not_nan])

    return flat.reshape(arr.shape)


import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from scipy.signal import correlate, find_peaks

def _get_q_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extracts the 'q' column into a 2D numpy array (time x joints)."""
    if 'q' not in df.columns or df.empty:
        return np.array([])
    # Handle cases where q is a list/array
    q_data = df['q'].tolist()
    return np.array(q_data)

def _get_time_seconds(df: pd.DataFrame) -> np.ndarray:
    """Extracts time in seconds from 'tick'."""
    if 'tick' not in df.columns or df.empty:
        return np.array([])
    return df['tick'].to_numpy() / 1000.0

def compute_smoothness(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes Log Dimensionless Jerk (LDLJ).
    Scale-invariant across speeds. Higher (less negative) is smoother.
    """
    q_mat = _get_q_matrix(df)
    t_sec = _get_time_seconds(df)
    
    if q_mat.size == 0 or len(t_sec) < 4:
        return {"mean_ldlj": 0.0, "status": "insufficient_data"}
        
    dt = np.gradient(t_sec)
    dt[dt == 0] = 1e-6 # prevent div by zero
    
    # 1st deriv (Velocity)
    v = np.gradient(q_mat, axis=0) / dt[:, np.newaxis]
    # 2nd deriv (Acceleration)
    a = np.gradient(v, axis=0) / dt[:, np.newaxis]
    # 3rd deriv (Jerk)
    j = np.gradient(a, axis=0) / dt[:, np.newaxis]
    
    duration = t_sec[-1] - t_sec[0]
    
    ldlj_per_joint = []
    for col in range(q_mat.shape[1]):
        v_peak = np.max(np.abs(v[:, col]))
        if v_peak < 1e-3:
            # Not moving
            ldlj_per_joint.append(0.0)
            continue
            
        jerk_sq_integral = _trapezoid(j[:, col]**2, t_sec)
        
        # LDLJ formula: -ln( (D^3 / v_peak^2) * integral(j^2 dt) )
        term = (duration**3 / (v_peak**2)) * jerk_sq_integral
        if term <= 0:
            ldlj = 0.0
        else:
            ldlj = -np.log(term)
        ldlj_per_joint.append(ldlj)
        
    mean_ldlj = float(np.mean(ldlj_per_joint)) if ldlj_per_joint else 0.0
    
    return {
        "mean_ldlj": mean_ldlj,
        "ldlj_per_joint": ldlj_per_joint
    }

def compute_spectral_arc_length(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes Spectral Arc Length (SPARC) on the velocity profile.
    Shorter arc length = smoother movement.
    """
    q_mat = _get_q_matrix(df)
    t_sec = _get_time_seconds(df)
    
    if q_mat.size == 0 or len(t_sec) < 4:
        return {"mean_sparc": 0.0, "status": "insufficient_data"}
        
    dt = np.gradient(t_sec)
    dt[dt == 0] = 1e-6
    v = np.gradient(q_mat, axis=0) / dt[:, np.newaxis]
    
    sparc_per_joint = []
    for col in range(v.shape[1]):
        sig = v[:, col]
        if np.max(np.abs(sig)) < 1e-3:
            sparc_per_joint.append(0.0)
            continue
            
        # FFT
        f_mag = np.abs(np.fft.rfft(sig))
        if f_mag[0] == 0:
            f_mag[0] = 1e-6
        f_mag_norm = f_mag / f_mag[0] # normalize to DC component
        
        # Compute arc length
        df_mag = np.diff(f_mag_norm)
        # Normalize frequency axis from 0 to 1
        dw = 1.0 / len(f_mag_norm) 
        
        arc_len = np.sum(np.sqrt(dw**2 + df_mag**2))
        # Negative because traditionally higher SPARC (closer to 0) is smoother
        sparc_per_joint.append(-float(arc_len))
        
    mean_sparc = float(np.mean(sparc_per_joint)) if sparc_per_joint else 0.0
    
    return {
        "mean_sparc": mean_sparc,
        "sparc_per_joint": sparc_per_joint
    }

def compute_symmetry(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes Symmetry Index (SI) between assumed left and right joints.
    If exact mapping isn't known, splits the joint array in half as a heuristic.
    Returns SI in percentage (0% = perfect symmetry).
    """
    q_mat = _get_q_matrix(df)
    
    if q_mat.size == 0 or q_mat.shape[1] < 2:
        return {"mean_symmetry_index": None, "status": "unavailable"}
        
    n_joints = q_mat.shape[1]
    half = n_joints // 2
    
    # Left and Right heuristic
    left_q = q_mat[:, :half]
    right_q = q_mat[:, half:half*2] # ignores the middle joint if odd
    
    # Calculate range of motion as the variable to compare
    left_rom = np.ptp(left_q, axis=0)
    right_rom = np.ptp(right_q, axis=0)
    
    si_values = []
    for l_val, r_val in zip(left_rom, right_rom):
        denom = max(l_val, r_val)
        if denom < 1e-3:
            continue
        si = (np.abs(l_val - r_val) / denom) * 100.0
        si_values.append(si)
        
    if not si_values:
        return {"mean_symmetry_index": 0.0}
        
    mean_si = float(np.mean(si_values))
    
    return {
        "mean_symmetry_index": mean_si,
        "si_per_pair": si_values
    }

def compute_periodicity(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Uses autocorrelation of joint movements to determine periodicity.
    Returns a regularity score (height of first peak).
    """
    q_mat = _get_q_matrix(df)
    
    if q_mat.size == 0 or q_mat.shape[0] < 10:
        return {"regularity_score": 0.0, "status": "insufficient_data"}
        
    # Average movement across all joints to find the global gait cycle
    # We use velocity to remove static offsets
    v_mat = np.gradient(q_mat, axis=0)
    global_movement = np.mean(np.abs(v_mat), axis=1)
    
    # Normalize
    if np.var(global_movement) < 1e-6:
        return {"regularity_score": 0.0}
        
    global_movement = global_movement - np.mean(global_movement)
    
    # Autocorrelation
    autocorr = correlate(global_movement, global_movement, mode='full')
    autocorr = autocorr[len(autocorr)//2:] # keep positive lags
    autocorr = autocorr / autocorr[0] # normalize to 1.0 at lag 0
    
    # Find peaks (min distance to avoid finding the lag 0 peak itself)
    peaks, properties = find_peaks(autocorr, distance=10, height=0.1)
    
    if len(peaks) > 0:
        first_peak_height = autocorr[peaks[0]]
        return {"regularity_score": float(first_peak_height)}
    else:
        return {"regularity_score": 0.0}

def compute_range_of_motion(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes angular range of motion for each joint.
    Returns ROM in radians and an overall utilisation score.
    """
    q_mat = _get_q_matrix(df)
    
    if q_mat.size == 0:
        return {"mean_rom": 0.0, "rom_per_joint": []}
        
    rom_per_joint = np.ptp(q_mat, axis=0).tolist()
    mean_rom = float(np.mean(rom_per_joint))
    
    return {
        "mean_rom": mean_rom,
        "rom_per_joint": rom_per_joint
    }

def compute_jumping_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Heuristic estimation of jumping metrics using only joint trajectories.
    Assumes rapid synchronous leg extension = jump, followed by impact.
    """
    q_mat = _get_q_matrix(df)
    t_sec = _get_time_seconds(df)
    
    if q_mat.size == 0 or len(t_sec) < 4:
        return {"flight_time": 0.0, "peak_z_accel": 0.0, "landing_jerk": 0.0}
        
    dt = np.gradient(t_sec)
    dt[dt == 0] = 1e-6
    v = np.gradient(q_mat, axis=0) / dt[:, np.newaxis]
    a = np.gradient(v, axis=0) / dt[:, np.newaxis]
    j = np.gradient(a, axis=0) / dt[:, np.newaxis]
    
    # Heuristic: use average absolute leg joint movement to represent vertical power
    # We'll just use the global average variance for simplicity since we don't have explicit joint names here,
    # but jumping is a full-body explosive movement.
    global_accel = np.mean(np.abs(a), axis=1)
    global_jerk = np.mean(np.abs(j), axis=1)
    global_vel = np.mean(np.abs(v), axis=1)
    
    peak_z_accel = float(np.max(global_accel))
    landing_jerk = float(np.max(global_jerk))
    
    # Flight time: period where velocity is very low AFTER a huge acceleration peak
    # Find the peak acceleration (push-off)
    push_off_idx = np.argmax(global_accel)
    # Find the peak jerk (landing impact) AFTER push off
    landing_jerk_idx = np.argmax(global_jerk[push_off_idx:]) + push_off_idx
    
    if landing_jerk_idx > push_off_idx:
        flight_time = float(t_sec[landing_jerk_idx] - t_sec[push_off_idx])
    else:
        flight_time = 0.0
        
    return {
        "flight_time": max(0.0, flight_time),
        "peak_z_accel": peak_z_accel,
        "landing_jerk": landing_jerk
    }

def compute_transition_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Heuristic estimation for Stand <-> Sit transitions.
    """
    q_mat = _get_q_matrix(df)
    t_sec = _get_time_seconds(df)
    
    if q_mat.size == 0 or len(t_sec) < 4:
        return {"com_oscillation": 0.0, "transition_time": 0.0}
        
    dt = np.gradient(t_sec)
    dt[dt == 0] = 1e-6
    v = np.gradient(q_mat, axis=0) / dt[:, np.newaxis]
    
    global_vel = np.mean(np.abs(v), axis=1)
    
    # Transition time: duration where global velocity is above a small threshold
    threshold = np.max(global_vel) * 0.15 # 15% of peak velocity
    active_indices = np.where(global_vel > threshold)[0]
    
    if len(active_indices) > 0:
        transition_time = float(t_sec[active_indices[-1]] - t_sec[active_indices[0]])
    else:
        transition_time = 0.0
        
    # CoM oscillation: Variance of the derivatives (wobble) during the transition
    if len(active_indices) > 2:
        active_v = v[active_indices, :]
        com_oscillation = float(np.mean(np.var(active_v, axis=0)))
    else:
        com_oscillation = 0.0
        
    return {
        "com_oscillation": com_oscillation,
        "transition_time": max(0.0, transition_time)
    }

def compute_walk_grade_3d(
    poses_3d: np.ndarray,
    fps: float,
) -> Dict[str, Any]:
    pass

def compute_walk_grade_3d(
    poses_3d: np.ndarray,
    fps: float,
    keypoints_2d: Optional[np.ndarray] = None,
    confidence_2d: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Grading the walking cycle of the robot based on clearance, stride symmetry, speed, and torso levelness.
    Poses are in (T, 17, 3), coordinates: X=right, Y=up, Z=towards-camera (meters)
    """
    T, J, _ = poses_3d.shape
    if T < 10 or fps <= 0:
        return {
            "walk_grade": 0.0, 
            "status": "insufficient_data",
            "mean_clearance_cm": 0.0,
            "stride_length_m": 0.0,
            "speed_m_s": 0.0,
            "torso_oscillation_cm": 0.0
        }
        
    dt = 1.0 / fps
    poses = _interpolate_nans(poses_3d.copy())
    
    # 1. Torso Levelness & Fall Detection (10%)
    pelvis_y = (poses[:, 11, 1] + poses[:, 12, 1]) / 2.0
    lowest_ankle_y = np.minimum(poses[:, 15, 1], poses[:, 16, 1])
    pelvis_clearance = pelvis_y - lowest_ankle_y
    
    # Filter out false falls from objects (like chairs) detected when person walks out of frame
    fall_detected = False
    
    if keypoints_2d is not None and confidence_2d is not None:
        # A real person will have > 0.3 mean confidence across all joints
        valid_frames = np.mean(confidence_2d, axis=1) > 0.3
        
        # Check for teleportation (tracking switch to background object)
        com_2d = np.nanmean(keypoints_2d, axis=1) # (T, 2)
        dist = np.linalg.norm(np.diff(com_2d, axis=0), axis=1)
        dist = np.insert(dist, 0, 0.0) # pad to length T
        
        # If the tracking center shifts > 150 pixels in one frame, it's a tracking switch.
        # We invalidate that frame and ALL subsequent frames!
        has_teleported = np.cumsum(dist > 150.0) > 0
        valid_frames = valid_frames & (~has_teleported)
        
        if np.any(valid_frames):
            fall_detected = bool(np.min(pelvis_clearance[valid_frames]) < 0.2)
    else:
        fall_detected = bool(np.min(pelvis_clearance) < 0.2)
    
    torso_oscillation = np.std(pelvis_y)
    
    torso_score = 100.0
    if torso_oscillation > 0.02:
        penalty = ((torso_oscillation - 0.02) / 0.03) * 100.0
        torso_score = max(0.0, 100.0 - penalty)
        
    # 2. Foot Clearance (35%)
    def analyze_foot(y_traj):
        peaks, _ = find_peaks(y_traj, distance=int(fps*0.3), prominence=0.01)
        clearances = []
        for p in peaks:
            left = max(0, p - int(fps*0.5))
            right = min(T, p + int(fps*0.5))
            min_y = np.min(y_traj[left:right])
            clearances.append(y_traj[p] - min_y)
        return clearances

    left_clearances = analyze_foot(poses[:, 15, 1])
    right_clearances = analyze_foot(poses[:, 16, 1])
    all_clearances = left_clearances + right_clearances
    
    clearance_score = 100.0
    mean_clearance = 0.0
    if not all_clearances:
        clearance_score = 0.0
    else:
        mean_clearance = float(np.mean(all_clearances))
        if mean_clearance < 0.02: # Shuffle
            clearance_score -= 50.0 
        elif mean_clearance > 0.06: # Wasting energy
            clearance_score -= 30.0 
            
        std_clearance = float(np.std(all_clearances))
        if std_clearance > 0.005:
            clearance_score -= 50.0 
        elif std_clearance > 0.0025:
            penalty = ((std_clearance - 0.0025) / 0.0025) * 50.0
            clearance_score -= penalty
            
    clearance_score = max(0.0, clearance_score)

    # 3. Stride Length & Symmetry (35%)
    def find_valleys(y_traj):
        valleys, _ = find_peaks(-y_traj, distance=int(fps*0.3), prominence=0.01)
        return valleys
        
    left_valleys = find_valleys(poses[:, 15, 1])
    right_valleys = find_valleys(poses[:, 16, 1])
    
    stride_lengths = []
    for v in left_valleys:
        dist = np.linalg.norm(poses[v, 15, [0, 2]] - poses[v, 16, [0, 2]])
        stride_lengths.append(('L', dist))
    for v in right_valleys:
        dist = np.linalg.norm(poses[v, 15, [0, 2]] - poses[v, 16, [0, 2]])
        stride_lengths.append(('R', dist))
        
    symmetry_score = 100.0
    mean_stride = 0.0
    l_strides = [s[1] for s in stride_lengths if s[0] == 'L']
    r_strides = [s[1] for s in stride_lengths if s[0] == 'R']
    
    if l_strides and r_strides:
        mean_l = np.mean(l_strides)
        mean_r = np.mean(r_strides)
        mean_stride = float((mean_l + mean_r) / 2.0)
        
        diff = abs(mean_l - mean_r)
        if mean_stride > 0:
            asym_ratio = diff / mean_stride
            if asym_ratio > 0.05: 
                penalty = min(100.0, ((asym_ratio - 0.05) / 0.15) * 100.0)
                symmetry_score -= penalty
    else:
        symmetry_score = 0.0
        
    symmetry_score = max(0.0, symmetry_score)

    # 4. Speed (20%)
    pelvis_xz = (poses[:, 11, [0, 2]] + poses[:, 12, [0, 2]]) / 2.0
    dist_travelled = 0.0
    for i in range(1, T):
        dist_travelled += np.linalg.norm(pelvis_xz[i] - pelvis_xz[i-1])
    
    duration = T * dt
    speed = float(dist_travelled / duration) if duration > 0 else 0.0
    
    speed_score = 100.0
    if speed < 0.2:
        speed_score = 0.0
    elif speed < 0.4:
        speed_score = ((speed - 0.2) / 0.2) * 100.0
        
    final_score = (
        (clearance_score * 0.35) +
        (symmetry_score * 0.35) +
        (speed_score * 0.20) +
        (torso_score * 0.10)
    )
    
    return {
        "walk_grade": float(final_score),
        "mean_clearance_cm": mean_clearance * 100.0,
        "stride_length_m": mean_stride,
        "speed_m_s": speed,
        "torso_oscillation_cm": float(torso_oscillation * 100.0),
        "clearance_score": float(clearance_score),
        "symmetry_score": float(symmetry_score),
        "speed_score": float(speed_score),
        "torso_score": float(torso_score),
        "fall_detected": fall_detected
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2D Walk Grade — pixel-based grading with reference-length scaling
# ──────────────────────────────────────────────────────────────────────────────

# COCO keypoint indices used by ViTPose
_COCO_L_SHOULDER = 5
_COCO_R_SHOULDER = 6
_COCO_L_ELBOW = 7
_COCO_R_ELBOW = 8
_COCO_L_HIP = 11
_COCO_R_HIP = 12
_COCO_L_ANKLE = 15
_COCO_R_ANKLE = 16


def _pixel_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two 2D points."""
    return float(np.linalg.norm(a - b))


def compute_walk_grade_2d(
    keypoints: np.ndarray,
    confidence: np.ndarray,
    fps: float,
    ref_length_cm: float = 20.0,
    conf_threshold: float = 0.3,
    view: str = "side",
) -> Dict[str, Any]:
    """Grade a walk cycle from 2D pixel keypoints.

    Parameters
    ----------
    keypoints : (T, J, 2)
        Per-frame COCO keypoints in pixel coordinates (x, y).
    confidence : (T, J)
        Per-frame joint confidence scores.
    fps : float
        Video frame rate.
    ref_length_cm : float
        The physical distance (in cm) between the shoulder and elbow joints.
        Used to convert pixel distances to real-world units.
    conf_threshold : float
        Minimum confidence to consider a joint detection valid.
    view : str
        "side" or "front". Determines which metrics to extract.

    Returns
    -------
    dict with walk_grade (0-100), sub-scores, and metric values in cm / m / m/s.
    """
    T = keypoints.shape[0]
    empty = {
        "walk_grade": 0.0,
        "status": "insufficient_data",
        "mean_clearance_cm": 0.0,
        "stride_length_m": 0.0,
        "speed_m_s": 0.0,
        "torso_oscillation_cm": 0.0,
        "clearance_score": 0.0,
        "symmetry_score": 0.0,
        "speed_score": 0.0,
        "torso_score": 0.0,
        "cm_per_pixel": 0.0,
        "fall_detected": False,
    }
    if T < 10 or fps <= 0:
        return empty

    # ── Compute cm-per-pixel scale from shoulder-to-elbow reference ──
    ref_pixels = []
    for t in range(T):
        # Try left arm
        if (confidence[t, _COCO_L_SHOULDER] >= conf_threshold and
                confidence[t, _COCO_L_ELBOW] >= conf_threshold):
            d = _pixel_dist(keypoints[t, _COCO_L_SHOULDER],
                            keypoints[t, _COCO_L_ELBOW])
            if d > 1.0:
                ref_pixels.append(d)
        # Try right arm
        if (confidence[t, _COCO_R_SHOULDER] >= conf_threshold and
                confidence[t, _COCO_R_ELBOW] >= conf_threshold):
            d = _pixel_dist(keypoints[t, _COCO_R_SHOULDER],
                            keypoints[t, _COCO_R_ELBOW])
            if d > 1.0:
                ref_pixels.append(d)

    if len(ref_pixels) < 5:
        empty["status"] = "reference_too_few_detections"
        return empty

    median_ref_px = float(np.median(ref_pixels))
    cm_per_px = ref_length_cm / median_ref_px

    # ── Build per-frame joint positions in cm (with NaN for low-conf) ──
    kp_cm = keypoints.copy().astype(np.float64)
    for t in range(T):
        for j in range(keypoints.shape[1]):
            if confidence[t, j] < conf_threshold:
                kp_cm[t, j] = [np.nan, np.nan]
    kp_cm *= cm_per_px  # convert pixel -> cm

    # Linear-interpolate NaN gaps per joint
    for j in range(kp_cm.shape[1]):
        for axis in range(2):
            traj = kp_cm[:, j, axis]
            nans = np.isnan(traj)
            if nans.all():
                continue
            good = ~nans
            traj[nans] = np.interp(np.where(nans)[0], np.where(good)[0], traj[good])

    dt = 1.0 / fps

    # ── 1. Torso Levelness & Sway (10-40%) ──
    # Average Y of left and right hip across frames
    hip_y = (kp_cm[:, _COCO_L_HIP, 1] + kp_cm[:, _COCO_R_HIP, 1]) / 2.0
    lowest_ankle_y = np.maximum(kp_cm[:, _COCO_L_ANKLE, 1], kp_cm[:, _COCO_R_ANKLE, 1])
    pelvis_clearance = lowest_ankle_y - hip_y
    
    # Filter out false falls from background objects tracking
    valid_frames = np.mean(confidence, axis=1) > 0.3
    
    # Check for teleportation (tracking switch to background object)
    com_2d = np.nanmean(keypoints, axis=1) # (T, 2)
    dist = np.linalg.norm(np.diff(com_2d, axis=0), axis=1)
    dist = np.insert(dist, 0, 0.0) # pad to length T
    
    # If the tracking center shifts > 150 pixels in one frame, it's a tracking switch.
    # We invalidate that frame and ALL subsequent frames!
    has_teleported = np.cumsum(dist > 150.0) > 0
    valid_frames = valid_frames & (~has_teleported)
    
    if np.any(valid_frames):
        fall_detected = bool(np.min(pelvis_clearance[valid_frames]) < 20.0)
    else:
        fall_detected = False
    
    torso_osc_cm = float(np.std(hip_y))
    
    torso_score = 100.0
    # > 2cm oscillation starts penalty; > 5cm = 0
    if torso_osc_cm > 2.0:
        penalty = ((torso_osc_cm - 2.0) / 3.0) * 100.0
        torso_score = max(0.0, 100.0 - penalty)
        
    lateral_sway_cm = 0.0
    sway_score = 100.0
    if view == "front":
        hip_x = (kp_cm[:, _COCO_L_HIP, 0] + kp_cm[:, _COCO_R_HIP, 0]) / 2.0
        lateral_sway_cm = float(np.std(hip_x))
        if lateral_sway_cm > 2.0:
            penalty = ((lateral_sway_cm - 2.0) / 4.0) * 100.0
            sway_score = max(0.0, 100.0 - penalty)

    # ── 2. Foot Clearance (35%) ──
    # In 2D pixel space, Y increases downward, so foot *up* = lower Y.
    # We look at negative-Y peaks (highest foot lift) relative to nearby stance.
    def analyze_foot_2d(y_traj_cm):
        """Find swing-phase peaks and compute clearance in cm."""
        # Invert Y so peaks correspond to foot lifts
        inv = -y_traj_cm
        peaks, _ = find_peaks(inv, distance=int(fps * 0.3), prominence=0.5)
        clearances = []
        for p in peaks:
            left_idx = max(0, p - int(fps * 0.5))
            right_idx = min(T, p + int(fps * 0.5))
            # Ground level = max Y (lowest point) in local window
            ground_y = np.max(y_traj_cm[left_idx:right_idx])
            clearance = ground_y - y_traj_cm[p]
            if clearance > 0:
                clearances.append(clearance)
        return clearances

    l_clear = analyze_foot_2d(kp_cm[:, _COCO_L_ANKLE, 1])
    r_clear = analyze_foot_2d(kp_cm[:, _COCO_R_ANKLE, 1])
    all_clearances = l_clear + r_clear

    clearance_score = 100.0
    mean_clearance_cm = 0.0
    if not all_clearances:
        clearance_score = 0.0
    else:
        mean_clearance_cm = float(np.mean(all_clearances))
        # < 2cm is a shuffle
        if mean_clearance_cm < 2.0:
            clearance_score -= 50.0
        # > 6cm is wasting energy
        elif mean_clearance_cm > 6.0:
            clearance_score -= 30.0

        std_clearance = float(np.std(all_clearances))
        # Inconsistent clearance penalised
        if std_clearance > 0.5:
            clearance_score -= 50.0
        elif std_clearance > 0.25:
            penalty = ((std_clearance - 0.25) / 0.25) * 50.0
            clearance_score -= penalty

    clearance_score = max(0.0, clearance_score)

    # ── 3. Stride Length / Step Width & Symmetry (35%) ──
    def find_stance_frames(y_traj_cm):
        """Find frames where the foot is on the ground (valleys in inverted Y)."""
        valleys, _ = find_peaks(y_traj_cm, distance=int(fps * 0.3), prominence=0.5)
        return valleys

    l_stance = find_stance_frames(kp_cm[:, _COCO_L_ANKLE, 1])
    r_stance = find_stance_frames(kp_cm[:, _COCO_R_ANKLE, 1])

    stride_lengths = []
    for v in l_stance:
        dist = abs(kp_cm[v, _COCO_L_ANKLE, 0] - kp_cm[v, _COCO_R_ANKLE, 0])
        stride_lengths.append(("L", dist))
    for v in r_stance:
        dist = abs(kp_cm[v, _COCO_L_ANKLE, 0] - kp_cm[v, _COCO_R_ANKLE, 0])
        stride_lengths.append(("R", dist))

    symmetry_score = 100.0
    mean_stride_cm = 0.0
    l_strides = [s[1] for s in stride_lengths if s[0] == "L"]
    r_strides = [s[1] for s in stride_lengths if s[0] == "R"]

    if l_strides and r_strides:
        mean_l = np.mean(l_strides)
        mean_r = np.mean(r_strides)
        mean_stride_cm = float((mean_l + mean_r) / 2.0)

        diff = abs(mean_l - mean_r)
        if mean_stride_cm > 0:
            asym_ratio = diff / mean_stride_cm
            if asym_ratio > 0.05:
                penalty = min(100.0, ((asym_ratio - 0.05) / 0.15) * 100.0)
                symmetry_score -= penalty
    else:
        symmetry_score = 0.0

    symmetry_score = max(0.0, symmetry_score)

    # ── 4. Speed (20%) ──
    speed_m_s = 0.0
    speed_score = 100.0
    
    if view == "side":
        # Horizontal displacement of hip midpoint per second
        hip_x = (kp_cm[:, _COCO_L_HIP, 0] + kp_cm[:, _COCO_R_HIP, 0]) / 2.0
        dist_travelled_cm = 0.0
        for i in range(1, T):
            dist_travelled_cm += abs(hip_x[i] - hip_x[i - 1])

        duration = T * dt
        speed_cm_s = float(dist_travelled_cm / duration) if duration > 0 else 0.0
        speed_m_s = speed_cm_s / 100.0

        if speed_m_s < 0.2:
            speed_score = 0.0
        elif speed_m_s < 0.4:
            speed_score = ((speed_m_s - 0.2) / 0.2) * 100.0

    # ── Final weighted score ──
    if view == "side":
        final_score = (
            (clearance_score * 0.35)
            + (symmetry_score * 0.35)
            + (speed_score * 0.20)
            + (torso_score * 0.10)
        )
    else:
        # Front view weights: clearance, sway, and symmetry (step width)
        final_score = (
            (clearance_score * 0.40)
            + (sway_score * 0.40)
            + (symmetry_score * 0.20)
        )

    return {
        "walk_grade": float(final_score),
        "mean_clearance_cm": mean_clearance_cm,
        "stride_length_m": mean_stride_cm / 100.0,
        "speed_m_s": speed_m_s,
        "torso_oscillation_cm": torso_osc_cm,
        "lateral_sway_cm": lateral_sway_cm,
        "clearance_score": float(clearance_score),
        "symmetry_score": float(symmetry_score),
        "speed_score": float(speed_score),
        "torso_score": float(torso_score),
        "sway_score": float(sway_score),
        "cm_per_pixel": float(cm_per_px),
        "fall_detected": fall_detected,
        "status": "ok",
    }
