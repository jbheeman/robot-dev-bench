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
            
        jerk_sq_integral = np.trapezoid(j[:, col]**2, t_sec)
        
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
