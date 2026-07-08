<<<<<<< HEAD
# Balance and stability metrics derived from IMU and foot-contact data.
# Examples: centre-of-mass sway, tilt envelope, gait phase detection.
=======
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def compute_imu_variance(df: pd.DataFrame, rpy_col: str = 'rpy') -> Dict[str, float]:
    """
    Computes variance of IMU roll, pitch, and yaw angles to assess base stabilization.
    
    Args:
        df (pd.DataFrame): Synced telemetry dataframe.
        rpy_col (str): Column name containing Euler angles [roll, pitch, yaw] in radians.
        
    Returns:
        Dict[str, float]: Variance for roll, pitch, and yaw.
    """
    if df.empty or rpy_col not in df.columns:
        logger.warning(f"RPY column '{rpy_col}' missing or dataframe is empty.")
        return {'roll_variance': 0.0, 'pitch_variance': 0.0, 'yaw_variance': 0.0}
        
    try:
        rpy_arr = np.stack(df[rpy_col].dropna().values)
        if len(rpy_arr) == 0:
            return {'roll_variance': 0.0, 'pitch_variance': 0.0, 'yaw_variance': 0.0}
            
        # Ensure it has at least roll and pitch (shape N x 2 or N x 3)
        if len(rpy_arr.shape) == 2 and rpy_arr.shape[1] >= 2:
            roll_var = float(np.nanvar(rpy_arr[:, 0]))
            pitch_var = float(np.nanvar(rpy_arr[:, 1]))
            yaw_var = float(np.nanvar(rpy_arr[:, 2])) if rpy_arr.shape[1] >= 3 else 0.0
        else:
            # Handle fallback if it's stored as scalar or flat array
            flat_vals = rpy_arr.flatten()
            roll_var = float(np.nanvar(flat_vals))
            pitch_var = 0.0
            yaw_var = 0.0
            
        return {
            'roll_variance': roll_var,
            'pitch_variance': pitch_var,
            'yaw_variance': yaw_var
        }
        
    except Exception as e:
        logger.error(f"Error computing IMU variance: {e}")
        return {'roll_variance': 0.0, 'pitch_variance': 0.0, 'yaw_variance': 0.0}

def compute_com_stability(
    df: pd.DataFrame,
    odom_col: str = 'odometry',
    accel_col: str = 'accel',
    time_col: str = 'tick'
) -> Dict[str, Any]:
    """
    Computes indicators of Center of Mass (CoM) stability:
    - Base height (z-axis) position variance (from odometry).
    - Base acceleration variance (IMU).
    - Base jerk variance (IMU rate of change of acceleration).
    
    Args:
        df (pd.DataFrame): Synced telemetry dataframe.
        odom_col (str): Column containing odometry [x, y, z] values.
        accel_col (str): Column containing IMU linear acceleration [ax, ay, az].
        time_col (str): Column name containing time ticks (DDS tick).
        
    Returns:
        Dict[str, Any]: Stability metrics.
    """
    metrics = {
        'height_variance': 0.0,
        'accel_variance': [0.0, 0.0, 0.0],
        'jerk_variance': [0.0, 0.0, 0.0]
    }
    
    if df.empty:
        return metrics
        
    # 1. Height Variance from Odometry
    if odom_col in df.columns:
        try:
            odom_arr = np.stack(df[odom_col].dropna().values)
            if len(odom_arr) > 0 and len(odom_arr.shape) == 2 and odom_arr.shape[1] >= 3:
                # Base height is index 2 (z)
                metrics['height_variance'] = float(np.nanvar(odom_arr[:, 2]))
        except Exception as e:
            logger.error(f"Error computing height variance: {e}")
            
    # 2. Acceleration and Jerk Variance from IMU
    if accel_col in df.columns:
        try:
            accel_arr = np.stack(df[accel_col].dropna().values)
            if len(accel_arr) > 0 and len(accel_arr.shape) == 2 and accel_arr.shape[1] >= 3:
                metrics['accel_variance'] = np.nanvar(accel_arr, axis=0).tolist()
                
                # Compute base jerk (rate of change of linear acceleration)
                if len(accel_arr) > 1 and time_col in df.columns:
                    ticks = df[time_col].to_numpy()
                    dt = np.diff(ticks)
                    mean_dt = np.mean(dt)
                    dt_sec = mean_dt / 1000.0 if mean_dt > 0.5 else mean_dt
                    dt_sec = max(dt_sec, 1e-4)
                    
                    d_accel = np.diff(accel_arr, axis=0)
                    jerk_arr = d_accel / dt_sec
                    metrics['jerk_variance'] = np.nanvar(jerk_arr, axis=0).tolist()
        except Exception as e:
            logger.error(f"Error computing acceleration/jerk variance: {e}")
            
    return metrics
>>>>>>> d8d255ef7cce25e829b0eef8d4032f0ebc4ac185
