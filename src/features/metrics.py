# Scalar performance metrics computed from telemetry frames.
# Examples: peak torque, RMS velocity, energy consumption per step cycle.
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def compute_control_precision(df: pd.DataFrame, cmd_col: str = 'q_cmd', state_col: str = 'q') -> Dict[str, Any]:
    """
    Computes the Root Mean Square Error (RMSE) between target joint commands and actual joint states.
    
    Args:
        df (pd.DataFrame): Synced telemetry dataframe.
        cmd_col (str): Column name containing commanded joint positions (list/array of floats per row).
        state_col (str): Column name containing actual joint positions (list/array of floats per row).
        
    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'joint_rmse': List of RMSE values for each joint.
            - 'mean_rmse': Average RMSE across all joints.
    """
    if df.empty or cmd_col not in df.columns or state_col not in df.columns:
        logger.warning(f"Required columns {cmd_col} or {state_col} missing or dataframe is empty.")
        return {'joint_rmse': [], 'mean_rmse': 0.0}
    
    try:
        # Stack the arrays into N x M matrices where N = time steps, M = joints
        cmd_arr = np.stack(df[cmd_col].dropna().values)
        state_arr = np.stack(df[state_col].dropna().values)
        
        # Handle length mismatch if any dropna happened asymmetrically
        min_len = min(len(cmd_arr), len(state_arr))
        if min_len == 0:
            return {'joint_rmse': [], 'mean_rmse': 0.0}
        
        cmd_arr = cmd_arr[:min_len]
        state_arr = state_arr[:min_len]
        
        # Compute difference
        diff = state_arr - cmd_arr
        
        # Compute RMSE per joint (axis 0 is time)
        joint_rmse = np.sqrt(np.nanmean(diff ** 2, axis=0))
        mean_rmse = float(np.nanmean(joint_rmse))
        
        return {
            'joint_rmse': joint_rmse.tolist(),
            'mean_rmse': mean_rmse
        }
    except Exception as e:
        logger.error(f"Error computing control precision: {e}")
        return {'joint_rmse': [], 'mean_rmse': 0.0}

def compute_cost_of_transport(
    df: pd.DataFrame,
    mass: float = 35.0,
    g: float = 9.81,
    power_col: Optional[str] = None,
    voltage_col: str = 'voltage',
    current_col: str = 'current',
    velocity_col: str = 'base_velocity',
    time_col: str = 'tick'
) -> float:
    """
    Computes the Cost of Transport (CoT): CoT = E / (m * g * d)
    where E is energy consumed (Joules) and d is total distance traveled (meters).
    
    Args:
        df (pd.DataFrame): Synced telemetry dataframe.
        mass (float): Mass of the robot in kg (default Unitree G1-Edu: 35.0 kg).
        g (float): Gravitational acceleration (default: 9.81 m/s^2).
        power_col (str, optional): Explicit electrical power column.
        voltage_col (str): Column name for voltage.
        current_col (str): Column name for current.
        velocity_col (str): Column name for base velocity (list of 3 floats [vx, vy, vz]).
        time_col (str): Column name for time ticks (DDS tick).
        
    Returns:
        float: Computed Cost of Transport.
    """
    if df.empty or len(df) < 2:
        logger.warning("Dataframe too short to compute CoT.")
        return 0.0
    
    try:
        # Calculate time steps (dt) in seconds
        ticks = df[time_col].to_numpy()
        dt = np.diff(ticks)
        
        # Determine if ticks are in milliseconds (typical for DDS tick) or seconds
        # If mean dt is > 0.5, we assume milliseconds and divide by 1000.
        mean_dt = np.mean(dt)
        if mean_dt > 0.5:
            dt = dt / 1000.0
        else:
            # Prevent negative or extremely small dt
            dt = np.maximum(dt, 1e-6)
            
        # 1. Calculate Power (Watts)
        if power_col in df.columns:
            power = df[power_col].to_numpy()
        elif voltage_col in df.columns and current_col in df.columns:
            power = (df[voltage_col] * df[current_col]).to_numpy()
        elif 'tau' in df.columns and 'dq' in df.columns:
            # Fallback to mechanical power: P = sum(|tau_i * dq_i|)
            tau_arr = np.stack(df['tau'].values)
            dq_arr = np.stack(df['dq'].values)
            power = np.sum(np.abs(tau_arr * dq_arr), axis=1)
        else:
            logger.warning("No power or joint torque/velocity telemetry found. CoT calculation cannot proceed.")
            return 0.0
            
        # Integrate power over time to get Energy (Joules)
        # Using trapezoidal or simple rectangular integration
        energy = np.sum(power[:-1] * dt)
        
        # 2. Calculate Distance Traveled (meters)
        total_distance = 0.0
        if 'odometry' in df.columns:
            odom_arr = np.stack(df['odometry'].values)
            # Distance is sum of incremental Euclidean norms
            diffs = np.diff(odom_arr, axis=0)
            # Handle 2D or 3D odometry
            total_distance = float(np.sum(np.linalg.norm(diffs, axis=1)))
        elif velocity_col in df.columns:
            vel_arr = np.stack(df[velocity_col].values)
            # Velocity can be 1D speed or 3D vector [vx, vy, vz]
            if len(vel_arr.shape) > 1 and vel_arr.shape[1] >= 2:
                speeds = np.linalg.norm(vel_arr, axis=1)
            else:
                speeds = np.abs(vel_arr.flatten())
            total_distance = float(np.sum(speeds[:-1] * dt))
        else:
            logger.warning("No odometry or base velocity found. Assuming stationary or fallback distance of 1.0m.")
            total_distance = 1.0
            
        if total_distance < 1e-3:
            logger.warning(f"Total distance {total_distance:.4f}m is too small to compute a reliable CoT.")
            return 0.0
            
        cot = energy / (mass * g * total_distance)
        return float(cot)
        
    except Exception as e:
        logger.error(f"Error computing Cost of Transport: {e}")
        return 0.0

def compute_control_latency(
    df: pd.DataFrame,
    cmd_col: str = 'q_cmd',
    state_col: str = 'q',
    time_col: str = 'tick',
    max_lag_steps: int = 20
) -> Dict[str, Any]:
    """
    Computes control latency (time delay) of actual joint states behind target commands
    using cross-correlation lag analysis.
    
    Args:
        df (pd.DataFrame): Synced telemetry dataframe.
        cmd_col (str): Column name containing commanded joint positions (list of floats).
        state_col (str): Column name containing actual joint positions (list of floats).
        time_col (str): Column name containing time ticks (DDS tick).
        max_lag_steps (int): Maximum number of discrete time lags to test.
        
    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'joint_latency_seconds': List of estimated latencies per joint in seconds.
            - 'joint_lag_steps': List of step lags per joint.
            - 'mean_latency_seconds': Mean control latency across all joints in seconds.
    """
    if df.empty or len(df) < max_lag_steps + 5 or cmd_col not in df.columns or state_col not in df.columns:
        logger.warning("Dataframe too short or missing columns for control latency calculation.")
        return {'joint_latency_seconds': [], 'joint_lag_steps': [], 'mean_latency_seconds': 0.0}
        
    try:
        ticks = df[time_col].to_numpy()
        mean_dt = np.mean(np.diff(ticks))
        dt_sec = mean_dt / 1000.0 if mean_dt > 0.5 else mean_dt
        
        cmd_arr = np.stack(df[cmd_col].dropna().values)
        state_arr = np.stack(df[state_col].dropna().values)
        
        min_len = min(len(cmd_arr), len(state_arr))
        if min_len < max_lag_steps + 5:
            return {'joint_latency_seconds': [], 'joint_lag_steps': [], 'mean_latency_seconds': 0.0}
            
        cmd_arr = cmd_arr[:min_len]
        state_arr = state_arr[:min_len]
        num_joints = cmd_arr.shape[1]
        
        joint_lag_steps = []
        joint_latency_seconds = []
        
        for j in range(num_joints):
            cmd_j = cmd_arr[:, j]
            state_j = state_arr[:, j]
            
            # Center the signals to be robust against static biases
            cmd_j_centered = cmd_j - np.mean(cmd_j)
            state_j_centered = state_j - np.mean(state_j)
            
            # Compute Mean Squared Error (MSE) for various positive lags
            # state lags command, so state[t + lag] corresponds to command[t]
            best_lag = 0
            min_mse = float('inf')
            
            for lag in range(max_lag_steps + 1):
                if lag == 0:
                    mse = np.mean((cmd_j_centered - state_j_centered) ** 2)
                else:
                    # Compare command[0:N-lag] with state[lag:N]
                    mse = np.mean((cmd_j_centered[:-lag] - state_j_centered[lag:]) ** 2)
                
                if mse < min_mse:
                    min_mse = mse
                    best_lag = lag
                    
            joint_lag_steps.append(best_lag)
            joint_latency_seconds.append(best_lag * dt_sec)
            
        mean_latency = float(np.mean(joint_latency_seconds))
        
        return {
            'joint_latency_seconds': joint_latency_seconds,
            'joint_lag_steps': joint_lag_steps,
            'mean_latency_seconds': mean_latency
        }
        
    except Exception as e:
        logger.error(f"Error computing control latency: {e}")
        return {'joint_latency_seconds': [], 'joint_lag_steps': [], 'mean_latency_seconds': 0.0}

def compute_hardware_stress(
    df: pd.DataFrame,
    tau_col: str = 'tau',
    time_col: str = 'tick',
    limit_threshold: float = 40.0
) -> Dict[str, Any]:
    """
    Analyzes joint torques to detect extreme spikes, rates of torque change (jerk),
    and safety limit violations.
    
    Args:
        df (pd.DataFrame): Synced telemetry dataframe.
        tau_col (str): Column name containing motor torques (list of floats).
        time_col (str): Column name containing time ticks (DDS tick).
        limit_threshold (float): Torque limit threshold in Nm to identify stress/breaches.
        
    Returns:
        Dict[str, Any]: A dictionary containing stress metrics:
            - 'max_torque': Max absolute torque observed per joint.
            - 'rms_torque': Root Mean Square torque per joint.
            - 'torque_jerk_max': Max rate of change of torque per joint (Nm/s).
            - 'breach_count': Number of limit violations per joint.
            - 'overall_max_torque': Maximum absolute torque across all joints.
            - 'overall_breach_count': Total limit violations across all joints.
    """
    if df.empty or tau_col not in df.columns:
        logger.warning(f"Torque column {tau_col} missing or dataframe is empty.")
        return {}
        
    try:
        tau_arr = np.stack(df[tau_col].dropna().values)
        if len(tau_arr) == 0:
            return {}
            
        ticks = df[time_col].to_numpy()
        dt = np.diff(ticks)
        mean_dt = np.mean(dt)
        dt_sec = mean_dt / 1000.0 if mean_dt > 0.5 else mean_dt
        # Handle cases where dt is zero or negative
        dt_sec = max(dt_sec, 1e-4)
        
        # 1. Absolute Maximum Torque
        abs_tau = np.abs(tau_arr)
        max_torque = np.max(abs_tau, axis=0)
        overall_max = float(np.max(abs_tau))
        
        # 2. RMS Torque
        rms_torque = np.sqrt(np.mean(tau_arr ** 2, axis=0))
        
        # 3. Breach Counts
        breaches = abs_tau > limit_threshold
        breach_count = np.sum(breaches, axis=0)
        overall_breaches = int(np.sum(breaches))
        
        # 4. Torque Jerk (Rate of change: d_tau / dt)
        if len(tau_arr) > 1:
            d_tau = np.diff(tau_arr, axis=0)
            torque_jerk = np.abs(d_tau / dt_sec)
            torque_jerk_max = np.max(torque_jerk, axis=0)
        else:
            torque_jerk_max = np.zeros(tau_arr.shape[1])
            
        return {
            'max_torque': max_torque.tolist(),
            'rms_torque': rms_torque.tolist(),
            'torque_jerk_max': torque_jerk_max.tolist(),
            'breach_count': breach_count.tolist(),
            'overall_max_torque': overall_max,
            'overall_breach_count': overall_breaches
        }
        
    except Exception as e:
        logger.error(f"Error computing hardware stress: {e}")
        return {}
