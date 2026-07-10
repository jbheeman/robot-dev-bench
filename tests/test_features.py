import pytest
import numpy as np
import pandas as pd
from features.metrics import (
    compute_control_precision,
    compute_cost_of_transport,
    compute_control_latency,
    compute_hardware_stress
)
from features.stability import (
    compute_imu_variance,
    compute_com_stability
)

def test_control_precision():
    """Test Root Mean Square Error (RMSE) calculation for control precision."""
    # Create synthetic command and state data for 3 time steps, 2 joints
    # Diff sequence: joint 0: [0.1, 0.2, 0.3], joint 1: [0.0, 0.0, 0.0]
    q_cmd = [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]
    q_act = [[0.1, 1.0], [1.2, 2.0], [2.3, 3.0]]
    
    df = pd.DataFrame({
        'tick': [0, 10, 20],
        'q_cmd': q_cmd,
        'q': q_act
    })
    
    result = compute_control_precision(df, cmd_col='q_cmd', state_col='q')
    
    # Expected RMSE joint 0: sqrt((0.1^2 + 0.2^2 + 0.3^2) / 3) = sqrt((0.01 + 0.04 + 0.09) / 3) = sqrt(0.14 / 3) = 0.2160
    # Expected RMSE joint 1: 0.0
    expected_rmse_0 = np.sqrt(0.14 / 3)
    
    assert len(result['joint_rmse']) == 2
    assert pytest.approx(result['joint_rmse'][0], abs=1e-4) == expected_rmse_0
    assert pytest.approx(result['joint_rmse'][1], abs=1e-4) == 0.0
    assert pytest.approx(result['mean_rmse'], abs=1e-4) == (expected_rmse_0 + 0.0) / 2.0

def test_cost_of_transport_electrical():
    """Test Cost of Transport (CoT) with electrical power (voltage & current)."""
    # 5 steps at 100Hz (dt = 0.01s)
    # Total time = 0.04s
    ticks = [0, 10, 20, 30, 40]
    voltage = [24.0, 24.0, 24.0, 24.0, 24.0]
    current = [2.0, 2.0, 2.0, 2.0, 2.0]  # Power = 48W constant
    
    # Base velocity constant 1.0m/s -> distance = 1.0m/s * 0.04s = 0.04m
    base_velocity = [[1.0, 0.0, 0.0]] * 5
    
    df = pd.DataFrame({
        'tick': ticks,
        'voltage': voltage,
        'current': current,
        'base_velocity': base_velocity
    })
    
    # Mass = 35.0kg, g = 9.81
    # Energy = 48W * 0.04s = 1.92 Joules
    # Expected CoT = 1.92 / (35.0 * 9.81 * 0.04) = 1.92 / 13.734 = 0.139799
    expected_cot = 1.92 / (35.0 * 9.81 * 0.04)
    
    cot = compute_cost_of_transport(df, mass=35.0, g=9.81)
    assert pytest.approx(cot, abs=1e-4) == expected_cot

def test_cost_of_transport_mechanical():
    """Test Cost of Transport (CoT) fallback to mechanical joint power."""
    ticks = [0, 10, 20, 30, 40]  # dt = 0.01s, total = 0.04s
    tau = [[10.0, 5.0]] * 5
    dq = [[1.0, 2.0]] * 5  # Power = |10*1| + |5*2| = 20W constant
    
    # Base velocity constant 0.5m/s -> distance = 0.5m/s * 0.04s = 0.02m
    base_velocity = [[0.5, 0.0, 0.0]] * 5
    
    df = pd.DataFrame({
        'tick': ticks,
        'tau': tau,
        'dq': dq,
        'base_velocity': base_velocity
    })
    
    # Energy = 20W * 0.04s = 0.8 Joules
    # Expected CoT = 0.8 / (35.0 * 9.81 * 0.02) = 0.8 / 6.867 = 0.1165
    expected_cot = 0.8 / (35.0 * 9.81 * 0.02)
    
    cot = compute_cost_of_transport(df, mass=35.0, g=9.81)
    assert pytest.approx(cot, abs=1e-4) == expected_cot

def test_cost_of_transport_odometry():
    """Test CoT distance calculation from 3D odometry."""
    ticks = [0, 10, 20, 30, 40]  # dt = 0.01s, total = 0.04s
    voltage = [24.0] * 5
    current = [1.0] * 5  # Power = 24W
    
    # Robot moves along X-axis: 0.1m every 10ms -> total distance = 0.4m
    odometry = [
        [0.0, 0.0, 0.5],
        [0.1, 0.0, 0.5],
        [0.2, 0.0, 0.5],
        [0.3, 0.0, 0.5],
        [0.4, 0.0, 0.5]
    ]
    
    df = pd.DataFrame({
        'tick': ticks,
        'voltage': voltage,
        'current': current,
        'odometry': odometry
    })
    
    # Energy = 24W * 0.04s = 0.96 Joules
    # Distance = 0.4m
    # Expected CoT = 0.96 / (35.0 * 9.81 * 0.4) = 0.96 / 137.34 = 0.006989
    expected_cot = 0.96 / (35.0 * 9.81 * 0.4)
    
    cot = compute_cost_of_transport(df, mass=35.0, g=9.81)
    assert pytest.approx(cot, abs=1e-6) == expected_cot

def test_control_latency():
    """Test control latency calculation via cross-correlation lag detection."""
    # dt = 0.01s (100Hz)
    ticks = [i * 10 for i in range(50)]
    
    # 2 joints: Joint 0 has 3 steps lag, Joint 1 has 0 steps lag
    cmd_j0 = np.sin(np.linspace(0, 10, 50))
    cmd_j1 = np.cos(np.linspace(0, 10, 50))
    
    act_j0 = np.zeros(50)
    # Introduce 3 steps lag to joint 0 (state_j0[t+3] = cmd_j0[t])
    act_j0[3:] = cmd_j0[:-3]
    act_j0[:3] = cmd_j0[0]
    
    act_j1 = cmd_j1.copy()
    
    q_cmd = np.stack([cmd_j0, cmd_j1], axis=1).tolist()
    q_act = np.stack([act_j0, act_j1], axis=1).tolist()
    
    df = pd.DataFrame({
        'tick': ticks,
        'q_cmd': q_cmd,
        'q': q_act
    })
    
    result = compute_control_latency(df, cmd_col='q_cmd', state_col='q', max_lag_steps=10)
    
    assert result['joint_lag_steps'][0] == 3
    assert result['joint_lag_steps'][1] == 0
    assert pytest.approx(result['joint_latency_seconds'][0], abs=1e-4) == 0.03
    assert pytest.approx(result['joint_latency_seconds'][1], abs=1e-4) == 0.0
    assert pytest.approx(result['mean_latency_seconds'], abs=1e-4) == 0.015

def test_hardware_stress():
    """Test hardware stress metrics (limit breach and jerk)."""
    ticks = [0, 10, 20, 30]  # dt = 0.01s
    
    # 2 joints: joint 0 goes over threshold 40.0 once; joint 1 stays below.
    tau = [
        [10.0, 5.0],
        [45.0, 15.0],  # Spike at step 1 for joint 0
        [20.0, 10.0],
        [5.0, 2.0]
    ]
    
    df = pd.DataFrame({
        'tick': ticks,
        'tau': tau
    })
    
    result = compute_hardware_stress(df, limit_threshold=40.0)
    
    assert result['max_torque'][0] == 45.0
    assert result['max_torque'][1] == 15.0
    assert result['overall_max_torque'] == 45.0
    
    assert result['breach_count'][0] == 1
    assert result['breach_count'][1] == 0
    assert result['overall_breach_count'] == 1
    
    # Verify RMS calculation
    expected_rms_0 = np.sqrt((10.0**2 + 45.0**2 + 20.0**2 + 5.0**2) / 4)
    assert pytest.approx(result['rms_torque'][0], abs=1e-4) == expected_rms_0
    
    # Jerk validation:
    # d_tau joint 0: [35, -25, -15]
    # dt: 0.01s
    # torque_jerk joint 0: [3500, 2500, 1500] -> max = 3500.0 Nm/s
    assert pytest.approx(result['torque_jerk_max'][0], abs=1e-4) == 3500.0

def test_imu_variance():
    """Test roll and pitch variance metrics for postural stability."""
    # Roll, Pitch, Yaw
    rpy = [
        [0.1, -0.05, 1.2],
        [-0.1, 0.05, 1.3],
        [0.2, -0.1, 1.4],
        [-0.2, 0.1, 1.5]
    ]
    df = pd.DataFrame({
        'rpy': rpy
    })
    
    result = compute_imu_variance(df)
    
    # Expected variance roll: var([0.1, -0.1, 0.2, -0.2]) = 0.025
    # Expected variance pitch: var([-0.05, 0.05, -0.1, 0.1]) = 0.00625
    assert pytest.approx(result['roll_variance'], abs=1e-5) == 0.025
    assert pytest.approx(result['pitch_variance'], abs=1e-5) == 0.00625

def test_com_stability():
    """Test Center of Mass (CoM) stability metrics (height, accel, and jerk variance)."""
    ticks = [0, 10, 20]  # dt = 0.01s
    odometry = [
        [0.0, 0.0, 0.45],
        [0.1, 0.0, 0.50],
        [0.2, 0.0, 0.55]
    ]
    accel = [
        [0.1, 0.2, 9.8],
        [0.2, 0.1, 9.7],
        [0.3, 0.3, 9.9]
    ]
    
    df = pd.DataFrame({
        'tick': ticks,
        'odometry': odometry,
        'accel': accel
    })
    
    result = compute_com_stability(df)
    
    # Expected height variance (z-axis: [0.45, 0.50, 0.55]) -> var = 0.00166667 (unbiased) or 0.00083333?
    # numpy var uses ddof=0 by default -> var([0.45, 0.5, 0.55]) = ((-0.05)**2 + 0 + (0.05)**2)/3 = 0.005 / 3 = 0.00166667
    assert pytest.approx(result['height_variance'], abs=1e-6) == 0.005 / 3.0
    
    # Expected accel variance:
    # x: [0.1, 0.2, 0.3] -> mean = 0.2 -> var = (0.01 + 0 + 0.01)/3 = 0.02 / 3 = 0.00666667
    assert pytest.approx(result['accel_variance'][0], abs=1e-6) == 0.02 / 3.0
    
    # Jerk variance:
    # d_accel: [[0.1, -0.1, -0.1], [0.1, 0.2, 0.2]]
    # jerk (dt=0.01): [[10.0, -10.0, -10.0], [10.0, 20.0, 20.0]]
    # Mean jerk x: 10.0 -> var: var([10, 10]) = 0.0
    # Mean jerk y: 5.0 -> var: var([-10, 20]) = 225.0
    assert pytest.approx(result['jerk_variance'][0], abs=1e-6) == 0.0
    assert pytest.approx(result['jerk_variance'][1], abs=1e-6) == 225.0
