# Typed container definitions for robot telemetry frames.
# LowStateData holds per-joint and IMU readings; HighStateData holds whole-body kinematics.
# Both are TypedDicts so they serialize cleanly to/from dicts for Parquet/HDF5 export.
from typing import TypedDict, List

class LowStateData(TypedDict):
    tick: int          # hardware monotonic counter (ms) from the robot controller
    q: List[float]     # joint positions (rad), one entry per motor
    dq: List[float]    # joint velocities (rad/s), one entry per motor
    tau: List[float]   # estimated joint torques (N·m), one entry per motor
    rpy: List[float]   # IMU roll/pitch/yaw (rad)
    accel: List[float] # IMU linear acceleration (m/s²), [x, y, z]

class HighStateData(TypedDict):
    tick: int                  # hardware monotonic counter (ms)
    base_velocity: List[float] # base body linear velocity (m/s), [x, y, z]
    odometry: List[float]      # base body position in world frame (m), [x, y, z]
    foot_contact: List[int]    # foot force / contact flags, one entry per foot
