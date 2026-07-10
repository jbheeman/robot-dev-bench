"""
generate_test_parquet.py

Generates a synthetic but realistic .parquet telemetry log file for
end-to-end testing of the robot benchmarking pipeline.

The data simulates ~5 seconds of a robot walking at 100Hz.
All columns match the schema expected by metrics.py and stability.py.

Usage:
    python scripts/generate_test_parquet.py
    # Writes to: tests/data/test_log.parquet
"""

import numpy as np
import pandas as pd
import os

# --- Configuration ---
NUM_STEPS = 500       # 5 seconds at 100Hz
DT_MS = 10            # 10ms per tick
NUM_JOINTS = 12       # Unitree G1-Edu has 12 lower-body joints

ROBOT_MASS_KG = 35.0
FORWARD_SPEED_MS = 0.5  # meters per second
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "tests", "data", "test_log.parquet")


def generate() -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)
    t = np.arange(NUM_STEPS)

    # --- Timestamps (ticks in milliseconds) ---
    ticks = t * DT_MS

    # --- Joint positions (q_cmd): sinusoidal gait-like commands ---
    # Each joint oscillates at a slightly different frequency
    phase_offsets = np.linspace(0, np.pi, NUM_JOINTS)
    q_cmd = np.stack(
        [0.3 * np.sin(2 * np.pi * 1.5 * t * DT_MS / 1000.0 + phi) for phi in phase_offsets],
        axis=1
    )

    # --- Actual joint positions (q): q_cmd with 3-step lag + small noise ---
    lag = 3
    q = np.zeros_like(q_cmd)
    q[lag:] = q_cmd[:-lag] + rng.normal(0, 0.005, (NUM_STEPS - lag, NUM_JOINTS))
    q[:lag] = q_cmd[0] + rng.normal(0, 0.005, (lag, NUM_JOINTS))

    # --- Joint velocities (dq) ---
    dq = np.diff(q_cmd, axis=0, prepend=q_cmd[[0]]) / (DT_MS / 1000.0)

    # --- Joint torques (tau): proportional to velocity, with one spike ---
    tau = 5.0 * dq + rng.normal(0, 0.5, q.shape)
    # Inject a single torque spike at step 50 on joint 0 to test stress detection
    tau[50, 0] = 45.0

    # --- IMU (rpy): small roll/pitch variation simulating walking sway ---
    roll = 0.05 * np.sin(2 * np.pi * 1.5 * t * DT_MS / 1000.0) + rng.normal(0, 0.003, NUM_STEPS)
    pitch = 0.03 * np.cos(2 * np.pi * 1.5 * t * DT_MS / 1000.0) + rng.normal(0, 0.002, NUM_STEPS)
    yaw = np.zeros(NUM_STEPS)
    rpy = np.stack([roll, pitch, yaw], axis=1)

    # --- Linear acceleration (IMU) ---
    accel = np.stack([
        rng.normal(0.05, 0.1, NUM_STEPS),   # ax
        rng.normal(0.01, 0.05, NUM_STEPS),  # ay
        rng.normal(9.81, 0.1, NUM_STEPS),   # az (gravity dominant)
    ], axis=1)

    # --- Odometry (position): robot walks forward along x-axis ---
    x = FORWARD_SPEED_MS * t * DT_MS / 1000.0
    y = np.zeros(NUM_STEPS)
    z = np.full(NUM_STEPS, 0.78)  # ~0.78m standing height
    odometry = np.stack([x, y, z], axis=1)

    # --- Base velocity: forward speed + noise ---
    vx = np.full(NUM_STEPS, FORWARD_SPEED_MS) + rng.normal(0, 0.02, NUM_STEPS)
    vy = rng.normal(0, 0.005, NUM_STEPS)
    vz = rng.normal(0, 0.001, NUM_STEPS)
    base_velocity = np.stack([vx, vy, vz], axis=1)

    # --- Electrical power ---
    # Simulate ~200W average draw during walking
    voltage = np.full(NUM_STEPS, 24.0) + rng.normal(0, 0.1, NUM_STEPS)
    current = np.full(NUM_STEPS, 8.5) + rng.normal(0, 0.2, NUM_STEPS)

    # --- Assemble DataFrame ---
    df = pd.DataFrame({
        "tick": ticks,
        "q_cmd": q_cmd.tolist(),
        "q": q.tolist(),
        "dq": dq.tolist(),
        "tau": tau.tolist(),
        "rpy": rpy.tolist(),
        "accel": accel.tolist(),
        "odometry": odometry.tolist(),
        "base_velocity": base_velocity.tolist(),
        "voltage": voltage,
        "current": current,
    })

    return df


if __name__ == "__main__":
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_PATH)), exist_ok=True)
    df = generate()
    df.to_parquet(OUTPUT_PATH, engine="pyarrow", index=False)
    print(f"Generated {len(df)} rows → {os.path.abspath(OUTPUT_PATH)}")
    print(f"Columns: {list(df.columns)}")
