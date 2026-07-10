"""
extractor.py
Shared metric-extraction entry point used by the upload endpoint, the log
validator, and the multi-model evaluator.

Only the four hardware-grounded metrics (control_precision, dynamic_stability,
cost_of_transport, system_latency) feed the classifier's weighted sum, per the
thresholds/weights defined in baselines.py — the biomechanical metrics are
informational/display-only.

Scoring metrics whose source telemetry columns are absent are returned as None
rather than 0.0: all four are lower-is-better, so a 0.0 placeholder would read
as a perfect score. The classifier skips None values and renormalises the
remaining weights, keeping scores comparable across models whose logs carry
different telemetry.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional

from src.features.metrics import (
    compute_control_precision,
    compute_cost_of_transport,
    compute_control_latency,
)
from src.features.stability import compute_imu_variance
from src.features.biomechanics import (
    compute_smoothness,
    compute_spectral_arc_length,
    compute_symmetry,
    compute_periodicity,
    compute_range_of_motion,
    compute_jumping_metrics,
    compute_transition_metrics,
)


def metric_availability(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Reports which of the four scoring metrics can actually be computed from
    the columns present in a (normalised) telemetry DataFrame.
    """
    cols = set(df.columns)
    has_power = ("voltage" in cols and "current" in cols) or ("tau" in cols and "dq" in cols)
    has_distance = "odometry" in cols or "base_velocity" in cols
    return {
        "control_precision": "q" in cols and "q_cmd" in cols,
        "dynamic_stability": "rpy" in cols,
        "cost_of_transport": has_power and has_distance,
        "system_latency": "q" in cols and "q_cmd" in cols and "tick" in cols,
    }


def extract_metrics_from_dataframe(df: pd.DataFrame) -> dict:
    """
    Runs all feature extractors against a telemetry DataFrame.
    Returns a flat dict of scalar metrics: the four scoring metrics
    (None when their telemetry is unavailable) plus the display-only
    biomechanical metrics.
    """
    availability = metric_availability(df)

    # Hardware-grounded metrics (used in the real weighted-sum score)
    control_precision: Optional[float] = None
    dynamic_stability: Optional[float] = None
    cost_of_transport: Optional[float] = None
    system_latency: Optional[float] = None

    if availability["control_precision"]:
        precision = compute_control_precision(df)
        control_precision = round(precision.get("mean_rmse", 0.0), 5)

    if availability["dynamic_stability"]:
        imu_variance = compute_imu_variance(df)
        dynamic_stability = round(float(np.mean([
            imu_variance.get("roll_variance", 0.0),
            imu_variance.get("pitch_variance", 0.0),
        ])), 5)

    if availability["cost_of_transport"]:
        cost_of_transport = round(compute_cost_of_transport(df), 3)

    if availability["system_latency"]:
        latency = compute_control_latency(df)
        system_latency = round(latency.get("mean_latency_seconds", 0.0), 5)

    # Biomechanical metrics (display-only, joint-position-derived)
    smoothness = compute_smoothness(df)
    sparc = compute_spectral_arc_length(df)
    symmetry = compute_symmetry(df)
    periodicity = compute_periodicity(df)
    rom = compute_range_of_motion(df)
    jumping = compute_jumping_metrics(df)
    transitions = compute_transition_metrics(df)

    mean_ldlj = smoothness.get("mean_ldlj")
    mean_sparc = sparc.get("mean_sparc")
    mean_symmetry_index = symmetry.get("mean_symmetry_index")
    regularity_score = periodicity.get("regularity_score")
    mean_rom = rom.get("mean_rom")

    if mean_symmetry_index is not None:
        mean_symmetry_index = round(mean_symmetry_index, 3)

    return {
        "control_precision": control_precision,
        "dynamic_stability": dynamic_stability,
        "cost_of_transport": cost_of_transport,
        "system_latency": system_latency,
        "smoothness_ldlj": round(mean_ldlj, 3) if mean_ldlj is not None else 0.0,
        "smoothness_sparc": round(mean_sparc, 3) if mean_sparc is not None else 0.0,
        "symmetry": mean_symmetry_index,
        "periodicity": round(regularity_score, 3) if regularity_score is not None else 0.0,
        "rom_utilisation": round(mean_rom, 3) if mean_rom is not None else 0.0,
        "flight_time": round(jumping.get("flight_time", 0.0), 3),
        "peak_z_accel": round(jumping.get("peak_z_accel", 0.0), 3),
        "landing_jerk": round(jumping.get("landing_jerk", 0.0), 3),
        "com_oscillation": round(transitions.get("com_oscillation", 0.0), 3),
        "transition_time": round(transitions.get("transition_time", 0.0), 3),
    }
