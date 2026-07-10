"""
baselines.py
Static configuration mapping for performance tiers (Class 1, Class 2, Class 3).
These thresholds act as the ground-truth for our rule-based scoring classification,
using the four core hardware-grounded evaluation metrics:

  - control_precision:  RMSE (radians) between commanded and actual joint positions.
  - dynamic_stability:  Mean variance (rad^2) of IMU pitch/roll during the run.
  - cost_of_transport:  Dimensionless CoT = energy / (mass * g * distance).
  - system_latency:     Mean control loop delay (seconds) between command and actuation.

All four are "lower is better". Numeric bounds below are engineering placeholders
(industrial servo/RL-policy tracking is sub-centimeter/sub-degree; human-competitive
CoT is ~0.2-2; teleoperation-grade latency is single-digit milliseconds) and should be
recalibrated once we have a larger corpus of real G1 hardware runs.
"""

from typing import Dict, TypedDict, Any
import copy

class ThresholdBound(TypedDict):
    ideal: float      # The best possible value (max score)
    acceptable: float # The cutoff for an acceptable run in this tier
    weight: float     # How much this metric matters in the final score (0.0 to 1.0)

CLASS_1_THRESHOLDS = {
    "control_precision": {"ideal": 0.01, "acceptable": 0.03, "weight": 0.25},
    "dynamic_stability": {"ideal": 0.0005, "acceptable": 0.002, "weight": 0.25},
    "cost_of_transport": {"ideal": 0.3, "acceptable": 1.0, "weight": 0.25},
    "system_latency": {"ideal": 0.005, "acceptable": 0.02, "weight": 0.25},
}

CLASS_2_THRESHOLDS = {
    "control_precision": {"ideal": 0.03, "acceptable": 0.08, "weight": 0.25},
    "dynamic_stability": {"ideal": 0.005, "acceptable": 0.02, "weight": 0.25},
    "cost_of_transport": {"ideal": 1.0, "acceptable": 3.0, "weight": 0.25},
    "system_latency": {"ideal": 0.02, "acceptable": 0.05, "weight": 0.25},
}

CLASS_3_THRESHOLDS = {
    "control_precision": {"ideal": 0.08, "acceptable": 0.25, "weight": 0.25},
    "dynamic_stability": {"ideal": 0.02, "acceptable": 0.08, "weight": 0.25},
    "cost_of_transport": {"ideal": 3.0, "acceptable": 10.0, "weight": 0.25},
    "system_latency": {"ideal": 0.05, "acceptable": 0.15, "weight": 0.25},
}

# Task-specific weight overrides: how much each metric should count toward the
# final score for a given evaluation task. Tasks that don't cover meaningful
# ground distance (reaching/manipulation) zero out cost_of_transport since it's
# undefined/meaningless without locomotion.
TASK_WEIGHTS = {
    "walking": {
        "control_precision": 0.25,
        "dynamic_stability": 0.35,
        "cost_of_transport": 0.25,
        "system_latency": 0.15,
    },
    "reaching": {
        "control_precision": 0.45,
        "dynamic_stability": 0.10,
        "cost_of_transport": 0.0,
        "system_latency": 0.45,
    },
    "manipulation": {
        "control_precision": 0.45,
        "dynamic_stability": 0.10,
        "cost_of_transport": 0.0,
        "system_latency": 0.45,
    },
    "jumping": {
        "control_precision": 0.20,
        "dynamic_stability": 0.35,
        "cost_of_transport": 0.10,
        "system_latency": 0.35,
    },
    "transitions": {
        "control_precision": 0.25,
        "dynamic_stability": 0.40,
        "cost_of_transport": 0.10,
        "system_latency": 0.25,
    },
}

def get_tier_thresholds(tier_name: str, task: str = "general") -> Dict[str, ThresholdBound]:
    """Retrieve thresholds for a specific tier, applying task-specific weights."""
    mapping = {
        "Superhuman/Industrial": CLASS_1_THRESHOLDS,
        "Research": CLASS_2_THRESHOLDS,
        "Experimental": CLASS_3_THRESHOLDS
    }
    base = mapping.get(tier_name, CLASS_3_THRESHOLDS)

    result = copy.deepcopy(base)

    task_key = task.lower()
    if task_key in TASK_WEIGHTS:
        for metric, weight in TASK_WEIGHTS[task_key].items():
            if metric in result:
                result[metric]["weight"] = weight

    return result
