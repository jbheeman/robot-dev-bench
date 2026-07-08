"""
baselines.py
Static configuration mapping for performance tiers (Class 1, Class 2, Class 3).
These thresholds act as the ground-truth for our rule-based scoring classification.
"""

from typing import Dict, TypedDict, Any

class ThresholdBound(TypedDict):
    ideal: float      # The best possible value (max score)
    acceptable: float # The cutoff for an acceptable run in this tier
    weight: float     # How much this metric matters in the final score (0.0 to 1.0)

# Class 1: Superhuman/Industrial
# Derived from theoretical optimal control limits or Isaac Sim/Lab RL targets.
CLASS_1_THRESHOLDS = {
    "rmse": {"ideal": 0.0, "acceptable": 0.02, "weight": 0.3},
    "cot": {"ideal": 0.1, "acceptable": 0.3, "weight": 0.2},
    "latency_ms": {"ideal": 1.0, "acceptable": 5.0, "weight": 0.2},
    "stress": {"ideal": 0.0, "acceptable": 0.1, "weight": 0.15},
    "imu_variance": {"ideal": 0.0, "acceptable": 0.005, "weight": 0.15}
}

# Class 2: Research
# Derived from typical human-operated teleop datasets and standard academic RL models.
CLASS_2_THRESHOLDS = {
    "rmse": {"ideal": 0.02, "acceptable": 0.05, "weight": 0.3},
    "cot": {"ideal": 0.3, "acceptable": 0.6, "weight": 0.2},
    "latency_ms": {"ideal": 5.0, "acceptable": 20.0, "weight": 0.2},
    "stress": {"ideal": 0.1, "acceptable": 0.4, "weight": 0.15},
    "imu_variance": {"ideal": 0.005, "acceptable": 0.02, "weight": 0.15}
}

# Experimental (Class 3) - Novice Human / Poor RL
CLASS_3_THRESHOLDS = {
    "rmse": {"ideal": 0.05, "acceptable": 0.15, "weight": 0.3},
    "cot": {"ideal": 0.6, "acceptable": 1.2, "weight": 0.2},
    "latency_ms": {"ideal": 20.0, "acceptable": 50.0, "weight": 0.2},
    "stress": {"ideal": 0.4, "acceptable": 0.8, "weight": 0.15},
    "imu_variance": {"ideal": 0.02, "acceptable": 0.08, "weight": 0.15}
}

def get_tier_thresholds(tier_name: str) -> Dict[str, ThresholdBound]:
    """Retrieve the threshold dictionary for a specific tier name."""
    mapping = {
        "Superhuman/Industrial": CLASS_1_THRESHOLDS,
        "Research": CLASS_2_THRESHOLDS,
        "Experimental": CLASS_3_THRESHOLDS
    }
    return mapping.get(tier_name, CLASS_3_THRESHOLDS)
