"""
baselines.py
Static configuration mapping for performance tiers (Class 1, Class 2, Class 3).
These thresholds act as the ground-truth for our rule-based scoring classification,
now using clinically-grounded biomechanical metrics.
"""

from typing import Dict, TypedDict, Any
import copy

class ThresholdBound(TypedDict):
    ideal: float      # The best possible value (max score)
    acceptable: float # The cutoff for an acceptable run in this tier
    weight: float     # How much this metric matters in the final score (0.0 to 1.0)

# Base Biomechanical Thresholds (General)
# ldlj: ideal 0 (perfect), acceptable -15 (higher is better)
# sparc: ideal 0, acceptable -20 (higher is better)
# symmetry: ideal 2.0 (%), acceptable 30.0 (%) (lower is better)
# periodicity: ideal 0.9, acceptable 0.2 (higher is better)
# rom_utilisation: ideal 1.5 rad, acceptable 0.2 rad (higher is better)

CLASS_1_THRESHOLDS = {
    "smoothness_ldlj": {"ideal": -1.0, "acceptable": -5.0, "weight": 0.2},
    "smoothness_sparc": {"ideal": -2.0, "acceptable": -10.0, "weight": 0.2},
    "symmetry": {"ideal": 2.0, "acceptable": 10.0, "weight": 0.2},
    "periodicity": {"ideal": 0.9, "acceptable": 0.6, "weight": 0.2},
    "rom_utilisation": {"ideal": 1.5, "acceptable": 0.5, "weight": 0.2}
}

CLASS_2_THRESHOLDS = {
    "smoothness_ldlj": {"ideal": -5.0, "acceptable": -10.0, "weight": 0.2},
    "smoothness_sparc": {"ideal": -10.0, "acceptable": -20.0, "weight": 0.2},
    "symmetry": {"ideal": 10.0, "acceptable": 25.0, "weight": 0.2},
    "periodicity": {"ideal": 0.6, "acceptable": 0.3, "weight": 0.2},
    "rom_utilisation": {"ideal": 0.5, "acceptable": 0.2, "weight": 0.2}
}

CLASS_3_THRESHOLDS = {
    "smoothness_ldlj": {"ideal": -10.0, "acceptable": -20.0, "weight": 0.2},
    "smoothness_sparc": {"ideal": -20.0, "acceptable": -40.0, "weight": 0.2},
    "symmetry": {"ideal": 25.0, "acceptable": 50.0, "weight": 0.2},
    "periodicity": {"ideal": 0.3, "acceptable": 0.1, "weight": 0.2},
    "rom_utilisation": {"ideal": 0.2, "acceptable": 0.05, "weight": 0.2}
}

# Task-specific weight overrides
TASK_WEIGHTS = {
    "walking": {
        "smoothness_ldlj": 0.15,
        "smoothness_sparc": 0.15,
        "symmetry": 0.3,
        "periodicity": 0.3,
        "rom_utilisation": 0.1
    },
    "reaching": {
        "smoothness_ldlj": 0.3,
        "smoothness_sparc": 0.3,
        "symmetry": 0.1,
        "periodicity": 0.0,  # Not periodic
        "rom_utilisation": 0.3
    },
    "manipulation": {
        "smoothness_ldlj": 0.25,
        "smoothness_sparc": 0.25,
        "symmetry": 0.1,
        "periodicity": 0.1,
        "rom_utilisation": 0.3
    }
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
