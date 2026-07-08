import math
from typing import Dict, Any, Tuple
from .baselines import CLASS_1_THRESHOLDS, CLASS_3_THRESHOLDS

class RuleBasedClassifier:
    def __init__(self):
        # We will score primarily against Class 1 thresholds to see how close to "Superhuman" the policy is.
        self.ideal_thresholds = CLASS_1_THRESHOLDS
        # We use Class 3 "acceptable" thresholds as our worst-case boundaries (score = 0)
        self.worst_thresholds = CLASS_3_THRESHOLDS

    def _score_metric(self, metric_name: str, actual_value: float) -> float:
        """
        Calculate a normalized score (0.0 to 1.0) for a specific metric.
        1.0 means it met or exceeded the ideal Superhuman limit.
        0.0 means it performed worse than the Experimental acceptable limit.
        """
        if metric_name not in self.ideal_thresholds or metric_name not in self.worst_thresholds:
            return 0.0
            
        ideal = self.ideal_thresholds[metric_name]["ideal"]
        worst = self.worst_thresholds[metric_name]["acceptable"]
        
        # If lower is better (which is true for RMSE, CoT, Latency, Stress, Variance)
        if actual_value <= ideal:
            return 1.0
        elif actual_value >= worst:
            return 0.0
        else:
            # Linear interpolation between worst (0.0) and ideal (1.0)
            return 1.0 - ((actual_value - ideal) / (worst - ideal))

    def classify(self, metrics: Dict[str, float]) -> Tuple[float, str]:
        """
        Classify a set of telemetry metrics.
        Returns a tuple: (overall_score, tier_label)
        """
        total_score = 0.0
        total_weight = 0.0
        
        for metric_name, bound in self.ideal_thresholds.items():
            if metric_name in metrics:
                actual_val = metrics[metric_name]
                weight = bound["weight"]
                
                score = self._score_metric(metric_name, actual_val)
                
                total_score += score * weight
                total_weight += weight
                
        # Normalize in case some metrics were missing
        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            final_score = 0.0
            
        # Determine Tier
        if final_score >= 0.85:
            tier = "Superhuman/Industrial"
        elif final_score >= 0.60:
            tier = "Research"
        else:
            tier = "Experimental"
            
        return final_score, tier
