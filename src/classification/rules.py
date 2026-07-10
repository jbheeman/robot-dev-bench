from typing import Dict, Tuple
from .baselines import get_tier_thresholds

class RuleBasedClassifier:
    def _score_metric(self, metric_name: str, actual_value: float, ideal_thresholds: Dict, worst_thresholds: Dict) -> float:
        """
        Calculate a normalized score (0.0 to 1.0) for a specific metric.
        Handles both "lower is better" and "higher is better" dynamically.
        """
        if metric_name not in ideal_thresholds or metric_name not in worst_thresholds:
            return 0.0
            
        ideal = ideal_thresholds[metric_name]["ideal"]
        worst = worst_thresholds[metric_name]["acceptable"]
        
        # Check if lower is better (e.g. symmetry: ideal=2.0, worst=50.0)
        if ideal < worst:
            if actual_value <= ideal:
                return 1.0
            elif actual_value >= worst:
                return 0.0
            else:
                return 1.0 - ((actual_value - ideal) / (worst - ideal))
        # Check if higher is better (e.g. periodicity: ideal=0.9, worst=0.1)
        else:
            if actual_value >= ideal:
                return 1.0
            elif actual_value <= worst:
                return 0.0
            else:
                return (actual_value - worst) / (ideal - worst)

    def score_breakdown(self, metrics: Dict[str, float], task: str = "general") -> Dict[str, Dict[str, float]]:
        """
        Per-metric scoring detail for the given task: raw value, normalized
        score (0-1), and task weight for every metric that participates in the
        weighted sum. Metrics that are missing, None, or zero-weighted are
        excluded — exactly the set classify() aggregates over. Used by the
        multi-model calibration report.
        """
        ideal_thresholds = get_tier_thresholds("Superhuman/Industrial", task)
        worst_thresholds = get_tier_thresholds("Experimental", task)

        breakdown = {}
        for metric_name, bound in ideal_thresholds.items():
            if metric_name not in metrics:
                continue
            actual_val = metrics[metric_name]
            if actual_val is None:  # Handle unavailable metrics
                continue

            weight = bound["weight"]
            if weight <= 0:
                continue

            score = self._score_metric(metric_name, actual_val, ideal_thresholds, worst_thresholds)
            breakdown[metric_name] = {
                "value": actual_val,
                "normalized_score": score,
                "weight": weight,
            }
        return breakdown

    def classify(self, metrics: Dict[str, float], task: str = "general") -> Tuple[float, str]:
        """
        Classify a set of telemetry metrics.
        Returns a tuple: (overall_score, tier_label)
        """
        breakdown = self.score_breakdown(metrics, task)

        total_score = sum(b["normalized_score"] * b["weight"] for b in breakdown.values())
        total_weight = sum(b["weight"] for b in breakdown.values())

        # Normalize in case some metrics were missing or weight was 0
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
