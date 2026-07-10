"""
evaluator.py
Multi-model evaluation service.

Runs the full benchmarking pipeline over several registered logs in parallel —
the intended workflow is a set of baseline AI models each producing a log on
the same physical robot and task — and builds a calibration summary comparing
every model's raw metric values, normalized sub-scores, and final scores
against the current tier bounds in baselines.py. That summary is what lets us
check the scoring thresholds against real observed model behaviour.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.ingestion.schema_mapper import SchemaMapper
from src.features.extractor import extract_metrics_from_dataframe
from src.classification.rules import RuleBasedClassifier
from src.classification.baselines import get_tier_thresholds

logger = logging.getLogger(__name__)

SCORING_METRICS = ["control_precision", "dynamic_stability", "cost_of_transport", "system_latency"]

_classifier = RuleBasedClassifier()


def evaluate_log_record(record: Dict[str, Any], task: str) -> Dict[str, Any]:
    """
    Evaluates one registered log (as stored in the database) end-to-end.
    Returns a result dict; failures are captured as status='error' entries
    rather than raised, so one bad log doesn't sink a whole batch.
    """
    base = {
        "log_id": record["id"],
        "model_name": record["model_name"],
        "filename": record["filename"],
    }
    try:
        df = pd.read_parquet(record["stored_path"], engine="pyarrow")
        if "episode_index" in df.columns:
            df = df[df["episode_index"] == df["episode_index"].iloc[0]]
        df = SchemaMapper.normalise(df)

        metrics = extract_metrics_from_dataframe(df)
        score, tier = _classifier.classify(metrics, task=task)
        breakdown = _classifier.score_breakdown(metrics, task=task)

        return {
            **base,
            "status": "ok",
            "metrics": metrics,
            "score": round(score, 3),
            "tier": tier,
            "breakdown": breakdown,
        }
    except Exception as e:
        logger.exception(f"Evaluation failed for log {record['id']} ({record['filename']})")
        return {**base, "status": "error", "error": str(e)}


def run_parallel_evaluations(records: List[Dict[str, Any]], task: str) -> List[Dict[str, Any]]:
    """Evaluates all given log records concurrently, preserving input order."""
    if not records:
        return []
    with ThreadPoolExecutor(max_workers=min(8, len(records))) as executor:
        return list(executor.map(lambda r: evaluate_log_record(r, task), records))


def build_calibration_summary(results: List[Dict[str, Any]], task: str) -> Dict[str, Any]:
    """
    Cross-model calibration statistics from a batch of successful evaluations:
    per-metric value spread across models, each model's normalized sub-score,
    and the current tier bounds for reference, so misaligned thresholds are
    visible at a glance.
    """
    ok = [r for r in results if r.get("status") == "ok"]

    ranking = sorted(
        ({"model_name": r["model_name"], "log_id": r["log_id"], "score": r["score"], "tier": r["tier"]}
         for r in ok),
        key=lambda x: x["score"],
        reverse=True,
    )

    scores = [r["score"] for r in ok]
    score_spread = {
        "min": round(min(scores), 3),
        "max": round(max(scores), 3),
        "mean": round(float(np.mean(scores)), 3),
        "std": round(float(np.std(scores)), 3),
    } if scores else None

    ideal_bounds = get_tier_thresholds("Superhuman/Industrial", task)
    worst_bounds = get_tier_thresholds("Experimental", task)

    metrics_summary: Dict[str, Any] = {}
    for metric in SCORING_METRICS:
        values = {r["model_name"]: r["metrics"].get(metric) for r in ok}
        present = {m: v for m, v in values.items() if v is not None}

        entry: Dict[str, Any] = {
            "values": values,
            "normalized_scores": {
                r["model_name"]: round(r["breakdown"][metric]["normalized_score"], 3)
                for r in ok if metric in r.get("breakdown", {})
            },
            "weight": ideal_bounds[metric]["weight"],
            "lower_is_better": ideal_bounds[metric]["ideal"] < worst_bounds[metric]["acceptable"],
            "reference_bounds": {
                "class1_ideal": ideal_bounds[metric]["ideal"],
                "class3_acceptable": worst_bounds[metric]["acceptable"],
            },
        }
        if present:
            vals = list(present.values())
            lower_better = entry["lower_is_better"]
            best_model = min(present, key=present.get) if lower_better else max(present, key=present.get)
            entry.update({
                "mean": round(float(np.mean(vals)), 5),
                "std": round(float(np.std(vals)), 5),
                "min": round(float(np.min(vals)), 5),
                "max": round(float(np.max(vals)), 5),
                "best_model": best_model,
            })
        metrics_summary[metric] = entry

    return {
        "task": task,
        "num_models": len(ok),
        "num_failed": len(results) - len(ok),
        "score_spread": score_spread,
        "ranking": ranking,
        "metrics": metrics_summary,
    }
