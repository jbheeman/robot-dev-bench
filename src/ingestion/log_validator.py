"""
log_validator.py
Pre-flight validation for physical-robot telemetry logs.

Solves the V1 "log incompatibility" problem: a log's format and pipeline
compatibility can be verified the moment it is uploaded, instead of only being
discovered after a model is run on the physical robot. The validator exercises
the same stages the real evaluation uses — schema detection, normalisation,
data-quality checks, and an actual end-to-end run of the metric extractors —
and returns a structured report of pass/warn/fail checks.

Report shape:
    {
      "status": "valid" | "warnings" | "invalid",
      "schema_format": str,
      "row_count": int,
      "duration_sec": float | None,
      "joint_count": int | None,
      "metric_availability": {metric: bool},
      "checks": [{"name": str, "status": "pass"|"warn"|"fail", "detail": str}],
      "pipeline_metrics": dict | None,   # metrics from the end-to-end dry run
    }
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.ingestion.schema_mapper import SchemaMapper
from src.features.extractor import extract_metrics_from_dataframe, metric_availability

logger = logging.getLogger(__name__)

MIN_ROWS_FAIL = 10
MIN_ROWS_WARN = 100
NAN_RATE_WARN = 0.01
NAN_RATE_FAIL = 0.20
DT_JITTER_WARN = 0.5  # coefficient of variation of tick deltas

SCORING_METRICS = ["control_precision", "dynamic_stability", "cost_of_transport", "system_latency"]


def _duration_seconds(df: pd.DataFrame) -> Optional[float]:
    """Duration from ticks, applying the same ms-vs-s heuristic as metrics.py."""
    if "tick" not in df.columns or len(df) < 2:
        return None
    ticks = df["tick"].to_numpy(dtype=float)
    span = float(ticks[-1] - ticks[0])
    dt = np.diff(ticks)
    if len(dt) == 0:
        return None
    return span / 1000.0 if np.mean(dt) > 0.5 else span


def validate_log(df: pd.DataFrame, filename: str = "") -> Dict[str, Any]:
    """
    Validates a raw telemetry DataFrame end-to-end against the benchmarking
    pipeline. Never raises for data problems — everything is folded into the
    returned report.
    """
    checks = []

    def add(name: str, status: str, detail: str):
        checks.append({"name": name, "status": status, "detail": detail})

    report: Dict[str, Any] = {
        "status": "invalid",
        "schema_format": "unknown",
        "row_count": int(len(df)),
        "duration_sec": None,
        "joint_count": None,
        "metric_availability": {m: False for m in SCORING_METRICS},
        "checks": checks,
        "pipeline_metrics": None,
    }

    # 1. Multi-episode note + first-episode filter (mirrors /api/upload behaviour)
    if "episode_index" in df.columns:
        n_episodes = int(df["episode_index"].nunique())
        first_episode = df["episode_index"].iloc[0]
        df = df[df["episode_index"] == first_episode]
        add("episode_filter", "pass",
            f"Multi-episode log ({n_episodes} episodes); validated against episode {first_episode} "
            f"({len(df)} rows), matching evaluation behaviour.")

    report["row_count"] = int(len(df))

    # 2. Schema detection + normalisation
    schema_format = SchemaMapper.detect_format(df)
    report["schema_format"] = schema_format
    if schema_format == "unknown":
        add("schema_detection", "fail",
            f"Unrecognised schema. Columns: {sorted(df.columns.tolist())[:20]}. "
            "Expected native pipeline columns (tick/q/q_cmd) or a HuggingFace LeRobot layout.")
    else:
        add("schema_detection", "pass", f"Detected schema format: {schema_format}")

    try:
        norm_df = SchemaMapper.normalise(df)
        add("normalisation", "pass", "SchemaMapper.normalise completed.")
    except Exception as e:
        norm_df = df
        add("normalisation", "fail", f"SchemaMapper.normalise raised: {e}")

    # 3. Required columns after normalisation
    missing_required = [c for c in ("tick", "q", "q_cmd") if c not in norm_df.columns]
    if missing_required:
        add("required_columns", "fail",
            f"Missing required column(s) after normalisation: {missing_required}. "
            "The pipeline needs tick, q (actual joints), and q_cmd (commanded joints).")
    else:
        add("required_columns", "pass", "tick, q, and q_cmd are all present.")

    # 4. Row count
    n = len(norm_df)
    if n < MIN_ROWS_FAIL:
        add("row_count", "fail", f"Only {n} rows — too short to compute any metric.")
    elif n < MIN_ROWS_WARN:
        add("row_count", "warn", f"Only {n} rows — metrics will be statistically weak.")
    else:
        add("row_count", "pass", f"{n} rows.")

    # 5. Tick sanity
    if "tick" in norm_df.columns and n >= 2:
        ticks = norm_df["tick"].to_numpy(dtype=float)
        dt = np.diff(ticks)
        negative = int(np.sum(dt < 0))
        if negative > 0:
            add("tick_monotonic", "warn",
                f"{negative} negative tick delta(s) — possible packet reordering; "
                "timestamp alignment will interpolate around them.")
        else:
            add("tick_monotonic", "pass", "Ticks are non-decreasing.")
        positive = dt[dt > 0]
        if len(positive) > 1:
            cv = float(np.std(positive) / np.mean(positive))
            if cv > DT_JITTER_WARN:
                add("tick_regularity", "warn",
                    f"Irregular sampling: tick-delta CV={cv:.2f} (>{DT_JITTER_WARN}). "
                    "Latency/CoT integration accuracy may degrade.")
            else:
                add("tick_regularity", "pass", f"Sampling is regular (tick-delta CV={cv:.2f}).")
        report["duration_sec"] = _duration_seconds(norm_df)

    # 6. Joint array consistency + NaN rate
    if "q" in norm_df.columns and n > 0:
        try:
            q_mat = np.array(norm_df["q"].tolist(), dtype=float)
            if q_mat.ndim != 2:
                raise ValueError(f"q stacked to shape {q_mat.shape}, expected 2-D (time x joints)")
            report["joint_count"] = int(q_mat.shape[1])
            add("joint_consistency", "pass",
                f"q is a consistent {q_mat.shape[0]}x{q_mat.shape[1]} matrix.")

            nan_rate = float(np.mean(np.isnan(q_mat)))
            if nan_rate > NAN_RATE_FAIL:
                add("nan_rate", "fail", f"{nan_rate:.1%} of joint samples are NaN.")
            elif nan_rate > NAN_RATE_WARN:
                add("nan_rate", "warn", f"{nan_rate:.1%} of joint samples are NaN.")
            else:
                add("nan_rate", "pass", f"NaN rate {nan_rate:.2%}.")

            if "q_cmd" in norm_df.columns:
                q_cmd_mat = np.array(norm_df["q_cmd"].tolist(), dtype=float)
                if q_cmd_mat.ndim == 2 and q_cmd_mat.shape[1] == q_mat.shape[1]:
                    add("cmd_state_alignment", "pass",
                        f"q_cmd matches q dimensionality ({q_mat.shape[1]} joints).")
                else:
                    add("cmd_state_alignment", "warn",
                        f"q_cmd shape {getattr(q_cmd_mat, 'shape', '?')} does not match q "
                        f"({q_mat.shape[1]} joints); RMSE/latency may be wrong.")
        except Exception as e:
            add("joint_consistency", "fail", f"Could not stack q into a matrix: {e}")

    # 7. Scoring-metric telemetry coverage
    availability = metric_availability(norm_df)
    report["metric_availability"] = availability
    missing_metrics = [m for m, ok in availability.items() if not ok]
    if not missing_metrics:
        add("scoring_telemetry", "pass",
            "All four scoring metrics (control precision, dynamic stability, "
            "cost of transport, system latency) are computable.")
    else:
        add("scoring_telemetry", "warn",
            f"Missing telemetry for: {', '.join(missing_metrics)}. These metrics will be "
            "excluded from the weighted score (weights renormalise over the rest).")

    # 8. End-to-end pipeline dry run — the actual extractors, on the actual data
    if not missing_required:
        try:
            metrics = extract_metrics_from_dataframe(norm_df)
            report["pipeline_metrics"] = metrics
            add("pipeline_execution", "pass",
                "Full feature-extraction pipeline executed successfully end-to-end.")
        except Exception as e:
            logger.exception(f"Pipeline dry run failed for '{filename}'")
            add("pipeline_execution", "fail", f"Feature extraction raised: {e}")
    else:
        add("pipeline_execution", "fail",
            "Skipped — required columns missing, pipeline cannot run.")

    # Aggregate status: any fail -> invalid, any warn -> warnings, else valid
    statuses = {c["status"] for c in checks}
    if "fail" in statuses:
        report["status"] = "invalid"
    elif "warn" in statuses:
        report["status"] = "warnings"
    else:
        report["status"] = "valid"

    return report
