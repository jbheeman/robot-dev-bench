"""
test_log_integration.py

Tests for the two V2 features:
  1. Log Integration — POST /api/logs feeds physical robot logs into the
     database and validates format/pipeline compatibility up front.
  2. Multi-Model Evaluation — POST /api/evaluate runs parallel tests over
     baseline model logs from the same robot and returns calibration stats.
"""

import io
import os
import sys

import numpy as np
import pytest
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from scripts.generate_test_parquet import generate

SCORING_METRICS = ["control_precision", "dynamic_stability", "cost_of_transport", "system_latency"]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """TestClient with an isolated on-disk database injected for this module."""
    import src.storage.database as dbmod
    previous = dbmod._db_instance
    dbmod._db_instance = dbmod.BenchmarkDB(str(tmp_path_factory.mktemp("bench_data")))
    from src.web.app import app
    with TestClient(app) as c:
        yield c
    dbmod._db_instance = previous


def to_parquet_bytes(df) -> bytes:
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(df), buf)
    return buf.getvalue()


def make_variant(noise: float = 0.0, extra_lag: int = 0, seed: int = 7):
    """Synthetic log with degraded tracking noise and/or extra actuation lag."""
    df = generate()
    q = np.array(df["q"].tolist())
    if extra_lag > 0:
        q = np.roll(q, extra_lag, axis=0)
        q[:extra_lag] = q[extra_lag]
    if noise > 0:
        rng = np.random.default_rng(seed)
        q = q + rng.normal(0, noise, q.shape)
    df = df.copy()
    df["q"] = q.tolist()
    return df


def register(client, df, model_name, task="walking"):
    response = client.post(
        "/api/logs",
        files={"file": ("log.parquet", to_parquet_bytes(df), "application/octet-stream")},
        data={"model_name": model_name, "task": task},
    )
    assert response.status_code == 200, response.text
    return response.json()["log"]


class TestLogIntegration:
    def test_register_valid_log(self, client):
        log = register(client, generate(), "baseline-a")
        assert log["id"] > 0
        assert log["model_name"] == "baseline-a"
        assert log["schema_format"] == "native"
        assert log["validation_status"] == "valid"
        report = log["validation_report"]
        assert report["pipeline_metrics"] is not None
        assert all(report["metric_availability"][m] for m in SCORING_METRICS)
        assert all(c["status"] == "pass" for c in report["checks"])

    def test_register_kinematics_only_log_warns(self, client):
        """A joint-position-only log (like public HF datasets) is flagged, not rejected."""
        df = generate()[["tick", "q", "q_cmd"]]
        log = register(client, df, "hf-public-model")
        assert log["validation_status"] == "warnings"
        availability = log["validation_report"]["metric_availability"]
        assert availability["control_precision"] is True
        assert availability["system_latency"] is True
        assert availability["dynamic_stability"] is False
        assert availability["cost_of_transport"] is False
        # The dry-run metrics must report unavailable scoring metrics as None,
        # never a false-perfect 0.0.
        metrics = log["validation_report"]["pipeline_metrics"]
        assert metrics["dynamic_stability"] is None
        assert metrics["cost_of_transport"] is None

    def test_register_unknown_schema_is_invalid(self, client):
        import pandas as pd
        df = pd.DataFrame({"foo": [1, 2, 3], "bar": [4.0, 5.0, 6.0]})
        log = register(client, df, "mystery-model")
        assert log["validation_status"] == "invalid"
        failed = {c["name"] for c in log["validation_report"]["checks"] if c["status"] == "fail"}
        assert "schema_detection" in failed
        assert "required_columns" in failed

    def test_register_unreadable_file_rejected(self, client):
        response = client.post(
            "/api/logs",
            files={"file": ("junk.parquet", b"not parquet at all", "application/octet-stream")},
            data={"model_name": "junk-model"},
        )
        assert response.status_code == 400

    def test_list_get_delete_log(self, client):
        log = register(client, generate(), "lifecycle-model")
        log_id = log["id"]

        listed = client.get("/api/logs").json()["logs"]
        assert any(l["id"] == log_id for l in listed)

        fetched = client.get(f"/api/logs/{log_id}").json()["log"]
        assert fetched["model_name"] == "lifecycle-model"
        assert "checks" in fetched["validation_report"]

        assert client.delete(f"/api/logs/{log_id}").status_code == 200
        assert client.get(f"/api/logs/{log_id}").status_code == 404


@pytest.fixture(scope="module")
def registered_ids(client):
    """Three baseline models recorded on the same robot, differing quality."""
    ids = {}
    ids["clean"] = register(client, make_variant(), "model-clean")["id"]
    ids["noisy"] = register(client, make_variant(noise=0.08), "model-noisy")["id"]
    ids["laggy"] = register(client, make_variant(extra_lag=8), "model-laggy")["id"]
    return ids


class TestMultiModelEvaluation:
    def test_parallel_evaluation_returns_all_models(self, client, registered_ids):
        response = client.post(
            "/api/evaluate",
            json={"log_ids": list(registered_ids.values()), "task": "walking"},
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["status"] == "success"
        assert len(data["results"]) == 3
        assert all(r["status"] == "ok" for r in data["results"])
        for r in data["results"]:
            assert 0.0 <= r["score"] <= 1.0
            assert set(SCORING_METRICS).issubset(r["metrics"].keys())

    def test_calibration_summary_structure(self, client, registered_ids):
        data = client.post(
            "/api/evaluate",
            json={"log_ids": list(registered_ids.values()), "task": "walking"},
        ).json()
        cal = data["calibration"]
        assert cal["num_models"] == 3
        assert cal["num_failed"] == 0
        assert len(cal["ranking"]) == 3
        # Ranking is best-first
        scores = [r["score"] for r in cal["ranking"]]
        assert scores == sorted(scores, reverse=True)
        assert cal["score_spread"]["min"] <= cal["score_spread"]["max"]
        for metric in SCORING_METRICS:
            entry = cal["metrics"][metric]
            assert len(entry["values"]) == 3
            assert entry["lower_is_better"] is True
            assert "class1_ideal" in entry["reference_bounds"]

    def test_degraded_models_score_worse(self, client, registered_ids):
        data = client.post(
            "/api/evaluate",
            json={"log_ids": list(registered_ids.values()), "task": "walking"},
        ).json()
        by_model = {r["model_name"]: r for r in data["results"]}
        # Extra noise must worsen (raise) RMSE; extra lag must worsen latency.
        assert by_model["model-noisy"]["metrics"]["control_precision"] > \
               by_model["model-clean"]["metrics"]["control_precision"]
        assert by_model["model-laggy"]["metrics"]["system_latency"] > \
               by_model["model-clean"]["metrics"]["system_latency"]
        cal = data["calibration"]
        assert cal["metrics"]["control_precision"]["best_model"] != "model-noisy"
        assert cal["metrics"]["system_latency"]["best_model"] != "model-laggy"

    def test_evaluations_persisted_as_batch(self, client, registered_ids):
        data = client.post(
            "/api/evaluate",
            json={"log_ids": list(registered_ids.values()), "task": "walking"},
        ).json()
        batch_id = data["batch_id"]
        history = client.get("/api/evaluations", params={"task": "walking"}).json()["evaluations"]
        batch_rows = [e for e in history if e["batch_id"] == batch_id]
        assert len(batch_rows) == 3
        assert {e["model_name"] for e in batch_rows} == {"model-clean", "model-noisy", "model-laggy"}

    def test_evaluate_unknown_log_returns_404(self, client):
        response = client.post("/api/evaluate", json={"log_ids": [999999], "task": "walking"})
        assert response.status_code == 404

    def test_evaluate_empty_selection_returns_400(self, client):
        response = client.post("/api/evaluate", json={"log_ids": [], "task": "walking"})
        assert response.status_code == 400
