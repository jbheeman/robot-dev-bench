"""
test_integration.py

End-to-end integration tests for the FastAPI benchmarking dashboard.

Tests the full pipeline:
  1. Synthetic .parquet file is generated via generate_test_parquet.generate()
  2. The file is uploaded to the /api/upload endpoint via FastAPI's TestClient
  3. The response is validated for correct structure, metric keys, and tier labels
"""

import io
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
import sys
import os

# Ensure the project root is on the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from src.web.app import app
from scripts.generate_test_parquet import generate

VALID_TIERS = {"Superhuman/Industrial", "Research", "Experimental"}
EXPECTED_METRIC_KEYS = {
    "control_precision",
    "dynamic_stability",
    "cost_of_transport",
    "system_latency",
    "smoothness_ldlj",
    "smoothness_sparc",
    "symmetry",
    "periodicity",
    "rom_utilisation",
}


@pytest.fixture(scope="module")
def client():
    """Provides a synchronous FastAPI TestClient for the duration of the module."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def parquet_bytes() -> bytes:
    """
    Generates the synthetic telemetry DataFrame and serializes it to
    in-memory Parquet bytes so we can POST it without touching the disk.
    """
    df = generate()
    table = pa.Table.from_pandas(df)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    buf.seek(0)
    return buf.read()


class TestUploadEndpoint:
    """Integration tests for the POST /api/upload endpoint."""

    def test_upload_valid_parquet_returns_200(self, client, parquet_bytes):
        """A valid .parquet upload should return HTTP 200."""
        response = client.post(
            "/api/upload",
            files={"file": ("test_log.parquet", parquet_bytes, "application/octet-stream")},
        )
        assert response.status_code == 200, f"Expected 200 but got {response.status_code}: {response.text}"

    def test_upload_valid_parquet_status_success(self, client, parquet_bytes):
        """Response body should report status: success."""
        response = client.post(
            "/api/upload",
            files={"file": ("test_log.parquet", parquet_bytes, "application/octet-stream")},
        )
        data = response.json()
        assert data["status"] == "success"

    def test_upload_returns_all_metric_keys(self, client, parquet_bytes):
        """Response must include all five expected metric keys."""
        response = client.post(
            "/api/upload",
            files={"file": ("test_log.parquet", parquet_bytes, "application/octet-stream")},
        )
        data = response.json()
        assert "metrics" in data, "Response is missing 'metrics' key"
        for key in EXPECTED_METRIC_KEYS:
            assert key in data["metrics"], f"Missing metric key: '{key}'"

    def test_upload_metrics_are_numeric(self, client, parquet_bytes):
        """All metric values must be real finite numbers (not NaN or None)."""
        response = client.post(
            "/api/upload",
            files={"file": ("test_log.parquet", parquet_bytes, "application/octet-stream")},
        )
        metrics = response.json()["metrics"]
        for key, val in metrics.items():
            assert isinstance(val, (int, float)), f"Metric '{key}' is not numeric: {val}"
            assert val == val, f"Metric '{key}' is NaN"  # NaN != NaN

    def test_upload_returns_valid_classification_tier(self, client, parquet_bytes):
        """Classification tier must be one of the three defined tiers."""
        response = client.post(
            "/api/upload",
            files={"file": ("test_log.parquet", parquet_bytes, "application/octet-stream")},
        )
        data = response.json()
        assert "classification" in data
        tier = data["classification"]["tier"]
        assert tier in VALID_TIERS, f"Unexpected tier '{tier}', expected one of {VALID_TIERS}"

    def test_upload_classification_score_in_range(self, client, parquet_bytes):
        """Classification score must be a float between 0.0 and 1.0."""
        response = client.post(
            "/api/upload",
            files={"file": ("test_log.parquet", parquet_bytes, "application/octet-stream")},
        )
        score = response.json()["classification"]["score"]
        assert isinstance(score, float), f"Score is not a float: {score}"
        assert 0.0 <= score <= 1.0, f"Score {score} is out of range [0.0, 1.0]"

    def test_upload_wrong_file_type_returns_400(self, client):
        """Uploading a non-Parquet file should return HTTP 400."""
        response = client.post(
            "/api/upload",
            files={"file": ("robot_log.mcap", b"fake mcap content", "application/octet-stream")},
        )
        assert response.status_code == 400, f"Expected 400 but got {response.status_code}"

    def test_upload_filename_echoed_in_response(self, client, parquet_bytes):
        """The response should echo back the original filename."""
        response = client.post(
            "/api/upload",
            files={"file": ("my_run.parquet", parquet_bytes, "application/octet-stream")},
        )
        data = response.json()
        assert data["filename"] == "my_run.parquet"

    def test_upload_control_precision_is_non_negative(self, client, parquet_bytes):
        """Control precision (RMSE) should never be negative."""
        response = client.post(
            "/api/upload",
            files={"file": ("test_log.parquet", parquet_bytes, "application/octet-stream")},
        )
        control_precision = response.json()["metrics"]["control_precision"]
        assert isinstance(control_precision, float)
        assert control_precision >= 0.0, f"Control precision should be >= 0, got {control_precision}"

    def test_upload_dynamic_stability_is_non_negative(self, client, parquet_bytes):
        """Dynamic stability (pitch/roll variance) should never be negative."""
        response = client.post(
            "/api/upload",
            files={"file": ("test_log.parquet", parquet_bytes, "application/octet-stream")},
        )
        dynamic_stability = response.json()["metrics"]["dynamic_stability"]
        assert isinstance(dynamic_stability, float)
        assert dynamic_stability >= 0.0, f"Dynamic stability should be >= 0, got {dynamic_stability}"

    def test_upload_ldlj_is_negative_or_zero(self, client, parquet_bytes):
        """LDLJ should be a non-positive value."""
        response = client.post(
            "/api/upload",
            files={"file": ("test_log.parquet", parquet_bytes, "application/octet-stream")},
        )
        ldlj = response.json()["metrics"]["smoothness_ldlj"]
        assert isinstance(ldlj, float)
        assert ldlj <= 0.0, f"LDLJ should be <= 0, got {ldlj}"

    def test_upload_symmetry_is_positive(self, client, parquet_bytes):
        """Symmetry index should be >= 0."""
        response = client.post(
            "/api/upload",
            files={"file": ("test_log.parquet", parquet_bytes, "application/octet-stream")},
        )
        symmetry = response.json()["metrics"]["symmetry"]
        if symmetry is not None:
            assert symmetry >= 0.0, f"Symmetry index should be >= 0, got {symmetry}"
