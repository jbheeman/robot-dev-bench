"""
database.py
SQLite-backed persistence for the benchmarking dashboard.

Two tables:
  - logs:        registered physical-robot telemetry logs, their stored parquet
                 file path, and the validation report produced at ingest time.
  - evaluations: classification results, grouped into batches so one
                 multi-model comparison run can be recalled as a unit.

Uses only the stdlib sqlite3 module. Each method opens a short-lived
connection, so the object is safe to share across FastAPI requests.
"""

import os
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_DEFAULT_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data")
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    model_name TEXT NOT NULL,
    task TEXT NOT NULL DEFAULT 'general',
    stored_path TEXT NOT NULL,
    uploaded_at TEXT NOT NULL,
    row_count INTEGER,
    duration_sec REAL,
    schema_format TEXT,
    validation_status TEXT,
    validation_report TEXT,
    notes TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT NOT NULL,
    log_id INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    task TEXT NOT NULL,
    metrics TEXT NOT NULL,
    score REAL NOT NULL,
    tier TEXT NOT NULL,
    evaluated_at TEXT NOT NULL,
    FOREIGN KEY (log_id) REFERENCES logs (id)
);
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class BenchmarkDB:
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or os.environ.get("G1_BENCH_DATA_DIR", _DEFAULT_DATA_DIR)
        self.logs_dir = os.path.join(self.data_dir, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)
        self.db_path = os.path.join(self.data_dir, "benchmark.db")
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # --- Log registry -------------------------------------------------------

    def store_log_file(self, filename: str, contents: bytes) -> str:
        """Persist raw parquet bytes into the managed logs directory."""
        safe_name = os.path.basename(filename)
        stored_path = os.path.join(self.logs_dir, f"{uuid.uuid4().hex[:12]}_{safe_name}")
        with open(stored_path, "wb") as f:
            f.write(contents)
        return stored_path

    def add_log(
        self,
        filename: str,
        model_name: str,
        task: str,
        stored_path: str,
        row_count: Optional[int],
        duration_sec: Optional[float],
        schema_format: Optional[str],
        validation_status: Optional[str],
        validation_report: Optional[Dict[str, Any]],
        notes: str = "",
    ) -> Dict[str, Any]:
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO logs
                   (filename, model_name, task, stored_path, uploaded_at, row_count,
                    duration_sec, schema_format, validation_status, validation_report, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    filename, model_name, task, stored_path, _utcnow(), row_count,
                    duration_sec, schema_format, validation_status,
                    json.dumps(validation_report) if validation_report is not None else None,
                    notes,
                ),
            )
            log_id = cur.lastrowid
        return self.get_log(log_id)

    def get_log(self, log_id: int) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM logs WHERE id = ?", (log_id,)).fetchone()
        return self._log_row_to_dict(row) if row else None

    def list_logs(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM logs ORDER BY id DESC").fetchall()
        return [self._log_row_to_dict(r) for r in rows]

    def delete_log(self, log_id: int) -> bool:
        record = self.get_log(log_id)
        if record is None:
            return False
        with self._connect() as conn:
            conn.execute("DELETE FROM evaluations WHERE log_id = ?", (log_id,))
            conn.execute("DELETE FROM logs WHERE id = ?", (log_id,))
        if record["stored_path"] and os.path.exists(record["stored_path"]):
            os.remove(record["stored_path"])
        return True

    @staticmethod
    def _log_row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        if d.get("validation_report"):
            d["validation_report"] = json.loads(d["validation_report"])
        return d

    # --- Evaluations --------------------------------------------------------

    def add_evaluation(
        self,
        batch_id: str,
        log_id: int,
        model_name: str,
        task: str,
        metrics: Dict[str, Any],
        score: float,
        tier: str,
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO evaluations
                   (batch_id, log_id, model_name, task, metrics, score, tier, evaluated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (batch_id, log_id, model_name, task, json.dumps(metrics), score, tier, _utcnow()),
            )
            return cur.lastrowid

    def list_evaluations(self, task: Optional[str] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM evaluations"
        params: tuple = ()
        if task:
            query += " WHERE task = ?"
            params = (task,)
        query += " ORDER BY id DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["metrics"] = json.loads(d["metrics"])
            out.append(d)
        return out


# Lazily-created singleton so tests can inject an isolated instance
# (set src.storage.database._db_instance directly) regardless of import order.
_db_instance: Optional[BenchmarkDB] = None


def get_db() -> BenchmarkDB:
    global _db_instance
    if _db_instance is None:
        _db_instance = BenchmarkDB()
    return _db_instance
