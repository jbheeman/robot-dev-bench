import io
import os
import uuid
import asyncio
import tempfile
import logging
import numpy as np
import pandas as pd
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.ingestion.schema_mapper import SchemaMapper
from src.ingestion.log_validator import validate_log
from src.classification.rules import RuleBasedClassifier
from src.classification.evaluator import run_parallel_evaluations, build_calibration_summary
from src.features.extractor import extract_metrics_from_dataframe
from src.storage.database import get_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Unitree G1-Edu Benchmarking Dashboard")

# Define path to static files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

# Initialize our classifier
classifier = RuleBasedClassifier()


@app.post("/api/upload")
async def upload_log_file(file: UploadFile = File(...), task: str = Form("general")):
    """
    Accepts a .parquet robot telemetry log, runs the full benchmarking pipeline,
    and returns real classification results.
    """
    if not file.filename.endswith(".parquet"):
        raise HTTPException(
            status_code=400,
            detail="Only .parquet files are supported. Please convert your log to Parquet format first.",
        )

    # Save the uploaded file to a temp path so pandas can read it
    try:
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        # Load into DataFrame
        df = pd.read_parquet(tmp_path, engine="pyarrow")
        logger.info(f"Loaded '{file.filename}': {len(df)} rows, columns: {list(df.columns)}")

        # If it's a multi-episode dataset, take only the first episode to avoid stitching discontinuities
        if 'episode_index' in df.columns:
            first_episode = df['episode_index'].iloc[0]
            df = df[df['episode_index'] == first_episode]
            logger.info(f"Filtered to episode {first_episode}: {len(df)} rows remain")

        # Normalise schema
        df = SchemaMapper.normalise(df)
        logger.info(f"Normalised columns: {list(df.columns)}")

        # Extract playback data (downsampled to ~30Hz for the viewer)
        duration_sec = 0.0
        if 'tick' in df and len(df) > 1:
            duration_sec = (df['tick'].iloc[-1] - df['tick'].iloc[0]) / 1000.0
            
        target_frames = max(100, int(duration_sec * 30)) if duration_sec > 0 else 300
        # Cap at 900 frames (30 seconds of 30fps playback) to prevent massive JSON payloads
        target_frames = min(900, target_frames)
        
        if len(df) > target_frames:
            indices = np.linspace(0, len(df) - 1, target_frames, dtype=int)
            playback_df = df.iloc[indices]
        else:
            playback_df = df
            
        # Calculate timeseries data for charts
        if 'tick' in playback_df and 'q' in playback_df and len(playback_df) > 1:
            t_sec = playback_df['tick'].to_numpy() / 1000.0
            dt = np.gradient(t_sec)
            dt[dt == 0] = 1e-6
            
            q_mat = np.array(playback_df['q'].tolist())
            v = np.gradient(q_mat, axis=0) / dt[:, np.newaxis]
            a = np.gradient(v, axis=0) / dt[:, np.newaxis]
            j = np.gradient(a, axis=0) / dt[:, np.newaxis]
            
            global_velocity = np.mean(np.abs(v), axis=1).tolist()
            global_acceleration = np.mean(np.abs(a), axis=1).tolist()
            global_jerk = np.mean(np.abs(j), axis=1).tolist()
            com_oscillation = np.var(v, axis=1).tolist()
            
            anomalies = {
                "Max Acceleration": int(np.argmax(global_acceleration)),
                "Max Jerk": int(np.argmax(global_jerk)),
                "Max CoM Wobble": int(np.argmax(com_oscillation))
            }
        else:
            global_velocity = []
            global_acceleration = []
            global_jerk = []
            com_oscillation = []
            anomalies = {}
            
        playback_data = {
            "ticks": [float(x) for x in playback_df['tick']] if 'tick' in playback_df else [],
            "q": [[float(val) for val in row] for row in playback_df['q']] if 'q' in playback_df else [],
            "timeseries": {
                "velocity": global_velocity,
                "acceleration": global_acceleration,
                "jerk": global_jerk,
                "com_oscillation": com_oscillation
            },
            "anomalies": anomalies
        }

        # Classify using real metrics unless it's testing only
        if task == "testing":
            score = 0.0
            tier = "Testing (No Score)"
            # Provide zeroed metrics for testing mode
            metrics = {
                "control_precision": 0.0,
                "dynamic_stability": 0.0,
                "cost_of_transport": 0.0,
                "system_latency": 0.0,
                "smoothness_ldlj": 0.0,
                "smoothness_sparc": 0.0,
                "symmetry": 0.0,
                "periodicity": 0.0,
                "rom_utilisation": 0.0,
                "flight_time": 0.0,
                "peak_z_accel": 0.0,
                "landing_jerk": 0.0,
                "com_oscillation": 0.0,
                "transition_time": 0.0
            }
        else:
            metrics = extract_metrics_from_dataframe(df)
            logger.info(f"Extracted metrics: {metrics}")
            score, tier = classifier.classify(metrics, task=task)

        return JSONResponse(content={
            "filename": file.filename,
            "task": task,
            "metrics": metrics,
            "classification": {
                "score": round(score, 3),
                "tier": tier,
            },
            "playback": playback_data,
            "status": "success",
        })

    except Exception as e:
        logger.error(f"Error processing '{file.filename}': {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)},
        )
    finally:
        # Always clean up the temp file
        if "tmp_path" in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)


# --- Log Integration: registered log library with pipeline validation --------

@app.post("/api/logs")
async def register_log(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    task: str = Form("general"),
    notes: str = Form(""),
):
    """
    Feeds a physical-robot log into the database: validates its format and
    end-to-end pipeline compatibility up front (no model run on the physical
    robot required), stores the parquet file, and records the validation report.
    """
    if not file.filename.endswith(".parquet"):
        raise HTTPException(
            status_code=400,
            detail="Only .parquet files are supported. Please convert your log to Parquet format first.",
        )

    contents = await file.read()
    try:
        df = pd.read_parquet(io.BytesIO(contents), engine="pyarrow")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"File is not readable as Parquet: {e}",
        )

    report = validate_log(df, filename=file.filename)

    db = get_db()
    stored_path = db.store_log_file(file.filename, contents)
    record = db.add_log(
        filename=file.filename,
        model_name=model_name,
        task=task,
        stored_path=stored_path,
        row_count=report["row_count"],
        duration_sec=report["duration_sec"],
        schema_format=report["schema_format"],
        validation_status=report["status"],
        validation_report=report,
        notes=notes,
    )
    logger.info(
        f"Registered log '{file.filename}' (model={model_name}, task={task}) "
        f"as id={record['id']} with validation status '{report['status']}'"
    )
    return JSONResponse(content={"status": "success", "log": record})


@app.get("/api/logs")
async def list_logs():
    """Lists all registered logs with their validation status."""
    return JSONResponse(content={"status": "success", "logs": get_db().list_logs()})


@app.get("/api/logs/{log_id}")
async def get_log(log_id: int):
    """Returns one registered log, including its full validation report."""
    record = get_db().get_log(log_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Log {log_id} not found.")
    return JSONResponse(content={"status": "success", "log": record})


@app.delete("/api/logs/{log_id}")
async def delete_log(log_id: int):
    """Removes a registered log, its stored file, and its evaluations."""
    if not get_db().delete_log(log_id):
        raise HTTPException(status_code=404, detail=f"Log {log_id} not found.")
    return JSONResponse(content={"status": "success"})


# --- Multi-Model Evaluation: parallel scoring + calibration ------------------

class EvaluateRequest(BaseModel):
    log_ids: List[int]
    task: str = "general"


@app.post("/api/evaluate")
async def evaluate_models(request: EvaluateRequest):
    """
    Runs the benchmarking pipeline over several registered logs in parallel —
    e.g. baseline AI models recorded on the same robot and task — persists the
    results as one batch, and returns a cross-model calibration summary.
    """
    if not request.log_ids:
        raise HTTPException(status_code=400, detail="log_ids must not be empty.")

    db = get_db()
    records = []
    for log_id in request.log_ids:
        record = db.get_log(log_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Log {log_id} not found.")
        records.append(record)

    results = await asyncio.to_thread(run_parallel_evaluations, records, request.task)

    batch_id = uuid.uuid4().hex[:12]
    for result in results:
        if result["status"] == "ok":
            db.add_evaluation(
                batch_id=batch_id,
                log_id=result["log_id"],
                model_name=result["model_name"],
                task=request.task,
                metrics=result["metrics"],
                score=result["score"],
                tier=result["tier"],
            )

    calibration = build_calibration_summary(results, request.task)
    return JSONResponse(content={
        "status": "success",
        "batch_id": batch_id,
        "task": request.task,
        "results": results,
        "calibration": calibration,
    })


@app.get("/api/evaluations")
async def list_evaluations(task: str = None):
    """Evaluation history, optionally filtered by task."""
    return JSONResponse(content={"status": "success", "evaluations": get_db().list_evaluations(task)})


# Mount the static directory to serve the frontend
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.web.app:app", host="0.0.0.0", port=3000, reload=True)
