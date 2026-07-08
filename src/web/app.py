import os
import tempfile
import logging
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.classification.rules import RuleBasedClassifier
from src.features.metrics import (
    compute_control_precision,
    compute_cost_of_transport,
    compute_control_latency,
    compute_hardware_stress,
)
from src.features.stability import compute_imu_variance, compute_com_stability

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Unitree G1-Edu Benchmarking Dashboard")

# Define path to static files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

# Initialize our classifier
classifier = RuleBasedClassifier()


def extract_metrics_from_dataframe(df: pd.DataFrame) -> dict:
    """
    Runs all feature extractors against a telemetry DataFrame loaded from a .parquet file.
    Returns a flat dict of scalar metrics ready for the classifier.
    """
    # --- Control Precision (RMSE) ---
    precision = compute_control_precision(df)
    rmse = precision.get("mean_rmse", 0.0)

    # --- Cost of Transport ---
    cot = compute_cost_of_transport(df)

    # --- Control Latency ---
    latency_result = compute_control_latency(df)
    latency_s = latency_result.get("mean_latency_seconds", 0.0)
    latency_ms = latency_s * 1000.0

    # --- Hardware Stress ---
    stress_result = compute_hardware_stress(df)
    # Normalise: overall_max_torque as a fraction of the 40 Nm limit threshold
    overall_max_torque = stress_result.get("overall_max_torque", 0.0)
    stress_normalised = min(overall_max_torque / 40.0, 1.0)

    # --- IMU / Stability Variance ---
    imu_result = compute_imu_variance(df)
    # Use the average of roll + pitch variance as the stability signal
    roll_var = imu_result.get("roll_variance", 0.0)
    pitch_var = imu_result.get("pitch_variance", 0.0)
    imu_variance = (roll_var + pitch_var) / 2.0

    return {
        "rmse": round(rmse, 4),
        "cot": round(cot, 3),
        "latency_ms": round(latency_ms, 2),
        "stress": round(stress_normalised, 3),
        "imu_variance": round(imu_variance, 4),
    }


@app.post("/api/upload")
async def upload_log_file(file: UploadFile = File(...)):
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

        # Extract real metrics
        metrics = extract_metrics_from_dataframe(df)
        logger.info(f"Extracted metrics: {metrics}")

        # Classify using real metrics
        score, tier = classifier.classify(metrics)

        return JSONResponse(content={
            "filename": file.filename,
            "metrics": metrics,
            "classification": {
                "score": round(score, 3),
                "tier": tier,
            },
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


# Mount the static directory to serve the frontend
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.web.app:app", host="0.0.0.0", port=8000, reload=True)
