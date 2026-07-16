import os
import tempfile
import logging
import math
import numpy as np
import pandas as pd
import torch
import threading
from fastapi import BackgroundTasks
from src.processing.jobs import JobStore
from src.processing.pose_estimation import PoseEstimator
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.ingestion.schema_mapper import SchemaMapper
from src.processing.monocular_depth import estimate_monocular_pose_and_depth
from src.features.biomechanics import (
    compute_smoothness_3d,
    compute_sparc_3d,
    compute_symmetry_3d,
    compute_periodicity_3d,
    compute_rom_3d,
    compute_jumping_metrics_3d,
    compute_transition_metrics_3d,
    compute_smoothness,
    compute_spectral_arc_length,
    compute_symmetry,
    compute_periodicity,
    compute_range_of_motion,
    compute_jumping_metrics,
    compute_transition_metrics,
)
from src.classification.rules import RuleBasedClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Unitree G1-Edu Benchmarking Dashboard (AV Mode)")

# Define path to static files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)


# Rough intrinsic guess used for /api/upload_mono when no stereo calibration
# has been run yet (monocular depth only needs one camera's K/dist, but the
# app currently only ever calibrates a stereo pair).
_DEFAULT_MONO_K = np.array([[1000.0, 0, 640.0], [0, 1000.0, 360.0], [0, 0, 1.0]])
_DEFAULT_MONO_DIST = np.zeros(5)

# Video containers accepted for camera/calibration uploads. cv2/moviepy read
# these via ffmpeg, which sniffs the container from content rather than the
# extension, but we still validate + preserve the extension on the temp file
# so downstream tools that DO key off the suffix behave correctly.
ALLOWED_VIDEO_EXTENSIONS = (".mp4", ".mov")
JOB_STORE = JobStore()
INFERENCE_LOCK = threading.Lock()
GLOBAL_ESTIMATOR = None

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"



def _video_suffix(filename: str) -> str:
    """Validate a video filename's extension, returning it (lowercased) for
    use as a temp file suffix."""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Only {'/'.join(ALLOWED_VIDEO_EXTENSIONS)} files are supported.",
        )
    return ext


def _sanitize_floats(value):
    """Replace NaN/Infinity with 0.0 so responses stay valid JSON.

    Biomechanics metrics can come out NaN/Inf when a video has no valid
    pose detections (e.g. no visible subject), which the standard JSON
    encoder rejects outright.
    """
    if isinstance(value, dict):
        return {k: _sanitize_floats(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_floats(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return 0.0
    return value







def _process_upload_task(job_id: str, tmp_name: str, filename: str, task: str):
    try:
        logger.info(f"Received mono AV payload. Camera: {filename}")
        JOB_STORE.update_job(job_id, 0.1, "Starting pose estimation...")

        K, dist = _DEFAULT_MONO_K, _DEFAULT_MONO_DIST

        device = get_device()
        logger.info(f"Running 2D pose estimation + monocular depth lifting on {device}...")
        
        def progress_cb(pct: float, msg: str):
            # Scale 2D pose estimation to be 10% -> 90% of the job
            overall_pct = 0.1 + (pct * 0.8)
            JOB_STORE.update_job(job_id, overall_pct, msg)

        global GLOBAL_ESTIMATOR
        with INFERENCE_LOCK:
            if GLOBAL_ESTIMATOR is None:
                GLOBAL_ESTIMATOR = PoseEstimator(device=device)

            pose_result, poses_3d, valid_mask = estimate_monocular_pose_and_depth(
                tmp_name, K, dist, device=device, progress_callback=progress_cb, estimator=GLOBAL_ESTIMATOR
            )

        JOB_STORE.update_job(job_id, 0.9, "Extracting biomechanical features...")

        # Biomechanics features
        fps = pose_result.fps or 30.0

        # Apply a temporal lowpass filter to the raw monocular 3D poses
        # to eliminate the aggressive Z-axis jitter inherent to monocular lifting.
        from src.features.biomechanics import _interpolate_nans
        from src.processing.filter import TelemetryFilter
        
        poses_3d = _interpolate_nans(poses_3d)
        filter = TelemetryFilter(sample_rate=fps, cutoff_freq=5.0, order=4)
        
        T, J, C = poses_3d.shape
        poses_3d_flat = poses_3d.reshape(T, J * C)
        poses_3d_smoothed = filter.filter_array(poses_3d_flat)
        poses_3d = poses_3d_smoothed.reshape(T, J, C)

        smoothness = compute_smoothness_3d(poses_3d, fps)
        sparc = compute_sparc_3d(poses_3d, fps)
        symmetry = compute_symmetry_3d(poses_3d)
        periodicity = compute_periodicity_3d(poses_3d, fps)
        rom = compute_rom_3d(poses_3d)
        jumping = compute_jumping_metrics_3d(poses_3d, fps)
        transition = compute_transition_metrics_3d(poses_3d, fps)

        metrics = {
            "smoothness_ldlj": smoothness.get("mean_ldlj", 0.0),
            "smoothness_sparc": sparc.get("mean_sparc", 0.0),
            "symmetry": symmetry.get("mean_symmetry_index", 0.0),
            "periodicity": periodicity.get("regularity_score", 0.0),
            "rom_utilisation": rom.get("mean_rom", 0.0),
            "flight_time": jumping.get("flight_time", 0.0),
            "peak_z_accel": jumping.get("peak_z_accel", 0.0),
            "landing_jerk": jumping.get("landing_jerk", 0.0),
            "com_oscillation": transition.get("com_oscillation", 0.0),
            "transition_time": transition.get("transition_time", 0.0),
        }
        metrics = _sanitize_floats(metrics)

        JOB_STORE.update_job(job_id, 0.95, "Running classifier...")

        classifier = RuleBasedClassifier()
        score, tier = classifier.classify(metrics, task)
        score = _sanitize_floats(score)

        poses_3d_clean = np.where(np.isnan(poses_3d), None, poses_3d).tolist()
        valid_mask_clean = valid_mask.tolist()
        
        result_payload = {
            "task": task,
            "status": "success",
            "message": "Analysis complete (monocular depth estimate).",
            "metrics": metrics,
            "poses_3d": poses_3d_clean,
            "valid_mask": valid_mask_clean,
            "classification": {
                "score": score,
                "tier": tier,
            }
        }
        
        JOB_STORE.finish_job(job_id, result_payload)

    except Exception as e:
        logger.error(f"Error processing mono upload: {e}", exc_info=True)
        JOB_STORE.fail_job(job_id, str(e))
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


@app.get("/api/job_status/{job_id}")
async def get_job_status(job_id: str):
    job = JOB_STORE.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(content=job)


@app.post("/api/upload_av")
async def upload_av_file(
    background_tasks: BackgroundTasks,
    camera: UploadFile = File(...),
    task: str = Form("general"),
):
    suffix = _video_suffix(camera.filename)
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(await camera.read())
    tmp.close()

    job_id = JOB_STORE.create_job()
    background_tasks.add_task(_process_upload_task, job_id, tmp.name, camera.filename, task)

    return JSONResponse(content={"job_id": job_id, "status": "accepted"})


def extract_metrics_from_dataframe(df: pd.DataFrame) -> dict:
    """
    Runs all feature extractors against a telemetry DataFrame loaded from a .parquet file.
    Returns a flat dict of scalar metrics ready for the classifier.
    """
    smoothness = compute_smoothness(df)
    sparc = compute_spectral_arc_length(df)
    symmetry = compute_symmetry(df)
    periodicity = compute_periodicity(df)
    rom = compute_range_of_motion(df)
    jumping = compute_jumping_metrics(df)
    transitions = compute_transition_metrics(df)
    
    mean_ldlj = smoothness.get("mean_ldlj")
    mean_sparc = sparc.get("mean_sparc")
    mean_symmetry_index = symmetry.get("mean_symmetry_index")
    regularity_score = periodicity.get("regularity_score")
    mean_rom = rom.get("mean_rom")

    if mean_symmetry_index is not None:
        mean_symmetry_index = round(mean_symmetry_index, 3)

    return {
        "smoothness_ldlj": round(mean_ldlj, 3) if mean_ldlj is not None else 0.0,
        "smoothness_sparc": round(mean_sparc, 3) if mean_sparc is not None else 0.0,
        "symmetry": mean_symmetry_index,
        "periodicity": round(regularity_score, 3) if regularity_score is not None else 0.0,
        "rom_utilisation": round(mean_rom, 3) if mean_rom is not None else 0.0,
        "flight_time": round(jumping.get("flight_time", 0.0), 3),
        "peak_z_accel": round(jumping.get("peak_z_accel", 0.0), 3),
        "landing_jerk": round(jumping.get("landing_jerk", 0.0), 3),
        "com_oscillation": round(transitions.get("com_oscillation", 0.0), 3),
        "transition_time": round(transitions.get("transition_time", 0.0), 3),
    }


class ReclassifyRequest(BaseModel):
    task: str
    metrics: dict

@app.post("/api/reclassify")
async def reclassify_metrics(req: ReclassifyRequest):
    """
    Re-evaluates an existing set of metrics against a new task profile.
    """
    if req.task == "testing":
        return JSONResponse(content={
            "score": 0.0,
            "tier": "Testing (No Score)",
            "task": req.task
        })
    
    classifier = RuleBasedClassifier()
    score, tier = classifier.classify(req.metrics, task=req.task)
    return JSONResponse(content={
        "score": round(score, 3),
        "tier": tier,
        "task": req.task
    })


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
            classifier = RuleBasedClassifier()
            score, tier = classifier.classify(metrics, task=task)

        metrics = _sanitize_floats(metrics)
        score = _sanitize_floats(round(score, 3))
        playback_data = _sanitize_floats(playback_data)

        return JSONResponse(content={
            "filename": file.filename,
            "task": task,
            "metrics": metrics,
            "classification": {
                "score": score,
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



# Mount the static directory to serve the frontend
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.web.app:app", host="0.0.0.0", port=3000, reload=True)
