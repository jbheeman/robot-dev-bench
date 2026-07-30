import os
import tempfile
import logging
import math
import numpy as np
import pandas as pd
import torch
import subprocess
import shutil
from typing import Optional

# PyTorch 2.6 defaults weights_only=True which breaks mmengine/mmpose checkpoints.
# Monkeypatch it here to default to False.
_original_load = torch.load
def _legacy_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_load(*args, **kwargs)
torch.load = _legacy_load

import threading
from fastapi import BackgroundTasks
from src.processing.jobs import JobStore
from src.processing.pose_estimation import PoseEstimator
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.ingestion.schema_mapper import SchemaMapper
from src.features.biomechanics import (
    compute_walk_grade_2d,
    compute_jumping_metrics_2d,
    compute_manipulation_metrics_2d,
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
ALLOWED_VIDEO_EXTENSIONS = (".mp4", ".mov", ".webm")
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
    if isinstance(value, (float, np.floating)):
        val = float(value)
        if not math.isfinite(val):
            return 0.0
        return val
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return [_sanitize_floats(v) for v in value.tolist()]
    return value







def _process_upload_task(job_id: str, tmp_name: str, filename: str, task: str,
                         ref_length_cm: float = 20.0,
                         camera_view: str = "side",
                         manual_bbox: Optional[list] = None):
    try:
        logger.info(f"Received AV payload. Camera: {filename}")
        
        transcoded_dir = os.path.join(STATIC_DIR, "transcoded")
        os.makedirs(transcoded_dir, exist_ok=True)
        transcoded_filename = f"{job_id}.mp4"
        transcoded_path = os.path.join(transcoded_dir, transcoded_filename)
        
        JOB_STORE.update_job(job_id, 0.05, "Transcoding video for web playback...")
        try:
            subprocess.run([
                'ffmpeg', '-y', '-i', tmp_name,
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-c:a', 'aac', '-b:a', '128k',
                transcoded_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            video_url = f"/transcoded/{transcoded_filename}"
        except Exception as e:
            logger.warning(f"Failed to transcode video: {e}")
            video_url = None

        JOB_STORE.update_job(job_id, 0.1, "Starting pose estimation...")

        K, dist = _DEFAULT_MONO_K, _DEFAULT_MONO_DIST

        device = get_device()
        logger.info(f"Running 2D pose estimation on {device}...")
        
        def progress_cb(pct: float, msg: str):
            overall_pct = 0.1 + (pct * 0.8)
            JOB_STORE.update_job(job_id, overall_pct, msg)

        global GLOBAL_ESTIMATOR
        with INFERENCE_LOCK:
            if GLOBAL_ESTIMATOR is None:
                GLOBAL_ESTIMATOR = PoseEstimator(device=device)

            pose_result = GLOBAL_ESTIMATOR.estimate_from_video(
                tmp_name, max_frames=None, progress_callback=progress_cb, task=task, manual_bbox=manual_bbox
            )

        JOB_STORE.update_job(job_id, 0.9, "Extracting biomechanical features...")

        fps = pose_result.fps or 30.0

        # ── 2D Pipeline: grade directly from pixel keypoints ──
        walk_grade = compute_walk_grade_2d(
            pose_result.keypoints,
            pose_result.confidence,
            fps,
            ref_length_cm=ref_length_cm,
            view=camera_view,
        )
        
        jumping_metrics = compute_jumping_metrics_2d(
            pose_result.keypoints,
            pose_result.confidence,
            fps,
            ref_length_cm=ref_length_cm,
        )
        
        manipulation_metrics = compute_manipulation_metrics_2d(
            pose_result.keypoints,
            pose_result.confidence,
            fps,
            ref_length_cm=ref_length_cm,
            objects=pose_result.objects,
        )
        
        cm_per_px = walk_grade.get("cm_per_pixel", 0.0)
        
        metrics = {
            "walk_grade": walk_grade.get("walk_grade", 0.0),
            "mean_clearance_cm": walk_grade.get("mean_clearance_cm", 0.0),
            "stride_length_m": walk_grade.get("stride_length_m", 0.0),
            "speed_m_s": walk_grade.get("speed_m_s", 0.0),
            "torso_oscillation_cm": walk_grade.get("torso_oscillation_cm", 0.0),
            "fall_detected": 1.0 if (walk_grade.get("fall_detected", False) or jumping_metrics.get("fall_detected", False)) else 0.0,
            "cm_per_pixel": walk_grade.get("cm_per_pixel", 0.0),
            "flight_time_s": jumping_metrics.get("flight_time", 0.0),
            "peak_z_accel_g": jumping_metrics.get("peak_z_accel", 0.0),
            "landing_jerk": jumping_metrics.get("landing_jerk", 0.0),
            "wrist_jerk": manipulation_metrics.get("wrist_jerk", 0.0),
            "red_block_displacement_cm": manipulation_metrics.get("red_block_displacement_cm", 0.0),
            "white_block_displacement_cm": manipulation_metrics.get("white_block_displacement_cm", 0.0),
            "wrist_to_block_min_dist_cm": manipulation_metrics.get("wrist_to_block_min_dist_cm", 100.0),
            "task_duration_s": manipulation_metrics.get("task_duration_s", 0.0),
            "block_path_efficiency": manipulation_metrics.get("block_path_efficiency", 0.0),
        }
        metrics = _sanitize_floats(metrics)

        JOB_STORE.update_job(job_id, 0.95, "Running classifier...")
        classifier = RuleBasedClassifier()
        score, tier, contributions = classifier.classify(metrics, task)
        score = _sanitize_floats(score)

        keypoints_list = _sanitize_floats(pose_result.keypoints.tolist())
        confidence_list = _sanitize_floats(pose_result.confidence.tolist())

        objects_list = None
        if pose_result.objects:
            objects_list = {k: _sanitize_floats(v.tolist()) for k, v in pose_result.objects.items()}

        result_payload = {
            "task": task,
            "status": "success",
            "pipeline_mode": "2d",
            "message": f"Analysis complete (2D overlay, {walk_grade.get('cm_per_pixel', 0):.3f} cm/px).",
            "metrics": metrics,
            "video_url": video_url,
            "keypoints_2d": keypoints_list,
            "confidence_2d": confidence_list,
            "objects_2d": objects_list,
            "frame_width": pose_result.frame_width,
            "frame_height": pose_result.frame_height,
            "fps": fps,
            "classification": {
                "score": score,
                "tier": tier,
                "contributions": contributions,
            },
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

@app.post("/api/thumbnail")
async def get_thumbnail(file: UploadFile = File(...)):
    import cv2
    suffix = _video_suffix(file.filename)
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(await file.read())
    tmp.close()
    
    cap = cv2.VideoCapture(tmp.name)
    ret, frame = cap.read()
    cap.release()
    os.remove(tmp.name)
    
    if not ret:
        return JSONResponse(status_code=400, content={"error": "Could not read video frame"})
        
    _, buffer = cv2.imencode('.jpg', frame)
    return Response(content=buffer.tobytes(), media_type="image/jpeg")



@app.post("/api/upload_av")
async def upload_av_file(
    background_tasks: BackgroundTasks,
    camera: UploadFile = File(...),
    task: str = Form("general"),
    ref_length_cm: float = Form(20.0),
    camera_view: str = Form("side"),
    crop_x: Optional[int] = Form(None),
    crop_y: Optional[int] = Form(None),
    crop_w: Optional[int] = Form(None),
    crop_h: Optional[int] = Form(None),
):
    suffix = _video_suffix(camera.filename)
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(await camera.read())
    tmp.close()

    process_path = tmp.name
    # Convert webm / non-mp4 recordings to standard x264 mp4 via ffmpeg
    # so OpenCV decodes every single frame reliably without codec errors
    if suffix in (".webm", ".mov"):
        mp4_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        mp4_tmp.close()
        try:
            import subprocess
            subprocess.run([
                "ffmpeg", "-y", "-i", tmp.name,
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                mp4_tmp.name
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            os.remove(tmp.name)
            process_path = mp4_tmp.name
            logger.info(f"Converted {camera.filename} ({suffix}) to standard MP4 via ffmpeg")
        except Exception as e:
            logger.warning(f"ffmpeg conversion failed: {e}")
            process_path = tmp.name

    manual_bbox = None
    if crop_x is not None and crop_y is not None and crop_w is not None and crop_h is not None:
        manual_bbox = [crop_x, crop_y, crop_x + crop_w, crop_y + crop_h]

    job_id = JOB_STORE.create_job()
    background_tasks.add_task(
        _process_upload_task, job_id, process_path, camera.filename, task,
        ref_length_cm, camera_view, manual_bbox
    )

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
            score, tier, contribs = classifier.classify(metrics, task=task)

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
                "contributions": contribs if task != "testing" else {}
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
    uvicorn.run("src.web.app:app", host="localhost", port=3000, reload=True)
