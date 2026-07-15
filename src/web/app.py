import os
import tempfile
import logging
import math
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.ingestion.schema_mapper import SchemaMapper
from src.processing.calibration import calibrate_stereo, CalibrationResult
from src.processing.pose_estimation import estimate_stereo_poses
from src.processing.triangulation import triangulate_pose_sequence
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

# Persistent calibration storage path
CALIBRATION_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "calibration")
CALIBRATION_FILE = os.path.join(CALIBRATION_DIR, "stereo_calibration.json")

# In-memory cache of the active calibration
_active_calibration: CalibrationResult | None = None

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


def _load_cached_calibration() -> CalibrationResult | None:
    """Load calibration from disk if it exists."""
    global _active_calibration
    if _active_calibration is not None:
        return _active_calibration
    if os.path.exists(CALIBRATION_FILE):
        try:
            _active_calibration = CalibrationResult.load(CALIBRATION_FILE)
            return _active_calibration
        except Exception as e:
            logger.warning("Failed to load cached calibration: %s", e)
    return None


@app.post("/api/calibrate")
async def calibrate_cameras(
    left_video: UploadFile = File(...),
    right_video: UploadFile = File(...),
    board_cols: int = Form(10),
    board_rows: int = Form(7),
    square_size: float = Form(0.025),
    marker_size: float = Form(0.015),
):
    """
    Accepts two calibration videos (.mp4/.mov, checkerboard visible in both),
    runs the full stereo calibration pipeline, persists the result,
    and returns the calibration parameters.
    """
    global _active_calibration

    left_suffix = _video_suffix(left_video.filename)
    right_suffix = _video_suffix(right_video.filename)

    left_tmp_path = None
    right_tmp_path = None

    try:
        # Save uploads to temp files
        left_tmp = tempfile.NamedTemporaryFile(suffix=left_suffix, delete=False)
        left_tmp.write(await left_video.read())
        left_tmp.close()
        left_tmp_path = left_tmp.name

        right_tmp = tempfile.NamedTemporaryFile(suffix=right_suffix, delete=False)
        right_tmp.write(await right_video.read())
        right_tmp.close()
        right_tmp_path = right_tmp.name

        logger.info(
            "Running stereo calibration (board=%dx%d, square=%.3fm) …",
            board_cols, board_rows, square_size,
        )

        result = calibrate_stereo(
            left_tmp_path,
            right_tmp_path,
            board_size=(board_cols, board_rows),
            square_size=square_size,
            marker_size=marker_size,
        )

        # Persist and cache
        os.makedirs(CALIBRATION_DIR, exist_ok=True)
        result.save(CALIBRATION_FILE)
        _active_calibration = result

        return JSONResponse(content={
            "status": "success",
            "calibration": result.to_dict(),
        })

    except ValueError as ve:
        return JSONResponse(status_code=400, content={"status": "error", "message": str(ve)})
    except Exception as e:
        logger.error("Calibration failed: %s", e, exc_info=True)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    finally:
        if left_tmp_path and os.path.exists(left_tmp_path):
            os.remove(left_tmp_path)
        if right_tmp_path and os.path.exists(right_tmp_path):
            os.remove(right_tmp_path)


@app.get("/api/calibration_status")
async def calibration_status():
    """Return the current active calibration, or null if none exists."""
    cal = _load_cached_calibration()
    if cal is None:
        return JSONResponse(content={"status": "no_calibration", "calibration": None})
    return JSONResponse(content={"status": "ok", "calibration": cal.to_dict()})


@app.post("/api/upload_av")
async def upload_av_files(
    left_camera: UploadFile = File(...),
    right_camera: UploadFile = File(...),
    task: str = Form("general"),
):
    """
    Accepts two video files (.mp4/.mov, left and right cameras),
    validates them, and saves them to temporary storage for processing.
    """
    left_suffix = _video_suffix(left_camera.filename)
    right_suffix = _video_suffix(right_camera.filename)

    try:
        # Save left camera
        left_tmp = tempfile.NamedTemporaryFile(suffix=left_suffix, delete=False)
        left_contents = await left_camera.read()
        left_tmp.write(left_contents)
        left_tmp.close()

        # Save right camera
        right_tmp = tempfile.NamedTemporaryFile(suffix=right_suffix, delete=False)
        right_contents = await right_camera.read()
        right_tmp.write(right_contents)
        right_tmp.close()

        logger.info(f"Received AV payload. Left: {left_camera.filename}, Right: {right_camera.filename}")

        # Check that a calibration exists
        cal = _load_cached_calibration()
        if cal is None:
            return JSONResponse(status_code=400, content={
                "status": "error",
                "message": "No stereo calibration found. Please calibrate cameras first.",
            })

        # Step 1: 2D Pose Estimation on both cameras
        logger.info("Running 2D pose estimation on stereo pair …")
        left_pose, right_pose = estimate_stereo_poses(
            left_tmp.name, right_tmp.name, device="cpu",
        )

        # Step 2: Triangulate into 3D
        logger.info("Triangulating 2D joints into 3D …")
        poses_3d, valid_mask = triangulate_pose_sequence(
            left_pose.keypoints,
            right_pose.keypoints,
            cal,
            confidence_left=left_pose.confidence,
            confidence_right=right_pose.confidence,
        )

        # Step 3: Biomechanics features
        fps = left_pose.fps or 30.0
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

        # Step 4: Rule-based classification
        classifier = RuleBasedClassifier()
        score, tier = classifier.classify(metrics, task)
        score = _sanitize_floats(score)

        poses_3d_clean = np.where(np.isnan(poses_3d), None, poses_3d).tolist()
        valid_mask_clean = valid_mask.tolist()

        return JSONResponse(content={
            "task": task,
            "status": "success",
            "message": "Analysis complete.",
            "metrics": metrics,
            "poses_3d": poses_3d_clean,
            "valid_mask": valid_mask_clean,
            "classification": {
                "score": score,
                "tier": tier,
            }
        })

    except Exception as e:
        logger.error(f"Error processing AV upload: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)},
        )
    finally:
        if "left_tmp" in locals() and os.path.exists(left_tmp.name):
            os.remove(left_tmp.name)
        if "right_tmp" in locals() and os.path.exists(right_tmp.name):
            os.remove(right_tmp.name)


@app.post("/api/upload_mono")
async def upload_mono_file(
    camera: UploadFile = File(...),
    task: str = Form("general"),
):
    """
    Accepts a single video file (.mp4/.mov) and estimates a 3D pose sequence
    using known G1 bone-length constraints in place of stereo triangulation
    (see src.processing.monocular_depth). Less accurate than the two-camera
    pipeline -- error grows outward from the hip along the kinematic chain --
    but only needs one camera.
    """
    suffix = _video_suffix(camera.filename)

    try:
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(await camera.read())
        tmp.close()

        logger.info(f"Received mono AV payload. Camera: {camera.filename}")

        # Reuse an existing stereo calibration's left-camera intrinsics if
        # available (more accurate); otherwise fall back to a rough guess.
        cal = _load_cached_calibration()
        if cal is not None:
            K, dist = cal.K_left, cal.dist_left
        else:
            logger.warning("No calibration found; using default intrinsics for monocular depth.")
            K, dist = _DEFAULT_MONO_K, _DEFAULT_MONO_DIST

        logger.info("Running 2D pose estimation + monocular depth lifting …")
        pose_result, poses_3d, valid_mask = estimate_monocular_pose_and_depth(
            tmp.name, K, dist, device="cpu",
        )

        # Biomechanics features
        fps = pose_result.fps or 30.0
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

        classifier = RuleBasedClassifier()
        score, tier = classifier.classify(metrics, task)
        score = _sanitize_floats(score)

        poses_3d_clean = np.where(np.isnan(poses_3d), None, poses_3d).tolist()
        valid_mask_clean = valid_mask.tolist()

        return JSONResponse(content={
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
        })

    except Exception as e:
        logger.error(f"Error processing mono upload: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)},
        )
    finally:
        if "tmp" in locals() and os.path.exists(tmp.name):
            os.remove(tmp.name)


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
