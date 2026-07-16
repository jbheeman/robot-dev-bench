import re

with open("src/web/app.py", "r") as f:
    content = f.read()

# 1. Imports
imports = """import torch
from fastapi import BackgroundTasks
from src.processing.jobs import JobStore
"""
content = re.sub(r'from fastapi import FastAPI', imports + 'from fastapi import FastAPI', content)

# 2. JOB_STORE
job_store = """
JOB_STORE = JobStore()

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
"""
content = re.sub(r'ALLOWED_VIDEO_EXTENSIONS = \("\.mp4", "\.mov"\)', 'ALLOWED_VIDEO_EXTENSIONS = (".mp4", ".mov")' + job_store, content)

# 3. New process function
process_fn = """
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

        pose_result, poses_3d, valid_mask = estimate_monocular_pose_and_depth(
            tmp_name, K, dist, device=device, progress_callback=progress_cb
        )

        JOB_STORE.update_job(job_id, 0.9, "Extracting biomechanical features...")

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
"""

# 4. Refactor upload_av_file
upload_av = """@app.post("/api/upload_av")
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
"""

content = re.sub(r'@app\.post\("/api/upload_av"\)\nasync def upload_av_file.*?finally:\n        if "tmp" in locals\(\) and os\.path\.exists\(tmp\.name\):\n            os\.remove\(tmp\.name\)\n', process_fn + '\n\n' + upload_av, content, flags=re.DOTALL)

with open("src/web/app.py", "w") as f:
    f.write(content)
