import re

with open("src/web/app.py", "r") as f:
    content = f.read()

# 1. Remove calibration and triangulation imports
content = re.sub(r'from src\.processing\.calibration import calibrate_stereo, CalibrationResult\n', '', content)
content = re.sub(r'from src\.processing\.pose_estimation import estimate_stereo_poses\n', '', content)
content = re.sub(r'from src\.processing\.triangulation import triangulate_pose_sequence\n', '', content)

# 2. Remove calibration variables and functions
content = re.sub(r'# Persistent calibration storage path.*?\n_active_calibration: CalibrationResult \| None = None\n', '', content, flags=re.DOTALL)
content = re.sub(r'def _load_cached_calibration.*?return None\n', '', content, flags=re.DOTALL)
content = re.sub(r'@app\.post\("/api/calibrate"\).*?@app\.get\("/api/calibration_status"\).*?return JSONResponse\(content=\{"status": "ok", "calibration": cal\.to_dict\(\)\}\)\n', '', content, flags=re.DOTALL)

# 3. Replace stereo /api/upload_av with the mono one (renamed to upload_av)
# First remove the stereo one
content = re.sub(r'@app\.post\("/api/upload_av"\).*?@app\.post\("/api/upload_mono"\)', '@app.post("/api/upload_av")', content, flags=re.DOTALL)

# 4. Remove the calibration logic inside upload_mono (now upload_av)
content = re.sub(r'def upload_mono_file', 'def upload_av_file', content)
content = re.sub(r'        # Reuse an existing stereo calibration.*?dist = _DEFAULT_MONO_DIST\n', '        K, dist = _DEFAULT_MONO_K, _DEFAULT_MONO_DIST\n', content, flags=re.DOTALL)


with open("src/web/app.py", "w") as f:
    f.write(content)
