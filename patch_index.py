import re

with open("src/web/static/index.html", "r") as f:
    content = f.read()

# 1. Remove the task selector's camera mode toggle
content = re.sub(r'<div class="task-selector-container">\s*<label class="task-label">Camera Mode:</label>.*?</div>\s*<p id="mono-hint" class="drop-desc hidden" style="margin: -0\.5rem 0 1rem;">.*?</p>', '', content, flags=re.DOTALL)

# 2. Remove the right dropzone
content = re.sub(r'<!-- Right Camera Drop Zone -->.*?</div>\s*</div>', '', content, flags=re.DOTALL)

# 3. Rename Camera 1 Feed to Camera Feed
content = content.replace('Camera 1 Feed', 'Camera Feed')
content = content.replace('primary camera', 'camera')
content = content.replace('Browse Camera 1', 'Browse Camera')
content = content.replace('Change Camera 1 File', 'Change Camera File')
content = content.replace('id="drop-zone-left"', 'id="drop-zone-camera"')
content = content.replace('id="file-input-left"', 'id="file-input-camera"')
content = content.replace('id="drop-content-default-left"', 'id="drop-content-default-camera"')
content = content.replace('id="browse-btn-left"', 'id="browse-btn-camera"')
content = content.replace('id="drop-content-success-left"', 'id="drop-content-success-camera"')
content = content.replace('id="uploaded-filename-left"', 'id="uploaded-filename-camera"')
content = content.replace('id="change-file-btn-left"', 'id="change-file-btn-camera"')

# 4. Remove calibration link if it exists
content = re.sub(r'<a href="/calibration.html".*?⚙ Calibrate Cameras</a>', '', content)

with open("src/web/static/index.html", "w") as f:
    f.write(content)
