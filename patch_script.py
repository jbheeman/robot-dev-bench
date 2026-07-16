import re

with open("src/web/static/script.js", "r") as f:
    content = f.read()

# 1. Remove mode toggles and state
content = re.sub(r'    let fileRight = null;\n    let cameraMode = \'stereo\'; // \'stereo\' | \'mono\'\n', '', content)
content = re.sub(r'    const modeBtnStereo = document.*?const monoHint = document\.getElementById\(\'mono-hint\'\);\n\n', '', content, flags=re.DOTALL)
content = re.sub(r'    function updateRunButtonVisibility\(\) \{.*?\}\n\n', '', content, flags=re.DOTALL)
content = re.sub(r'    function setCameraMode\(newMode\) \{.*?\}\n\n', '', content, flags=re.DOTALL)
content = re.sub(r'    if \(modeBtnStereo && modeBtnMono\) \{.*?\}\n\n', '', content, flags=re.DOTALL)

# 2. Update drop zones
content = re.sub(r'setupDropZone\(\'drop-zone-right\'.*?\n', '', content)
content = content.replace('setupDropZone(\'drop-zone-left\', \'file-input-left\', \'browse-btn-left\', \'change-file-btn-left\', \'drop-content-success-left\', \'drop-content-default-left\', \'uploaded-filename-left\', false);', 
                          'setupDropZone(\'drop-zone-camera\', \'file-input-camera\', \'browse-btn-camera\', \'change-file-btn-camera\', \'drop-content-success-camera\', \'drop-content-default-camera\', \'uploaded-filename-camera\', false);')

# 3. Handle file selection
# Instead of replacing specific logic, let's just rewrite handleFileSelection
new_handle_file_selection = """    function handleFileSelection(file, isRight, successId, defaultId, filenameId, zoneId) {
        const lowerName = file.name.toLowerCase();
        if (!lowerName.endsWith('.mp4') && !lowerName.endsWith('.mov')) {
            alert("Only .mp4 and .mov files are supported.");
            return;
        }

        fileLeft = file;
        document.getElementById(zoneId).classList.add('has-file');
        document.getElementById(defaultId).classList.add('hidden');
        document.getElementById(successId).classList.remove('hidden');
        document.getElementById(filenameId).textContent = file.name;
        
        runAnalysisBtn.style.display = 'block';
    }"""
content = re.sub(r'    function handleFileSelection.*?updateRunButtonVisibility\(\);\n    \}', new_handle_file_selection, content, flags=re.DOTALL)

# 4. Form submission logic
new_submit_logic = """    runAnalysisBtn.addEventListener('click', async () => {
        if (!fileLeft) {
            alert("Please upload a camera feed.");
            return;
        }

        resultsSection.classList.add('hidden');
        loadingOverlay.classList.remove('hidden');

        try {
            const formData = new FormData();
            formData.append('camera', fileLeft);
            if (taskSelect) {
                formData.append('task', taskSelect.value);
            }

            const response = await fetch('/api/upload_av', {"""

content = re.sub(r'    runAnalysisBtn\.addEventListener\(\'click\', async \(\) => \{.*?const response = await fetch\(endpoint, \{', new_submit_logic, content, flags=re.DOTALL)

with open("src/web/static/script.js", "w") as f:
    f.write(content)
