document.addEventListener('DOMContentLoaded', () => {
    const runAnalysisBtn = document.getElementById('run-analysis-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    const resultsSection = document.getElementById('results-section');
    const statusMsg = document.getElementById('upload-status-msg');
    const taskSelect = document.getElementById('task-select');

    let fileLeft = null;
    let fileRight = null;
    let cameraMode = 'stereo'; // 'stereo' | 'mono'

    const modeBtnStereo = document.getElementById('mode-btn-stereo');
    const modeBtnMono = document.getElementById('mode-btn-mono');
    const dropZoneRightContainer = document.getElementById('drop-zone-right-container');
    const monoHint = document.getElementById('mono-hint');

    function updateRunButtonVisibility() {
        const ready = cameraMode === 'stereo' ? (fileLeft && fileRight) : !!fileLeft;
        runAnalysisBtn.style.display = ready ? 'block' : 'none';
    }

    function setCameraMode(newMode) {
        cameraMode = newMode;
        const isStereo = cameraMode === 'stereo';

        modeBtnStereo.classList.toggle('active-mode', isStereo);
        modeBtnMono.classList.toggle('active-mode', !isStereo);
        dropZoneRightContainer.classList.toggle('hidden', !isStereo);
        monoHint.classList.toggle('hidden', isStereo);

        if (isStereo) {
            runAnalysisBtn.textContent = 'Run AV Analysis';
        } else {
            runAnalysisBtn.textContent = 'Run Single-Camera Analysis';
            fileRight = null;
        }
        updateRunButtonVisibility();
    }

    if (modeBtnStereo && modeBtnMono) {
        modeBtnStereo.addEventListener('click', () => setCameraMode('stereo'));
        modeBtnMono.addEventListener('click', () => setCameraMode('mono'));
    }

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function setupDropZone(zoneId, inputId, browseBtnId, changeBtnId, successId, defaultId, filenameId, isRight) {
        const zone = document.getElementById(zoneId);
        const input = document.getElementById(inputId);
        const browseBtn = document.getElementById(browseBtnId);
        const changeBtn = document.getElementById(changeBtnId);
        
        if (!zone) return;

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            zone.addEventListener(eventName, preventDefaults, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            zone.addEventListener(eventName, () => zone.classList.add('active-drop'), false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            zone.addEventListener(eventName, () => zone.classList.remove('active-drop'), false);
        });

        zone.addEventListener('drop', (e) => {
            if (e.dataTransfer.files.length > 0) handleFileSelection(e.dataTransfer.files[0], isRight, successId, defaultId, filenameId, zoneId);
        }, false);
        
        browseBtn.addEventListener('click', () => input.click());
        if (changeBtn) changeBtn.addEventListener('click', () => input.click());
        
        input.addEventListener('change', function() {
            if (this.files.length > 0) handleFileSelection(this.files[0], isRight, successId, defaultId, filenameId, zoneId);
        });
    }

    setupDropZone('drop-zone-left', 'file-input-left', 'browse-btn-left', 'change-file-btn-left', 'drop-content-success-left', 'drop-content-default-left', 'uploaded-filename-left', false);
    setupDropZone('drop-zone-right', 'file-input-right', 'browse-btn-right', 'change-file-btn-right', 'drop-content-success-right', 'drop-content-default-right', 'uploaded-filename-right', true);


    function handleFileSelection(file, isRight, successId, defaultId, filenameId, zoneId) {
        const lowerName = file.name.toLowerCase();
        if (!lowerName.endsWith('.mp4') && !lowerName.endsWith('.mov')) {
            alert("Only .mp4 and .mov files are supported.");
            return;
        }

        if (isRight) {
            fileRight = file;
            document.getElementById(zoneId).classList.add('has-file-b');
        } else {
            fileLeft = file;
            document.getElementById(zoneId).classList.add('has-file');
        }
        document.getElementById(defaultId).classList.add('hidden');
        document.getElementById(successId).classList.remove('hidden');
        document.getElementById(filenameId).textContent = file.name;

        updateRunButtonVisibility();
    }

    runAnalysisBtn.addEventListener('click', async () => {
        if (cameraMode === 'stereo' && (!fileLeft || !fileRight)) {
            alert("Please upload both Camera 1 and Camera 2 feeds.");
            return;
        }
        if (cameraMode === 'mono' && !fileLeft) {
            alert("Please upload a camera feed.");
            return;
        }

        resultsSection.classList.add('hidden');
        loadingOverlay.classList.remove('hidden');

        try {
            const formData = new FormData();
            let endpoint;
            if (cameraMode === 'stereo') {
                formData.append('left_camera', fileLeft);
                formData.append('right_camera', fileRight);
                endpoint = '/api/upload_av';
            } else {
                formData.append('camera', fileLeft);
                endpoint = '/api/upload_mono';
            }
            if (taskSelect) {
                formData.append('task', taskSelect.value);
            }

            const response = await fetch(endpoint, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Network response was not ok');
            const data = await response.json();
            
            if (data.status !== 'success') {
                throw new Error(data.message || 'Upload failed');
            }

            // Update Classification Tier
            const tierBadge = document.getElementById('tier-badge');
            const finalScore = document.getElementById('final-score');
            
            if (data.classification && tierBadge && finalScore) {
                tierBadge.textContent = data.classification.tier;
                finalScore.textContent = parseFloat(data.classification.score).toFixed(2);
                
                // Colorize badge based on tier
                if (data.classification.tier === 'Superhuman/Industrial') {
                    tierBadge.style.background = 'linear-gradient(135deg, #f59e0b, #ef4444)';
                    tierBadge.style.webkitBackgroundClip = 'text';
                } else if (data.classification.tier === 'Research') {
                    tierBadge.style.background = 'linear-gradient(135deg, #3b82f6, #8b5cf6)';
                    tierBadge.style.webkitBackgroundClip = 'text';
                } else {
                    tierBadge.style.background = 'linear-gradient(135deg, #6b7280, #9ca3af)';
                    tierBadge.style.webkitBackgroundClip = 'text';
                }
            }

            // Update Metrics Grid
            if (data.metrics) {
                const metricMap = {
                    'metric-ldlj': data.metrics.smoothness_ldlj,
                    'metric-sparc': data.metrics.smoothness_sparc,
                    'metric-symmetry': data.metrics.symmetry,
                    'metric-periodicity': data.metrics.periodicity,
                    'metric-rom': data.metrics.rom_utilisation,
                    'metric-flight': data.metrics.flight_time,
                    'metric-accel': data.metrics.peak_z_accel,
                    'metric-jerk': data.metrics.landing_jerk,
                    'metric-com': data.metrics.com_oscillation,
                    'metric-transition': data.metrics.transition_time
                };
                
                for (const [id, value] of Object.entries(metricMap)) {
                    const el = document.getElementById(id);
                    if (el) {
                        el.textContent = (value !== undefined && value !== null) ? parseFloat(value).toFixed(2) : '0.00';
                    }
                }
            }
            
            // Load 3D Playback Data
            if (data.poses_3d && window.loadPlaybackData) {
                window.loadPlaybackData(data.poses_3d, data.valid_mask);
            }

            loadingOverlay.classList.add('hidden');
            resultsSection.classList.remove('hidden');

        } catch (error) {
            console.error('Error running analysis:', error);
            alert('Failed to upload video files: ' + error.message);
            loadingOverlay.classList.add('hidden');
        }
    });
});
