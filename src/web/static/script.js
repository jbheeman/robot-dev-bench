document.addEventListener('DOMContentLoaded', () => {
    const runAnalysisBtn = document.getElementById('run-analysis-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    const resultsSection = document.getElementById('results-section');
    const statusMsg = document.getElementById('upload-status-msg');
    const taskSelect = document.getElementById('task-select');

    let fileLeft = null;

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

    setupDropZone('drop-zone-camera', 'file-input-camera', 'browse-btn-camera', 'change-file-btn-camera', 'drop-content-success-camera', 'drop-content-default-camera', 'uploaded-filename-camera', false);
    

    function handleFileSelection(file, isRight, successId, defaultId, filenameId, zoneId) {
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
    }

    runAnalysisBtn.addEventListener('click', async () => {
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

            const response = await fetch('/api/upload_av', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Network response was not ok');
            const data = await response.json();
            
            if (!data.job_id) {
                throw new Error('No job ID returned from server');
            }

            // Start polling
            const jobId = data.job_id;
            const progressFill = document.getElementById('progress-fill');
            const progressText = document.getElementById('progress-text');
            const statusMsg = document.getElementById('upload-status-msg');

            let jobResult = null;
            while (true) {
                await new Promise(r => setTimeout(r, 500));
                const statusRes = await fetch(`/api/job_status/${jobId}`);
                if (!statusRes.ok) throw new Error('Failed to fetch job status');
                
                const statusData = await statusRes.json();
                
                if (progressFill && progressText && statusMsg) {
                    const pct = Math.round(statusData.progress * 100);
                    progressFill.style.width = `${pct}%`;
                    progressText.textContent = `${pct}%`;
                    statusMsg.textContent = statusData.message || 'Processing...';
                }

                if (statusData.status === 'success') {
                    jobResult = statusData.result;
                    break;
                } else if (statusData.status === 'error') {
                    throw new Error(statusData.error || 'Job failed on server');
                }
            }
            
            if (!jobResult) throw new Error('No result returned');
            const resultData = jobResult;

            // Update Classification Tier
            const tierBadge = document.getElementById('tier-badge');
            const finalScore = document.getElementById('final-score');
            
            if (resultData.classification && tierBadge && finalScore) {
                tierBadge.textContent = resultData.classification.tier;
                finalScore.textContent = parseFloat(resultData.classification.score).toFixed(2);
                
                // Colorize badge based on tier
                if (resultData.classification.tier === 'Superhuman/Industrial') {
                    tierBadge.style.background = 'linear-gradient(135deg, #f59e0b, #ef4444)';
                    tierBadge.style.webkitBackgroundClip = 'text';
                } else if (resultData.classification.tier === 'Research') {
                    tierBadge.style.background = 'linear-gradient(135deg, #3b82f6, #8b5cf6)';
                    tierBadge.style.webkitBackgroundClip = 'text';
                } else {
                    tierBadge.style.background = 'linear-gradient(135deg, #6b7280, #9ca3af)';
                    tierBadge.style.webkitBackgroundClip = 'text';
                }
            }

            // Update Metrics Grid
            if (resultData.metrics) {
                const metricMap = {
                    'metric-ldlj': resultData.metrics.smoothness_ldlj,
                    'metric-sparc': resultData.metrics.smoothness_sparc,
                    'metric-symmetry': resultData.metrics.symmetry,
                    'metric-periodicity': resultData.metrics.periodicity,
                    'metric-rom': resultData.metrics.rom_utilisation,
                    'metric-flight': resultData.metrics.flight_time,
                    'metric-accel': resultData.metrics.peak_z_accel,
                    'metric-jerk': resultData.metrics.landing_jerk,
                    'metric-com': resultData.metrics.com_oscillation,
                    'metric-transition': resultData.metrics.transition_time
                };
                
                for (const [id, value] of Object.entries(metricMap)) {
                    const el = document.getElementById(id);
                    if (el) {
                        el.textContent = (value !== undefined && value !== null) ? parseFloat(value).toFixed(2) : '0.00';
                    }
                }
            }
            
            // Load 3D Playback Data
            if (resultData.poses_3d && window.loadPlaybackData) {
                window.loadPlaybackData(resultData.poses_3d, resultData.valid_mask);
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
