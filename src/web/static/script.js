document.addEventListener('DOMContentLoaded', () => {
    const runAnalysisBtn = document.getElementById('run-analysis-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    const resultsSection = document.getElementById('results-section');
    const statusMsg = document.getElementById('upload-status-msg');
    const taskSelect = document.getElementById('task-select');

    // ── State ──
    let fileLeft = null;          // Upload mode file
    let recordedBlob = null;      // Record mode blob
    let activeMode = 'upload';    // 'upload' | 'record'

    // ── Stereo settings ──
    const stereoToggle = document.getElementById('stereo-toggle');
    const stereoFields = document.getElementById('stereo-fields');
    const stereoBadge = document.getElementById('stereo-badge');
    const stereoBaseline = document.getElementById('stereo-baseline');
    const stereoFocalLength = document.getElementById('stereo-focal-length');

    stereoToggle.addEventListener('change', () => {
        const on = stereoToggle.checked;
        stereoFields.classList.toggle('hidden', !on);
        stereoBadge.textContent = on ? 'ON' : 'OFF';
        stereoBadge.classList.toggle('stereo-badge-on', on);
    });

    const stereoSwap = document.getElementById('stereo-swap');
    
    // Hidden elements for L/R swapping
    const hiddenVideo = document.createElement('video');
    hiddenVideo.autoplay = true;
    hiddenVideo.playsInline = true;
    hiddenVideo.muted = true;
    hiddenVideo.style.display = 'none';
    document.body.appendChild(hiddenVideo);
    
    const swapCanvas = document.createElement('canvas');
    swapCanvas.style.display = 'none';
    document.body.appendChild(swapCanvas);
    const swapCtx = swapCanvas.getContext('2d');
    let composeAnimationFrame = null;
    let rawMediaStream = null;

    // ── Input Mode Tabs ──
    const tabUpload = document.getElementById('tab-upload');
    const tabRecord = document.getElementById('tab-record');
    const modeUpload = document.getElementById('mode-upload');
    const modeRecord = document.getElementById('mode-record');

    function switchMode(mode) {
        activeMode = mode;
        tabUpload.classList.toggle('active-tab', mode === 'upload');
        tabRecord.classList.toggle('active-tab', mode === 'record');
        modeUpload.classList.toggle('hidden', mode !== 'upload');
        modeRecord.classList.toggle('hidden', mode !== 'record');
        updateRunButton();
    }

    tabUpload.addEventListener('click', () => switchMode('upload'));
    tabRecord.addEventListener('click', () => switchMode('record'));

    function updateRunButton() {
        const hasFile = (activeMode === 'upload' && fileLeft) ||
                        (activeMode === 'record' && recordedBlob);
        runAnalysisBtn.style.display = hasFile ? 'block' : 'none';
    }

    // ── Drag & Drop (Upload Mode) ──
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
        if (!lowerName.endsWith('.mp4') && !lowerName.endsWith('.mov') && !lowerName.endsWith('.webm')) {
            alert("Only .mp4, .mov, and .webm files are supported.");
            return;
        }

        fileLeft = file;
        document.getElementById(zoneId).classList.add('has-file');
        document.getElementById(defaultId).classList.add('hidden');
        document.getElementById(successId).classList.remove('hidden');
        document.getElementById(filenameId).textContent = file.name;
        
        updateRunButton();
    }

    // ── Video Recording (Record Mode) ──
    const cameraPreview = document.getElementById('camera-preview');
    const startCameraBtn = document.getElementById('start-camera-btn');
    const startRecordBtn = document.getElementById('start-record-btn');
    const stopRecordBtn = document.getElementById('stop-record-btn');
    const recordOverlay = document.getElementById('record-overlay');
    const recordPulse = document.getElementById('record-pulse');
    const recordTimer = document.getElementById('record-timer');
    const recordStatus = document.getElementById('record-status');
    const recordStatusText = document.getElementById('record-status-text');
    const discardRecordBtn = document.getElementById('discard-record-btn');
    const noCameraMsg = document.getElementById('no-camera-msg');

    let mediaStream = null;
    let mediaRecorder = null;
    let recordedChunks = [];
    let recordingStartTime = null;
    let timerInterval = null;

    function stopCamera() {
        if (rawMediaStream) {
            rawMediaStream.getTracks().forEach(t => t.stop());
            rawMediaStream = null;
        }
        if (mediaStream) {
            mediaStream.getTracks().forEach(t => t.stop());
            mediaStream = null;
        }
        if (composeAnimationFrame) {
            cancelAnimationFrame(composeAnimationFrame);
            composeAnimationFrame = null;
        }
    }

    async function startCamera() {
        try {
            stopCamera();
            rawMediaStream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 2560 }, height: { ideal: 720 } },
                audio: false
            });
            
            if (stereoToggle.checked && stereoSwap.checked) {
                hiddenVideo.srcObject = rawMediaStream;
                await new Promise(r => hiddenVideo.onplaying = r);
                
                const w = hiddenVideo.videoWidth;
                const h = hiddenVideo.videoHeight;
                swapCanvas.width = w;
                swapCanvas.height = h;
                const halfW = w / 2;
                
                function draw() {
                    swapCtx.drawImage(hiddenVideo, halfW, 0, halfW, h, 0, 0, halfW, h); // Right to Left
                    swapCtx.drawImage(hiddenVideo, 0, 0, halfW, h, halfW, 0, halfW, h); // Left to Right
                    composeAnimationFrame = requestAnimationFrame(draw);
                }
                draw();
                mediaStream = swapCanvas.captureStream(30);
            } else {
                mediaStream = rawMediaStream;
            }
            
            cameraPreview.srcObject = mediaStream;
            noCameraMsg.classList.add('hidden');
            startCameraBtn.classList.add('hidden');
            startRecordBtn.classList.remove('hidden');
        } catch (err) {
            alert('Could not access camera: ' + err.message);
        }
    }

    startCameraBtn.addEventListener('click', startCamera);
    stereoSwap.addEventListener('change', () => {
        if (!startCameraBtn.classList.contains('hidden')) return; // not running
        startCamera();
    });

    startRecordBtn.addEventListener('click', () => {
        if (!mediaStream) return;
        recordedChunks = [];
        recordedBlob = null;
        recordStatus.classList.add('hidden');

        // Pick a supported MIME type
        const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
            ? 'video/webm;codecs=vp9'
            : MediaRecorder.isTypeSupported('video/webm')
                ? 'video/webm'
                : 'video/mp4';

        mediaRecorder = new MediaRecorder(mediaStream, { mimeType });
        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) recordedChunks.push(e.data);
        };
        mediaRecorder.onstop = () => {
            recordedBlob = new Blob(recordedChunks, { type: mimeType });
            clearInterval(timerInterval);
            recordOverlay.classList.add('hidden');
            recordPulse.classList.remove('pulse-active');
            startRecordBtn.classList.remove('hidden');
            stopRecordBtn.classList.add('hidden');

            // Show status
            const sizeMB = (recordedBlob.size / (1024 * 1024)).toFixed(1);
            recordStatusText.textContent = `Recording captured! (${sizeMB} MB)`;
            recordStatus.classList.remove('hidden');
            updateRunButton();
        };

        mediaRecorder.start(100); // collect data every 100ms
        recordingStartTime = Date.now();
        recordOverlay.classList.remove('hidden');
        recordPulse.classList.add('pulse-active');
        startRecordBtn.classList.add('hidden');
        stopRecordBtn.classList.remove('hidden');

        // Timer display
        timerInterval = setInterval(() => {
            const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
            const mins = String(Math.floor(elapsed / 60)).padStart(2, '0');
            const secs = String(elapsed % 60).padStart(2, '0');
            recordTimer.textContent = `${mins}:${secs}`;
        }, 500);
    });

    stopRecordBtn.addEventListener('click', () => {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
        }
    });

    discardRecordBtn.addEventListener('click', () => {
        recordedBlob = null;
        recordedChunks = [];
        recordStatus.classList.add('hidden');
        recordTimer.textContent = '00:00';
        updateRunButton();
    });
    
    // Clean up when switching tabs away from Record
    tabUpload.addEventListener('click', stopCamera);

    // ── Run Analysis ──
    runAnalysisBtn.addEventListener('click', async () => {
        let uploadFile = null;

        if (activeMode === 'upload') {
            if (!fileLeft) {
                alert("Please upload a camera feed.");
                return;
            }
            uploadFile = fileLeft;
        } else {
            if (!recordedBlob) {
                alert("Please record a video first.");
                return;
            }
            // Convert blob to a File object for the FormData
            const ext = recordedBlob.type.includes('mp4') ? 'mp4' : 'webm';
            uploadFile = new File([recordedBlob], `recording.${ext}`, { type: recordedBlob.type });
        }

        resultsSection.classList.add('hidden');
        loadingOverlay.classList.remove('hidden');

        try {
            const formData = new FormData();
            formData.append('camera', uploadFile);
            if (taskSelect) {
                formData.append('task', taskSelect.value);
            }
            const morphSelect = document.getElementById('morphology-select');
            if (morphSelect) {
                formData.append('morphology', morphSelect.value);
            }

            // Stereo settings
            if (stereoToggle.checked) {
                formData.append('stereo', 'true');
                formData.append('baseline', stereoBaseline.value);
                formData.append('focal_length', stereoFocalLength.value);
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

            // Update pipeline badge
            const pipelineBadge = document.getElementById('pipeline-badge');
            if (pipelineBadge) {
                if (resultData.stereo_used) {
                    pipelineBadge.textContent = 'Stereo-Fused';
                    pipelineBadge.classList.add('pipeline-badge-stereo');
                } else {
                    pipelineBadge.textContent = 'Monocular';
                    pipelineBadge.classList.remove('pipeline-badge-stereo');
                }
            }

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
                // Populate Walk Grade
                const walkScore = resultData.metrics.walk_grade;
                const walkScoreEl = document.getElementById('walk-final-score');
                const walkBadgeEl = document.getElementById('walk-tier-badge');
                if (walkScoreEl && walkScore !== undefined) {
                    walkScoreEl.textContent = parseFloat(walkScore).toFixed(1);
                    if (walkScore >= 90) {
                        walkBadgeEl.textContent = 'A';
                        walkBadgeEl.style.background = 'linear-gradient(135deg, #10b981, #3b82f6)';
                    } else if (walkScore >= 80) {
                        walkBadgeEl.textContent = 'B';
                        walkBadgeEl.style.background = 'linear-gradient(135deg, #3b82f6, #8b5cf6)';
                    } else if (walkScore >= 70) {
                        walkBadgeEl.textContent = 'C';
                        walkBadgeEl.style.background = 'linear-gradient(135deg, #f59e0b, #ef4444)';
                    } else {
                        walkBadgeEl.textContent = 'D';
                        walkBadgeEl.style.background = 'linear-gradient(135deg, #ef4444, #7f1d1d)';
                    }
                    walkBadgeEl.style.webkitBackgroundClip = 'text';
                }

                const metricMap = {
                    'metric-clearance': resultData.metrics.mean_clearance_cm,
                    'metric-stride': resultData.metrics.stride_length_m,
                    'metric-speed': resultData.metrics.speed_m_s,
                    'metric-oscillation': resultData.metrics.torso_oscillation_cm,
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
