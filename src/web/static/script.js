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

    // ── Pipeline mode (2D / 3D) ──
    const pipeline2dToggle = document.getElementById('pipeline-2d-toggle');
    const pipeline2dFields = document.getElementById('pipeline-2d-fields');
    const pipelineBadgeToggle = document.getElementById('pipeline-badge-toggle');
    const refLengthInput = document.getElementById('ref-length-cm');
    const cameraViewSelect = document.getElementById('camera-view-select');

    pipeline2dToggle.addEventListener('change', () => {
        const is2d = pipeline2dToggle.checked;
        pipeline2dFields.classList.toggle('hidden', !is2d);
        pipelineBadgeToggle.textContent = is2d ? '2D' : '3D';
        pipelineBadgeToggle.classList.toggle('stereo-badge-on', is2d);
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

            // Stereo settings
            if (stereoToggle.checked) {
                formData.append('stereo', 'true');
                formData.append('baseline', stereoBaseline.value);
                formData.append('focal_length', stereoFocalLength.value);
            }

            // Pipeline mode
            if (pipeline2dToggle.checked) {
                formData.append('pipeline_mode', '2d');
                formData.append('ref_length_cm', refLengthInput.value);
                formData.append('camera_view', cameraViewSelect.value);
            } else {
                formData.append('pipeline_mode', '3d');
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

                const metricMap = {
                    'metric-clearance': resultData.metrics.mean_clearance_cm,
                    'metric-stride': resultData.metrics.stride_length_m,
                    'metric-speed': resultData.metrics.speed_m_s,
                    'metric-oscillation': resultData.metrics.torso_oscillation_cm
                };
                
                // Update dynamic labels if front view
                const view = cameraViewSelect ? cameraViewSelect.value : 'side';
                const is2DFront = (resultData.pipeline_mode === '2d' && view === 'front');
                
                const labelStride = document.getElementById('label-stride');
                const labelSpeed = document.getElementById('label-speed');
                const labelOscillation = document.getElementById('label-oscillation');
                
                if (labelStride) labelStride.textContent = is2DFront ? 'Step Width (cm)' : 'Stride Length (m)';
                if (labelSpeed) labelSpeed.textContent = is2DFront ? 'Lateral Sway (cm)' : 'Walking Speed (m/s)';
                if (labelOscillation) labelOscillation.textContent = is2DFront ? 'Vertical Bounce (cm)' : 'Torso Oscillation (cm)';
                
                // Override metrics map for front view so we display the correct data in those slots
                if (is2DFront) {
                    metricMap['metric-stride'] = (resultData.metrics.stride_length_m * 100); // Now step width in cm
                    metricMap['metric-speed'] = resultData.metrics.lateral_sway_cm; // Now lateral sway in cm
                }
                
                // Hide all metrics by default
                document.querySelectorAll('.metric-item').forEach(el => el.style.display = 'none');

                const relevantMetrics = {
                    'walking': ['metric-clearance', 'metric-stride', 'metric-speed', 'metric-oscillation'],
                    'jumping': [],
                    'manipulation': [],
                    'reaching': [],
                    'transitions': [],
                    'general': Object.keys(metricMap) // show all for general
                };

                const taskType = resultData.task || 'general';
                const toShow = relevantMetrics[taskType] || relevantMetrics['general'];

                for (const [id, value] of Object.entries(metricMap)) {
                    const el = document.getElementById(id);
                    if (el && toShow.includes(id)) {
                        el.parentElement.style.display = 'block';
                        el.textContent = (value !== undefined && value !== null) ? parseFloat(value).toFixed(2) : '0.00';
                    }
                }
            }
            
            // Show/hide the correct viewer based on pipeline mode
            const viewer2d = document.getElementById('viewer-2d-container');
            const viewer3d = document.getElementById('viewer-3d-container');

            if (resultData.pipeline_mode === '2d') {
                viewer2d.style.display = 'flex';
                viewer3d.style.display = 'none';
            } else {
                viewer2d.style.display = 'flex'; // ALWAYS show 2D side-by-side if data is available
                viewer3d.style.display = 'flex';

                // Load 3D Playback Data
                if (resultData.poses_3d && window.loadPlaybackData) {
                    window.loadPlaybackData(resultData.poses_3d, resultData.valid_mask);
                }
            }

            // Always load 2D overlay if we have the data, regardless of 3D mode
            if (resultData.keypoints_2d && window.load2DOverlay) {
                // Pass the uploaded file as the video source
                window.load2DOverlay(
                    uploadFile,
                    resultData.keypoints_2d,
                    resultData.confidence_2d,
                    resultData.frame_width,
                    resultData.frame_height,
                    resultData.fps || 30,
                    resultData.stereo_used
                );
            }

            loadingOverlay.classList.add('hidden');
            resultsSection.classList.remove('hidden');

        } catch (error) {
            console.error('Error running analysis:', error);
            alert('Failed to upload video files: ' + error.message);
            loadingOverlay.classList.add('hidden');
        }
    });

    // ── 2D Overlay Renderer ──
    const COCO_SKELETON_2D = [
        [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
        [5, 11], [6, 12], [5, 6],
        [5, 7], [7, 9], [6, 8], [8, 10],
        [0, 5], [0, 6],
    ];
    const JOINT_COLORS_2D = [
        '#ef4444', '#f97316', '#eab308', '#22c55e', '#3b82f6',
        '#8b5cf6', '#ec4899', '#14b8a6', '#f59e0b', '#6366f1',
        '#10b981', '#0ea5e9', '#a855f7', '#f43f5e', '#06b6d4',
        '#84cc16', '#d946ef'
    ];

    let overlay2dData = null;
    let overlay2dPlaying = false;
    let overlay2dFrame = 0;
    let overlay2dAnimId = null;

    window.load2DOverlay = function(videoFile, keypoints, confidence, frameW, frameH, fps, isStereo) {
        overlay2dData = { keypoints, confidence, frameW, frameH, fps, isStereo };
        overlay2dFrame = 0;
        overlay2dPlaying = false;

        const video = document.getElementById('overlay-video');
        const canvas = document.getElementById('pose-canvas');
        const timeline = document.getElementById('timeline-2d');

        timeline.max = keypoints.length - 1;
        timeline.value = 0;

        video.addEventListener('loadedmetadata', () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            drawPoseFrame(0);
        }, { once: true });

        // Load video after attaching listener to prevent race conditions
        const url = URL.createObjectURL(videoFile);
        video.src = url;
        video.load();

        // Play / Pause / Scrub
        document.getElementById('play-2d-btn').onclick = () => {
            const playPromise = video.play();
            if (playPromise !== undefined) {
                playPromise.catch(e => console.warn("Video play failed (codec issue?):", e));
            }
            overlay2dPlaying = true;
            window.last2DTime = Date.now();
            animate2DOverlay();
        };
        document.getElementById('pause-2d-btn').onclick = () => {
            video.pause();
            overlay2dPlaying = false;
            if (overlay2dAnimId) cancelAnimationFrame(overlay2dAnimId);
        };
        timeline.addEventListener('input', (e) => {
            const f = parseInt(e.target.value);
            overlay2dFrame = f;
            window.virtualTime2D = f / overlay2dData.fps;
            // Seek the video to the matching time
            if (overlay2dData) {
                video.currentTime = f / overlay2dData.fps;
            }
            drawPoseFrame(f);
        });

        // Sync overlay to video time
        video.addEventListener('timeupdate', () => {
            if (!overlay2dData) return;
            let f = 0;
            if (video.duration && video.duration > 0 && isFinite(video.duration)) {
                f = Math.round((video.currentTime / video.duration) * (overlay2dData.keypoints.length - 1));
            } else {
                f = Math.round(video.currentTime * overlay2dData.fps);
            }
            if (f >= 0 && f < overlay2dData.keypoints.length) {
                overlay2dFrame = f;
                timeline.value = f;
                drawPoseFrame(f);
            }
        });
    };

    function animate2DOverlay() {
        if (!overlay2dPlaying) return;
        
        if (overlay2dData) {
            const video = document.getElementById('overlay-video');
            const timeline = document.getElementById('timeline-2d');
            if (video) {
                let f = 0;
                if (video.videoWidth > 0 && !video.error) {
                    // Browser can decode video natively
                    if (video.duration && video.duration > 0 && isFinite(video.duration)) {
                        f = Math.round((video.currentTime / video.duration) * (overlay2dData.keypoints.length - 1));
                    } else {
                        f = Math.round(video.currentTime * overlay2dData.fps);
                    }
                } else {
                    // Fallback for unsupported codecs (HEVC etc.)
                    if (window.virtualTime2D === undefined) window.virtualTime2D = 0;
                    const now = Date.now();
                    if (!window.last2DTime) window.last2DTime = now;
                    const dt = (now - window.last2DTime) / 1000.0;
                    window.last2DTime = now;
                    
                    window.virtualTime2D += dt;
                    f = Math.round(window.virtualTime2D * overlay2dData.fps);
                    if (f >= overlay2dData.keypoints.length) {
                        window.virtualTime2D = 0;
                        f = 0;
                    }
                }
                if (f >= 0 && f < overlay2dData.keypoints.length && f !== overlay2dFrame) {
                    overlay2dFrame = f;
                    if (timeline) timeline.value = f;
                    drawPoseFrame(f);
                }
            }
        }
        
        overlay2dAnimId = requestAnimationFrame(animate2DOverlay);
    }

    function drawPoseFrame(frameIdx) {
        if (!overlay2dData) return;
        const canvas = document.getElementById('pose-canvas');
        const video = document.getElementById('overlay-video');
        if (!canvas || !video) return;

        // Ensure canvas dimensions match video dimensions on every frame in case loadedmetadata fired early
        if (video.videoWidth > 0 && video.videoHeight > 0) {
            if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
            }
        } else if (canvas.width === 0 && overlay2dData.frameW > 0) {
            // Fallback: If video cannot be decoded by browser (e.g. HEVC), videoWidth is 0.
            // Size the canvas to OpenCV's extracted dimensions so the skeleton still draws.
            canvas.width = overlay2dData.frameW;
            canvas.height = overlay2dData.frameH;
        }
        
        // Prevent drawing if dimensions are invalid
        if (canvas.width === 0 || canvas.height === 0 || !overlay2dData.frameW) return;

        const ctx = canvas.getContext('2d');

        // Scale from original video coords to canvas coords
        // If the original video is stereo, the keypoints were extracted from the left half.
        // So the source width corresponds to actual canvas.width / 2.
        const actualFrameW = overlay2dData.isStereo ? (overlay2dData.frameW * 2) : overlay2dData.frameW;
        const scaleX = canvas.width / actualFrameW;
        const scaleY = canvas.height / overlay2dData.frameH;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const kp = overlay2dData.keypoints[frameIdx];
        const conf = overlay2dData.confidence[frameIdx];
        if (!kp) return;

        const CONF_THRESH = 0.3;

        // Draw bones
        ctx.lineWidth = 6;
        for (const [u, v] of COCO_SKELETON_2D) {
            if (conf[u] >= CONF_THRESH && conf[v] >= CONF_THRESH) {
                ctx.strokeStyle = '#39ff14'; // Neon green
                ctx.beginPath();
                ctx.moveTo(kp[u][0] * scaleX, kp[u][1] * scaleY);
                ctx.lineTo(kp[v][0] * scaleX, kp[v][1] * scaleY);
                ctx.stroke();
            }
        }

        // Draw joints
        for (let j = 0; j < kp.length; j++) {
            if (conf[j] >= CONF_THRESH) {
                ctx.fillStyle = '#00ffff'; // Cyan
                ctx.beginPath();
                ctx.arc(kp[j][0] * scaleX, kp[j][1] * scaleY, 8, 0, Math.PI * 2);
                ctx.fill();
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 3;
                ctx.stroke();
            }
        }
    }
});
