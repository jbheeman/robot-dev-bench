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

    // ── Task View Lock ──
    const cameraViewSelect = document.getElementById('camera-view-select');
    
    function updateCameraViewLock() {
        if (!cameraViewSelect) return;
        const task = taskSelect.value;
        if (task === 'walking') {
            cameraViewSelect.value = 'side';
            cameraViewSelect.disabled = true;
        } else if (task === 'jumping' || task === 'manipulation') {
            cameraViewSelect.value = 'front';
            cameraViewSelect.disabled = true;
        } else {
            cameraViewSelect.disabled = false;
        }
    }
    
    if (taskSelect) {
        taskSelect.addEventListener('change', updateCameraViewLock);
        updateCameraViewLock();
    }

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
            
            mediaStream = rawMediaStream;
            
            cameraPreview.srcObject = mediaStream;
            noCameraMsg.classList.add('hidden');
            startCameraBtn.classList.add('hidden');
            startRecordBtn.classList.remove('hidden');
        } catch (err) {
            alert('Could not access camera: ' + err.message);
        }
    }

    startCameraBtn.addEventListener('click', startCamera);

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
    // ── Bounding Box State ──
    let bboxCoords = null; // {x, y, w, h}
    let isDrawingBBox = false;
    let startX, startY;
    let currentUploadFile = null;

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
            const ext = recordedBlob.type.includes('mp4') ? 'mp4' : 'webm';
            uploadFile = new File([recordedBlob], `recording.${ext}`, { type: recordedBlob.type });
        }

        currentUploadFile = uploadFile;
        
        if (taskSelect && taskSelect.value === 'manipulation') {
            showBBoxModal(uploadFile);
        } else {
            executeAnalysis(uploadFile, null);
        }
    });

    function showBBoxModal(file) {
        const modal = document.getElementById('bbox-modal');
        const canvas = document.getElementById('bbox-canvas');
        const ctx = canvas.getContext('2d');
        bboxCoords = null;

        // Show loading state temporarily
        modal.classList.remove('hidden');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = 'white';
        ctx.font = '16px Arial';
        ctx.fillText("Loading frame for bounding box...", 10, 30);

        const formData = new FormData();
        formData.append('file', file);
        
        fetch('/api/thumbnail', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) throw new Error("Could not load thumbnail");
            return response.blob();
        })
        .then(blob => {
            const img = new Image();
            img.onload = () => {
                // Setup canvas size to max 800px width while keeping aspect ratio
                const maxW = 800;
                const maxH = 600;
                let drawW = img.width;
                let drawH = img.height;
                
                if (drawW > maxW) {
                    drawH = (maxW / drawW) * drawH;
                    drawW = maxW;
                }
                if (drawH > maxH) {
                    drawW = (maxH / drawH) * drawW;
                    drawH = maxH;
                }
                
                canvas.width = drawW;
                canvas.height = drawH;
                
                // Draw first frame
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            
            // Setup drawing listeners
            let imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            
            canvas.onmousedown = (e) => {
                isDrawingBBox = true;
                const rect = canvas.getBoundingClientRect();
                startX = e.clientX - rect.left;
                startY = e.clientY - rect.top;
            };
            
            canvas.onmousemove = (e) => {
                const rect = canvas.getBoundingClientRect();
                const curX = e.clientX - rect.left;
                const curY = e.clientY - rect.top;
                
                ctx.putImageData(imgData, 0, 0); // Restore frame
                
                // Draw crosshairs
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)';
                ctx.lineWidth = 1;
                ctx.setLineDash([5, 5]);
                ctx.beginPath();
                ctx.moveTo(0, curY);
                ctx.lineTo(canvas.width, curY);
                ctx.moveTo(curX, 0);
                ctx.lineTo(curX, canvas.height);
                ctx.stroke();
                ctx.setLineDash([]); // Reset line dash
                
                if (isDrawingBBox) {
                    ctx.strokeStyle = '#00ffff';
                    ctx.lineWidth = 2;
                    ctx.strokeRect(startX, startY, curX - startX, curY - startY);
                } else if (bboxCoords) {
                    const scaleX = canvas.width / img.width;
                    const scaleY = canvas.height / img.height;
                    ctx.strokeStyle = '#00ffff';
                    ctx.lineWidth = 2;
                    ctx.strokeRect(bboxCoords.x * scaleX, bboxCoords.y * scaleY, bboxCoords.w * scaleX, bboxCoords.h * scaleY);
                }
            };
            
            canvas.onmouseleave = () => {
                ctx.putImageData(imgData, 0, 0); // Restore frame
                if (bboxCoords && !isDrawingBBox) {
                    const scaleX = canvas.width / img.width;
                    const scaleY = canvas.height / img.height;
                    ctx.strokeStyle = '#00ffff';
                    ctx.lineWidth = 2;
                    ctx.strokeRect(bboxCoords.x * scaleX, bboxCoords.y * scaleY, bboxCoords.w * scaleX, bboxCoords.h * scaleY);
                }
                isDrawingBBox = false;
            };
            
            canvas.onmouseup = (e) => {
                isDrawingBBox = false;
                const rect = canvas.getBoundingClientRect();
                const endX = e.clientX - rect.left;
                const endY = e.clientY - rect.top;
                
                const x = Math.min(startX, endX);
                const y = Math.min(startY, endY);
                const w = Math.abs(endX - startX);
                const h = Math.abs(endY - startY);
                
                if (w > 20 && h > 20) {
                    // Map back to original image dimensions
                    const scaleX = img.width / canvas.width;
                    const scaleY = img.height / canvas.height;
                    bboxCoords = {
                        x: Math.round(x * scaleX),
                        y: Math.round(y * scaleY),
                        w: Math.round(w * scaleX),
                        h: Math.round(h * scaleY)
                    };
                    canvas.onmouseleave(); // Trigger a render of the completed box
                } else {
                    bboxCoords = null; // Too small
                    ctx.putImageData(imgData, 0, 0); // Restore frame
                }
            };
        }; // Close img.onload
        img.src = URL.createObjectURL(blob);
    })
    .catch(err => {
        console.error("Thumbnail error:", err);
        modal.classList.add('hidden');
        // If it fails, just skip bbox and run analysis directly
        executeAnalysis(file, null);
    });
}

    document.getElementById('bbox-skip-btn').addEventListener('click', () => {
        document.getElementById('bbox-modal').classList.add('hidden');
        executeAnalysis(currentUploadFile, null);
    });

    document.getElementById('bbox-confirm-btn').addEventListener('click', () => {
        document.getElementById('bbox-modal').classList.add('hidden');
        executeAnalysis(currentUploadFile, bboxCoords);
    });

    async function executeAnalysis(uploadFile, bbox) {
        resultsSection.classList.add('hidden');
        loadingOverlay.classList.remove('hidden');

        try {
            const formData = new FormData();
            formData.append('camera', uploadFile);
            if (bbox) {
                formData.append('crop_x', bbox.x);
                formData.append('crop_y', bbox.y);
                formData.append('crop_w', bbox.w);
                formData.append('crop_h', bbox.h);
            }
            if (taskSelect) {
                formData.append('task', taskSelect.value);
            }
            const refLengthInput = document.getElementById('ref-length-cm');
            const cameraViewSelect = document.getElementById('camera-view-select');
            
            formData.append('ref_length_cm', refLengthInput.value);
            formData.append('camera_view', cameraViewSelect.value);

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
                if (resultData.classification.tier === 'Adult') {
                    tierBadge.style.background = 'linear-gradient(135deg, #f59e0b, #ef4444)';
                    tierBadge.style.webkitBackgroundClip = 'text';
                } else if (resultData.classification.tier === 'Adolescent') {
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
                    'metric-oscillation': resultData.metrics.torso_oscillation_cm,
                    'metric-flight-time': resultData.metrics.flight_time_s,
                    'metric-peak-z': resultData.metrics.peak_z_accel_g,
                    'metric-landing-jerk': resultData.metrics.landing_jerk,
                    'metric-wrist-jerk': resultData.metrics.wrist_jerk,
                    'metric-wrist-dist': resultData.metrics.wrist_to_block_min_dist_cm,
                    'metric-task-duration': resultData.metrics.task_duration_s,
                    'metric-path-efficiency': resultData.metrics.block_path_efficiency
                };

                const keyMap = {
                    'metric-clearance': 'mean_clearance_cm',
                    'metric-stride': 'stride_length_m',
                    'metric-speed': 'speed_m_s',
                    'metric-oscillation': 'torso_oscillation_cm',
                    'metric-flight-time': 'flight_time_s',
                    'metric-peak-z': 'peak_z_accel_g',
                    'metric-landing-jerk': 'landing_jerk',
                    'metric-wrist-jerk': 'wrist_jerk',
                    'metric-wrist-dist': 'wrist_to_block_min_dist_cm',
                    'metric-task-duration': 'task_duration_s',
                    'metric-path-efficiency': 'block_path_efficiency'
                };
                
                // Update dynamic labels if front view
                const view = cameraViewSelect ? cameraViewSelect.value : 'side';
                const is2DFront = (view === 'front');
                
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
                    'jumping': ['metric-flight-time', 'metric-peak-z', 'metric-landing-jerk'],

                    'manipulation': ['metric-wrist-jerk', 'metric-wrist-dist', 'metric-task-duration', 'metric-path-efficiency'],
                    'general': Object.keys(metricMap) // show all for general
                };

                const taskType = resultData.task || 'general';
                const toShow = relevantMetrics[taskType] || relevantMetrics['general'];
                const contributions = resultData.classification ? resultData.classification.contributions : null;

                for (const [id, value] of Object.entries(metricMap)) {
                    const el = document.getElementById(id);
                    if (el && toShow.includes(id)) {
                        el.parentElement.style.display = 'block';
                        let text = (value !== undefined && value !== null) ? parseFloat(value).toFixed(2) : '0.00';
                        
                        if (contributions && keyMap[id]) {
                            const contrib = contributions[keyMap[id]];
                            if (contrib !== undefined) {
                                text += ` <span style="font-size: 0.75rem; color: #10b981; margin-left: 0.5rem; background: rgba(16,185,129,0.1); padding: 0.1rem 0.3rem; border-radius: 4px; vertical-align: middle;">+${parseFloat(contrib).toFixed(2)} pts</span>`;
                            }
                        }
                        
                        el.innerHTML = text;
                    }
                }
            }
            
            // Show the viewer
            const viewer2d = document.getElementById('viewer-2d-container');
            if (viewer2d) {
                viewer2d.style.display = 'flex';
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
                    resultData.stereo_used,
                    resultData.video_url,
                    resultData.objects_2d
                );
            }

            loadingOverlay.classList.add('hidden');
            resultsSection.classList.remove('hidden');

        } catch (error) {
            console.error('Error running analysis:', error);
            alert('Failed to upload video files: ' + error.message);
            loadingOverlay.classList.add('hidden');
        }
    }

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
    window.load2DOverlay = function(videoFile, keypoints, confidence, frameW, frameH, fps, isStereo, transcodedUrl, objects2d) {
        overlay2dData = { keypoints, confidence, frameW, frameH, fps, isStereo, objects: objects2d };
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

        // Load transcoded video if available (fixes HEVC/H.265 playback in browser)
        if (transcodedUrl) {
            video.src = transcodedUrl;
        } else {
            video.src = URL.createObjectURL(videoFile);
        }
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
        const actualFrameW = overlay2dData.frameW;
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
        
        // Draw tracked objects
        if (overlay2dData.objects) {
            const redBlock = overlay2dData.objects.red_block;
            if (redBlock && redBlock[frameIdx]) {
                const [bx, by] = redBlock[frameIdx];
                if (!isNaN(bx) && !isNaN(by)) {
                    ctx.fillStyle = '#ff0000'; // Red
                    ctx.beginPath();
                    ctx.arc(bx * scaleX, by * scaleY, 12, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.strokeStyle = '#ffffff';
                    ctx.lineWidth = 3;
                    ctx.stroke();
                }
            }
            
            const whiteBlock = overlay2dData.objects.white_block;
            if (whiteBlock && whiteBlock[frameIdx]) {
                const [bx, by] = whiteBlock[frameIdx];
                if (!isNaN(bx) && !isNaN(by)) {
                    ctx.fillStyle = '#ffffff'; // White
                    ctx.beginPath();
                    ctx.arc(bx * scaleX, by * scaleY, 12, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.strokeStyle = '#000000'; // Black border for visibility
                    ctx.lineWidth = 3;
                    ctx.stroke();
                }
            }
        }
    }
});
