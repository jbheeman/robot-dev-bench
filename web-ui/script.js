const videoUpload = document.getElementById('video-upload');
const fileName = document.getElementById('file-name');
const videoPreview = document.getElementById('video-preview');
const previewContainer = document.getElementById('preview-container');
const canvas = document.getElementById('roi-canvas');
const ctx = canvas.getContext('2d');
const submitBtn = document.getElementById('submit-btn');
const clearRoiBtn = document.getElementById('clear-roi');
const bpsForm = document.getElementById('bps-form');
const loadingOverlay = document.getElementById('loading-overlay');
const resultsPanel = document.getElementById('results-panel');

const playPauseBtn = document.getElementById('play-pause-btn');
const videoTimeline = document.getElementById('video-timeline');
const videoTimeDisplay = document.getElementById('video-time-display');
const btnSetStart = document.getElementById('btn-set-start');
const btnSetEnd = document.getElementById('btn-set-end');
const startTimeInput = document.getElementById('start-time');
const endTimeInput = document.getElementById('end-time');

let roi = null;
let isDrawing = false;
let startX, startY;

// Handle file selection
videoUpload.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        fileName.textContent = file.name;
        const fileUrl = URL.createObjectURL(file);
        videoPreview.src = fileUrl;
        
        videoPreview.onloadedmetadata = () => {
            previewContainer.style.display = 'block';
            submitBtn.disabled = false;
            
            // Set up timeline
            videoTimeline.max = videoPreview.duration;
            updateTimeDisplay();

            // Set canvas size to match video display size
            setTimeout(resizeCanvas, 100);
        };
    }
});

// Video Controls Logic
playPauseBtn.addEventListener('click', () => {
    if (videoPreview.paused) {
        videoPreview.play();
        playPauseBtn.textContent = '⏸ Pause';
    } else {
        videoPreview.pause();
        playPauseBtn.textContent = '▶ Play';
    }
});

videoPreview.addEventListener('timeupdate', () => {
    if (!videoTimeline.matches(':active')) {
        videoTimeline.value = videoPreview.currentTime;
    }
    updateTimeDisplay();
});

videoTimeline.addEventListener('input', () => {
    videoPreview.currentTime = videoTimeline.value;
    updateTimeDisplay();
});

function updateTimeDisplay() {
    const current = videoPreview.currentTime.toFixed(1);
    const total = (videoPreview.duration || 0).toFixed(1);
    videoTimeDisplay.textContent = `${current} / ${total}`;
}

btnSetStart.addEventListener('click', () => {
    startTimeInput.value = videoPreview.currentTime.toFixed(1);
});

btnSetEnd.addEventListener('click', () => {
    endTimeInput.value = videoPreview.currentTime.toFixed(1);
});

function resizeCanvas() {
    canvas.width = videoPreview.clientWidth;
    canvas.height = videoPreview.clientHeight;
    drawROI();
}

window.addEventListener('resize', () => {
    if (previewContainer.style.display !== 'none') {
        resizeCanvas();
    }
});

// Canvas Drawing Logic
canvas.addEventListener('mousedown', (e) => {
    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    startX = e.clientX - rect.left;
    startY = e.clientY - rect.top;
    roi = { x: startX, y: startY, w: 0, h: 0 };
});

canvas.addEventListener('mousemove', (e) => {
    if (!isDrawing) return;
    const rect = canvas.getBoundingClientRect();
    const currentX = e.clientX - rect.left;
    const currentY = e.clientY - rect.top;
    
    roi.w = currentX - startX;
    roi.h = currentY - startY;
    
    drawROI();
});

canvas.addEventListener('mouseup', () => {
    isDrawing = false;
    // Normalize negative widths/heights
    if (roi && roi.w < 0) {
        roi.x += roi.w;
        roi.w = Math.abs(roi.w);
    }
    if (roi && roi.h < 0) {
        roi.y += roi.h;
        roi.h = Math.abs(roi.h);
    }
    drawROI();
});

clearRoiBtn.addEventListener('click', () => {
    roi = null;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
});

function drawROI() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (roi) {
        ctx.strokeStyle = '#fcd34d'; // accent-yellow
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.strokeRect(roi.x, roi.y, roi.w, roi.h);
        
        // Draw counting line indicator (always horizontal)
        ctx.strokeStyle = 'rgba(59, 130, 246, 0.8)'; // blueish
        ctx.setLineDash([]);
        ctx.beginPath();
        const midY = roi.y + roi.h / 2;
        ctx.moveTo(roi.x, midY);
        ctx.lineTo(roi.x + roi.w, midY);
        ctx.stroke();
    }
}

// Handle Form Submission
bpsForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const file = videoUpload.files[0];
    const minArea = document.getElementById('min-area').value;
    const startTime = document.getElementById('start-time').value;
    const endTime = document.getElementById('end-time').value;

    // Default to 0 if no ROI drawn (backend will handle fallback)
    let actualRoi = { x: 0, y: 0, w: 0, h: 0 };
    
    if (roi && roi.w !== 0 && roi.h !== 0) {
        const scaleX = videoPreview.videoWidth / canvas.width;
        const scaleY = videoPreview.videoHeight / canvas.height;
        actualRoi = {
            x: Math.round(roi.x * scaleX),
            y: Math.round(roi.y * scaleY),
            w: Math.round(roi.w * scaleX),
            h: Math.round(roi.h * scaleY)
        };
    }

    const formData = new FormData();
    formData.append('video', file);
    formData.append('min_area', minArea);
    formData.append('start_time', startTime);
    formData.append('end_time', endTime);
    formData.append('roi_x', actualRoi.x);
    formData.append('roi_y', actualRoi.y);
    formData.append('roi_w', actualRoi.w);
    formData.append('roi_h', actualRoi.h);
    formData.append('video_width', videoPreview.videoWidth);
    formData.append('video_height', videoPreview.videoHeight);

    loadingOverlay.style.display = 'flex';
    resultsPanel.style.display = 'none';

    try {
        const response = await fetch('http://localhost:8000/calculate-bps', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errText = await response.text();
            throw new Error(`Server error: ${response.status} - ${errText}`);
        }

        const data = await response.json();
        
        document.getElementById('res-avg').textContent = data.average_bps.toFixed(2);
        document.getElementById('res-peak').textContent = data.peak_bps;
        document.getElementById('res-total').textContent = data.total_balls;
        document.getElementById('res-dur').textContent = data.video_duration.toFixed(2);
        
        resultsPanel.style.display = 'block';
    } catch (error) {
        alert(`Error calculating BPS: ${error.message}`);
    } finally {
        loadingOverlay.style.display = 'none';
    }
});
