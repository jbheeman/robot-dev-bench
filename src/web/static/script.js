document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    const resultsSection = document.getElementById('results-section');

    // UI Elements for Data
    const elScore = document.getElementById('classification-score');
    const elTier = document.getElementById('policy-tier');
    const elRmse = document.getElementById('metric-rmse');
    const elCot = document.getElementById('metric-cot');
    const elLatency = document.getElementById('metric-latency');
    const elStress = document.getElementById('metric-stress');
    const elImu = document.getElementById('metric-imu');
    const scoreRing = document.querySelector('.score-ring');

    // Event Listeners for Drag & Drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.add('active-drop');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove('active-drop');
        }, false);
    });

    dropZone.addEventListener('drop', handleDrop, false);
    browseBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            uploadFile(files[0]);
        }
    }

    function handleFileSelect(e) {
        if (this.files.length > 0) {
            uploadFile(this.files[0]);
        }
    }

    async function uploadFile(file) {
        // Only accept certain types or just let backend handle validation
        
        // Show loading state
        resultsSection.classList.add('hidden');
        loadingOverlay.classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            
            if (data.status === 'success') {
                updateDashboard(data);
                
                // Hide loading, show results
                loadingOverlay.classList.add('hidden');
                resultsSection.classList.remove('hidden');
                resultsSection.style.display = 'block'; // force layout
                
                // Slight delay to allow CSS transitions to run
                setTimeout(() => {
                    animateScoreRing(data.classification.score);
                }, 100);
            } else {
                throw new Error(data.message || 'Upload failed');
            }

        } catch (error) {
            console.error('Error uploading file:', error);
            alert('Failed to upload and analyze log file.');
            loadingOverlay.classList.add('hidden');
        }
    }

    function updateDashboard(data) {
        // Classification
        const score = data.classification.score;
        elScore.textContent = score.toFixed(3);
        elTier.textContent = data.classification.tier;

        // Determine color based on score (simulating policy tier colors)
        let tierColor = '#10b981'; // success/green
        if (score < 0.8) tierColor = '#ef4444'; // red
        else if (score < 0.9) tierColor = '#f59e0b'; // yellow

        elTier.style.background = `linear-gradient(90deg, ${tierColor}, ${adjustColor(tierColor, -20)})`;
        elTier.style.webkitBackgroundClip = 'text';

        // Metrics
        elRmse.textContent = data.metrics.rmse.toFixed(4);
        elCot.textContent = data.metrics.cot.toFixed(3);
        
        // Add ms styling logic securely
        elLatency.innerHTML = `${data.metrics.latency_ms.toFixed(2)} <small>ms</small>`;
        
        elStress.textContent = data.metrics.stress.toFixed(3);
        elImu.textContent = data.metrics.imu_variance.toFixed(4);
    }

    function animateScoreRing(score) {
        let currentPercent = 0;
        const targetPercent = Math.round(score * 100);
        
        let color = '#10b981';
        if (score < 0.8) color = '#ef4444';
        else if (score < 0.9) color = '#f59e0b';

        // Animate the conic gradient
        const interval = setInterval(() => {
            if (currentPercent >= targetPercent) {
                clearInterval(interval);
            } else {
                currentPercent++;
                scoreRing.style.background = `conic-gradient(${color} ${currentPercent}%, rgba(255,255,255,0.05) ${currentPercent}%)`;
            }
        }, 15); // Speed of animation
    }

    // Helper for adjusting hex color darkness (simple implementation)
    function adjustColor(color, amount) {
        return '#' + color.replace(/^#/, '').replace(/../g, color => ('0'+Math.min(255, Math.max(0, parseInt(color, 16) + amount)).toString(16)).substr(-2));
    }
});
