document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    const resultsSection = document.getElementById('results-section');

    // UI Elements for Data
    const elScore = document.getElementById('classification-score');
    const elTier = document.getElementById('policy-tier');
    const elLdlj = document.getElementById('metric-ldlj');
    const elSparc = document.getElementById('metric-sparc');
    const elSymmetry = document.getElementById('metric-symmetry');
    const elPeriodicity = document.getElementById('metric-periodicity');
    const elRom = document.getElementById('metric-rom');
    const scoreRing = document.querySelector('.score-ring');
    const taskSelect = document.getElementById('task-select');

    // UI Elements for New Metrics
    const elFlightTime = document.getElementById('metric-flight-time');
    const elPeakZAccel = document.getElementById('metric-peak-z-accel');
    const elLandingJerk = document.getElementById('metric-landing-jerk');
    const elComOscillation = document.getElementById('metric-com-oscillation');
    const elTransitionTime = document.getElementById('metric-transition-time');

    // Toggle logic for metrics visibility based on task
    function toggleMetricsDisplay() {
        const task = taskSelect.value;
        const allCards = [
            'card-ldlj', 'card-sparc', 'card-symmetry', 'card-periodicity', 'card-rom',
            'card-flight-time', 'card-peak-z-accel', 'card-landing-jerk',
            'card-com-oscillation', 'card-transition-time'
        ];
        
        // Hide all first
        allCards.forEach(id => {
            const el = document.getElementById(id);
            if(el) el.style.display = 'none';
        });

        // Show based on task weights
        const cardsToShow = [];
        if (task === 'walking') {
            cardsToShow.push('card-ldlj', 'card-sparc', 'card-symmetry', 'card-periodicity', 'card-rom');
        } else if (task === 'reaching' || task === 'manipulation') {
            cardsToShow.push('card-ldlj', 'card-sparc', 'card-rom');
        } else if (task === 'jumping') {
            cardsToShow.push('card-flight-time', 'card-peak-z-accel', 'card-landing-jerk');
        } else if (task === 'transitions') {
            cardsToShow.push('card-com-oscillation', 'card-transition-time', 'card-ldlj', 'card-sparc');
        } else if (task === 'testing') {
            cardsToShow.push(...allCards); // Show all just to see them
        }

        cardsToShow.forEach(id => {
            const el = document.getElementById(id);
            if(el) el.style.display = 'flex';
        });
    }

    if (taskSelect) {
        taskSelect.addEventListener('change', toggleMetricsDisplay);
        // Initial setup
        toggleMetricsDisplay();
    }

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
    
    const changeFileBtn = document.getElementById('change-file-btn');
    if (changeFileBtn) {
        changeFileBtn.addEventListener('click', () => fileInput.click());
    }
    
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
        
        // Hide default drop content while loading/success
        document.getElementById('drop-content-default').classList.add('hidden');
        document.getElementById('drop-content-success').classList.add('hidden');

        const formData = new FormData();
        formData.append('file', file);
        if (taskSelect) {
            formData.append('task', taskSelect.value);
        }

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
                
                // Show success drop content
                document.getElementById('drop-content-success').classList.remove('hidden');
                document.getElementById('uploaded-filename').textContent = file.name;
                
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
            // Revert drop zone UI
            document.getElementById('drop-content-default').classList.remove('hidden');
            document.getElementById('drop-content-success').classList.add('hidden');
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
        elLdlj.textContent = data.metrics.smoothness_ldlj.toFixed(3);
        elSparc.textContent = data.metrics.smoothness_sparc.toFixed(3);
        
        if (data.metrics.symmetry !== null && data.metrics.symmetry !== undefined) {
            elSymmetry.innerHTML = `${data.metrics.symmetry.toFixed(3)} <small>%</small>`;
        } else {
            elSymmetry.innerHTML = `-- <small>%</small>`;
        }
        
        elPeriodicity.textContent = data.metrics.periodicity.toFixed(3);
        elRom.innerHTML = `${data.metrics.rom_utilisation.toFixed(3)} <small>rad</small>`;
        
        // New Metrics
        if (data.metrics.flight_time !== undefined) elFlightTime.innerHTML = `${data.metrics.flight_time.toFixed(3)} <small>s</small>`;
        if (data.metrics.peak_z_accel !== undefined) elPeakZAccel.innerHTML = `${data.metrics.peak_z_accel.toFixed(3)} <small>rad/s²</small>`;
        if (data.metrics.landing_jerk !== undefined) elLandingJerk.innerHTML = `${data.metrics.landing_jerk.toFixed(3)} <small>rad/s³</small>`;
        if (data.metrics.com_oscillation !== undefined) elComOscillation.textContent = data.metrics.com_oscillation.toFixed(3);
        if (data.metrics.transition_time !== undefined) elTransitionTime.innerHTML = `${data.metrics.transition_time.toFixed(3)} <small>s</small>`;
        
        // Pass playback data to viewer
        if (data.playback && window.loadPlaybackData) {
            window.loadPlaybackData(data.playback);
        }
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
