document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    const resultsSection = document.getElementById('results-section');

    // UI Elements for Data
    const elScore = document.getElementById('classification-score');
    const elTier = document.getElementById('policy-tier');
    const scoreRing = document.querySelector('.score-ring');
    const taskSelect = document.getElementById('task-select');

    // UI Elements for the four core evaluation metrics (drive the real weighted-sum score)
    const elControlPrecision = document.getElementById('metric-control-precision');
    const elDynamicStability = document.getElementById('metric-dynamic-stability');
    const elCostOfTransport = document.getElementById('metric-cost-of-transport');
    const elSystemLatency = document.getElementById('metric-system-latency');

    // UI Elements for the informational biomechanical metrics
    const elLdlj = document.getElementById('metric-ldlj');
    const elSparc = document.getElementById('metric-sparc');
    const elSymmetry = document.getElementById('metric-symmetry');
    const elPeriodicity = document.getElementById('metric-periodicity');
    const elRom = document.getElementById('metric-rom');
    const elFlightTime = document.getElementById('metric-flight-time');
    const elPeakZAccel = document.getElementById('metric-peak-z-accel');
    const elLandingJerk = document.getElementById('metric-landing-jerk');
    const elComOscillation = document.getElementById('metric-com-oscillation');
    const elTransitionTime = document.getElementById('metric-transition-time');

    // Toggle logic for the informational biomechanical cards based on task.
    // The four scoring metric cards (control precision, dynamic stability, cost of
    // transport, system latency) are always shown since they always drive the score.
    function toggleMetricsDisplay(taskOverride) {
        const task = taskOverride || taskSelect.value;
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

        // Show based on task relevance
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
        // Initial setup
        toggleMetricsDisplay();
    }

    let dataPrimary = null;
    let dataBaseline = null;

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function setupDropZone(zoneId, inputId, browseBtnId, changeBtnId, successId, defaultId, filenameId, isBaseline) {
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
            if (e.dataTransfer.files.length > 0) handleFileSelection(e.dataTransfer.files[0], isBaseline, successId, defaultId, filenameId);
        }, false);
        
        browseBtn.addEventListener('click', () => input.click());
        if (changeBtn) changeBtn.addEventListener('click', () => input.click());
        
        input.addEventListener('change', function() {
            if (this.files.length > 0) handleFileSelection(this.files[0], isBaseline, successId, defaultId, filenameId);
        });
    }

    // Set up both drop zones
    setupDropZone('drop-zone', 'file-input', 'browse-btn', 'change-file-btn', 'drop-content-success', 'drop-content-default', 'uploaded-filename', false);
    setupDropZone('drop-zone-b', 'file-input-b', 'browse-btn-b', 'change-file-btn-b', 'drop-content-success-b', 'drop-content-default-b', 'uploaded-filename-b', true);

    let filePrimary = null;
    let fileBaseline = null;

    function handleFileSelection(file, isBaseline, successId, defaultId, filenameId) {
        if (isBaseline) {
            fileBaseline = file;
        } else {
            filePrimary = file;
        }
        document.getElementById(defaultId).classList.add('hidden');
        document.getElementById(successId).classList.remove('hidden');
        document.getElementById(filenameId).textContent = file.name;
        
        document.getElementById('run-analysis-btn').style.display = 'block';
    }

    document.getElementById('run-analysis-btn').addEventListener('click', async () => {
        if (!filePrimary && !fileBaseline) return;
        
        resultsSection.classList.add('hidden');
        loadingOverlay.classList.remove('hidden');
        
        try {
            let dataA = null;
            let dataB = null;
            
            if (filePrimary) {
                dataA = await fetchFile(filePrimary);
                dataPrimary = dataA;
            }
            if (fileBaseline) {
                dataB = await fetchFile(fileBaseline);
                dataBaseline = dataB;
            }
            
            if (dataPrimary) {
                const uploadedTask = taskSelect ? taskSelect.value : null;
                toggleMetricsDisplay(uploadedTask);
                updateDashboard(dataPrimary, dataBaseline);
                
                if (uploadedTask) {
                    const badge = document.getElementById('evaluated-task-label');
                    if (badge) {
                        badge.textContent = uploadedTask;
                        badge.classList.remove('hidden');
                    }
                }
                
                loadingOverlay.classList.add('hidden');
                resultsSection.classList.remove('hidden');
                resultsSection.style.display = 'block';
                setTimeout(() => animateScoreRing(dataPrimary.classification.score), 100);
            } else if (dataBaseline) {
                loadingOverlay.classList.add('hidden');
                alert("Please upload a primary policy.");
            }
        } catch (error) {
            console.error('Error running analysis:', error);
            alert('Failed to analyze log files: ' + error.message);
            loadingOverlay.classList.add('hidden');
        }
    });

    async function fetchFile(file) {
        const formData = new FormData();
        formData.append('file', file);
        if (taskSelect) {
            formData.append('task', taskSelect.value);
        }
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        if (!response.ok) throw new Error('Network response was not ok');
        const data = await response.json();
        if (data.status !== 'success') throw new Error(data.message || 'Upload failed');
        return data;
    }

    function renderMetric(el, key, dataPrimary, dataBaseline, suffix='', precision=3) {
        let val = dataPrimary.metrics[key];
        if (val === undefined || val === null) {
            el.innerHTML = '--' + suffix;
            return;
        }
        let html = val.toFixed(precision) + suffix;
        if (dataBaseline && dataBaseline.metrics[key] !== undefined && dataBaseline.metrics[key] !== null) {
            let valB = dataBaseline.metrics[key];
            let delta = val - valB;
            let sign = delta > 0 ? '+' : '';
            html += ` <span style="font-size: 0.7em; margin-left: 8px; color: #9ca3af;">(${sign}${delta.toFixed(precision)})</span>`;
        }
        el.innerHTML = html;
    }

    function updateDashboard(dataPrimary, dataBaseline = null) {
        // Classification
        const score = dataPrimary.classification.score;
        let scoreHtml = `<span>${score.toFixed(3)}</span>`;
        if (dataBaseline && dataBaseline.classification.score) {
            let delta = score - dataBaseline.classification.score;
            let sign = delta > 0 ? '+' : '';
            let color = delta > 0 ? '#10b981' : (delta < 0 ? '#ef4444' : '#9ca3af');
            scoreHtml += `<span style="font-size: 0.45em; color: ${color}; margin-top: 4px;">(${sign}${delta.toFixed(3)})</span>`;
        }
        elScore.innerHTML = scoreHtml;
        elScore.style.display = 'flex';
        elScore.style.flexDirection = 'column';
        elScore.style.alignItems = 'center';
        elScore.style.lineHeight = '1';
        elTier.textContent = dataPrimary.classification.tier;

        // Determine color based on score (simulating policy tier colors)
        let tierColor = '#10b981'; // success/green
        if (score < 0.8) tierColor = '#ef4444'; // red
        else if (score < 0.9) tierColor = '#f59e0b'; // yellow

        elTier.style.background = `linear-gradient(90deg, ${tierColor}, ${adjustColor(tierColor, -20)})`;
        elTier.style.webkitBackgroundClip = 'text';

        // Core scoring metrics
        renderMetric(elControlPrecision, 'control_precision', dataPrimary, dataBaseline, ' <small>rad</small>', 4);
        renderMetric(elDynamicStability, 'dynamic_stability', dataPrimary, dataBaseline, ' <small>rad²</small>', 5);
        renderMetric(elCostOfTransport, 'cost_of_transport', dataPrimary, dataBaseline);
        renderMetric(elSystemLatency, 'system_latency', dataPrimary, dataBaseline, ' <small>s</small>', 4);

        // Informational biomechanical metrics
        renderMetric(elLdlj, 'smoothness_ldlj', dataPrimary, dataBaseline);
        renderMetric(elSparc, 'smoothness_sparc', dataPrimary, dataBaseline);
        renderMetric(elSymmetry, 'symmetry', dataPrimary, dataBaseline, ' <small>%</small>');
        renderMetric(elPeriodicity, 'periodicity', dataPrimary, dataBaseline);
        renderMetric(elRom, 'rom_utilisation', dataPrimary, dataBaseline, ' <small>rad</small>');
        renderMetric(elFlightTime, 'flight_time', dataPrimary, dataBaseline, ' <small>s</small>');
        renderMetric(elPeakZAccel, 'peak_z_accel', dataPrimary, dataBaseline, ' <small>rad/s²</small>');
        renderMetric(elLandingJerk, 'landing_jerk', dataPrimary, dataBaseline, ' <small>rad/s³</small>');
        renderMetric(elComOscillation, 'com_oscillation', dataPrimary, dataBaseline);
        renderMetric(elTransitionTime, 'transition_time', dataPrimary, dataBaseline, ' <small>s</small>');
        
        // Pass playback data to viewer
        if (dataPrimary.playback && window.loadPlaybackData) {
            window.loadPlaybackData(dataPrimary.playback, dataBaseline ? dataBaseline.playback : null);
            
            // Populate Charts
            if (dataPrimary.playback.timeseries) {
                const timeArrayA = dataPrimary.playback.ticks;
                const tsA = dataPrimary.playback.timeseries;
                const timeArrayB = dataBaseline && dataBaseline.playback.timeseries ? dataBaseline.playback.ticks : null;
                const tsB = dataBaseline && dataBaseline.playback.timeseries ? dataBaseline.playback.timeseries : null;
                
                initOrUpdateChart('chart-velocity', tsA.velocity, timeArrayA, tsB ? tsB.velocity : null, timeArrayB, 'Global Velocity', '#53a5a7');
                initOrUpdateChart('chart-acceleration', tsA.acceleration, timeArrayA, tsB ? tsB.acceleration : null, timeArrayB, 'Global Acceleration', '#1a965a');
                initOrUpdateChart('chart-jerk', tsA.jerk, timeArrayA, tsB ? tsB.jerk : null, timeArrayB, 'Global Jerk', '#f59e0b');
                if (tsA.com_oscillation) {
                    initOrUpdateChart('chart-com', tsA.com_oscillation, timeArrayA, tsB ? tsB.com_oscillation : null, timeArrayB, 'CoM Oscillation', '#ef4444');
                }
            }
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

    // --- Chart.js Logic ---
    let charts = {};
    
    // Default Chart options for our theme
    if (window.Chart) {
        Chart.defaults.color = 'rgba(255, 255, 255, 0.7)';
        Chart.defaults.font.family = "'Outfit', sans-serif";
    }
    
    const commonChartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
            legend: { display: false },
            tooltip: { 
                enabled: true,
                mode: 'index',
                intersect: false,
                displayColors: false,
                yAlign: 'bottom' // Keep tooltip consistently above the dot
            }
        },
        scales: {
            x: {
                display: false // Hide X axis labels to save space
            },
            y: {
                grid: { color: 'rgba(255, 255, 255, 0.1)' },
                border: { display: false }
            }
        },
        elements: {
            point: { radius: 0, hitRadius: 10, hoverRadius: 4 },
            line: { borderWidth: 2, tension: 0.1 }
        },
        interaction: {
            mode: 'index',
            intersect: false,
        }
    };

    function initOrUpdateChart(id, dataArrayA, timeArrayA, dataArrayB, timeArrayB, label, color) {
        if (!window.Chart) return;
        const ctx = document.getElementById(id).getContext('2d');
        const formattedLabels = timeArrayA.map(t => (t / 1000).toFixed(2) + 's');
        
        const datasets = [{
            label: label + ' (Primary)',
            data: dataArrayA,
            borderColor: color,
            backgroundColor: color + '33', // 20% opacity
            fill: true
        }];
        
        if (dataArrayB && timeArrayB) {
            // In a robust implementation, we would interpolate if timeArrayA and B differed in length.
            // For now, assume they align close enough for visual A/B or we pad/truncate based on lengths.
            let adjustedDataB = dataArrayB;
            if (dataArrayB.length > dataArrayA.length) {
                adjustedDataB = dataArrayB.slice(0, dataArrayA.length);
            } else if (dataArrayB.length < dataArrayA.length) {
                // Pad with last value or null
                const padding = new Array(dataArrayA.length - dataArrayB.length).fill(null);
                adjustedDataB = dataArrayB.concat(padding);
            }
            
            datasets.push({
                label: label + ' (Baseline)',
                data: adjustedDataB,
                borderColor: '#9ca3af', // Gray baseline
                borderDash: [5, 5], // Dashed line
                backgroundColor: 'transparent',
                fill: false
            });
        }
        
        if (charts[id]) {
            charts[id].data.labels = formattedLabels;
            charts[id].data.datasets = datasets;
            charts[id].update();
        } else {
            charts[id] = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: formattedLabels,
                    datasets: datasets
                },
                options: commonChartOptions
            });
        }
    }

    // Expose crosshair updater globally for viewer.js
    window.updateChartCrosshair = function(frameIndex) {
        Object.values(charts).forEach(chart => {
            if (chart.data.labels.length > frameIndex) {
                chart.setActiveElements([{ datasetIndex: 0, index: frameIndex }]);
                chart.tooltip.setActiveElements([{ datasetIndex: 0, index: frameIndex }], {x: 0, y: 0});
                chart.update();
            }
        });
    };

    // Toggle logic for stacked charts
    ['velocity', 'acceleration', 'jerk', 'com'].forEach(metric => {
        const toggle = document.getElementById(`toggle-${metric}`);
        const wrapper = document.getElementById(`wrapper-${metric}`);
        if (toggle && wrapper) {
            toggle.addEventListener('change', (e) => {
                wrapper.style.display = e.target.checked ? 'block' : 'none';
            });
        }
    });

    // Export Logic
    document.getElementById('export-csv-btn').addEventListener('click', () => {
        if (!dataPrimary) return;
        
        let csvContent = "data:text/csv;charset=utf-8,Metric,Primary Policy,Baseline Policy,Delta\n";
        
        const metrics = [
            { key: 'score', label: 'Overall Score', isClass: true },
            { key: 'control_precision', label: 'Control Precision (RMSE, rad)' },
            { key: 'dynamic_stability', label: 'Dynamic Stability (rad²)' },
            { key: 'cost_of_transport', label: 'Cost of Transport' },
            { key: 'system_latency', label: 'System Latency (s)' },
            { key: 'smoothness_ldlj', label: 'Smoothness (LDLJ)' },
            { key: 'smoothness_sparc', label: 'Smoothness (SPARC)' },
            { key: 'symmetry', label: 'Symmetry (%)' },
            { key: 'periodicity', label: 'Periodicity' },
            { key: 'rom_utilisation', label: 'RoM Utilisation (rad)' },
            { key: 'flight_time', label: 'Flight Time (s)' },
            { key: 'peak_z_accel', label: 'Peak Z Accel (rad/s²)' },
            { key: 'landing_jerk', label: 'Landing Jerk (rad/s³)' },
            { key: 'com_oscillation', label: 'CoM Oscillation' },
            { key: 'transition_time', label: 'Transition Time (s)' }
        ];
        
        metrics.forEach(m => {
            let valA = m.isClass ? dataPrimary.classification.score : dataPrimary.metrics[m.key];
            if (valA === undefined || valA === null) valA = '';
            else valA = valA.toFixed(3);
            
            let valB = '';
            let delta = '';
            if (dataBaseline) {
                let rawB = m.isClass ? dataBaseline.classification.score : dataBaseline.metrics[m.key];
                if (rawB !== undefined && rawB !== null) {
                    valB = rawB.toFixed(3);
                    let rawA = m.isClass ? dataPrimary.classification.score : dataPrimary.metrics[m.key];
                    if (rawA !== undefined && rawA !== null) {
                        delta = (rawA - rawB).toFixed(3);
                    }
                }
            }
            
            csvContent += `"${m.label}","${valA}","${valB}","${delta}"\n`;
        });
        
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", "model_evaluation.csv");
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });

    document.getElementById('export-pdf-btn').addEventListener('click', async () => {
        const btn = document.getElementById('export-pdf-btn');
        btn.textContent = "Generating...";
        btn.disabled = true;
        
        try {
            const resultsGrid = document.querySelector('.results-grid');
            const canvas = await html2canvas(resultsGrid, {
                backgroundColor: '#0a0a0e', // dark theme background
                scale: 2, // High resolution
                ignoreElements: (element) => element.classList.contains('viewer-card')
            });
            
            const imgData = canvas.toDataURL('image/png');
            const { jsPDF } = window.jspdf;
            
            // A4 page dimensions in mm
            const pdf = new jsPDF('p', 'mm', 'a4');
            const pageHeight = pdf.internal.pageSize.getHeight();
            const pageWidth = pdf.internal.pageSize.getWidth();
            
            // Fill background with dark theme color (#0a0a0e -> RGB: 10, 10, 14) to avoid white borders
            pdf.setFillColor(10, 10, 14);
            pdf.rect(0, 0, pageWidth, pageHeight, 'F');
            
            let pdfWidth = pageWidth;
            let pdfHeight = (canvas.height * pdfWidth) / canvas.width;
            
            // Scale down if it exceeds a single page (leave room for timestamp)
            if (pdfHeight > pageHeight - 20) {
                pdfHeight = pageHeight - 20;
                pdfWidth = (canvas.width * pdfHeight) / canvas.height;
            }
            
            // Center horizontally if scaled
            const xOffset = (pageWidth - pdfWidth) / 2;
            
            pdf.addImage(imgData, 'PNG', xOffset, 10, pdfWidth, pdfHeight);
            
            // Add Timestamp
            const dateStr = new Date().toLocaleString();
            pdf.setTextColor(150, 150, 150);
            pdf.setFontSize(10);
            pdf.text(`Exported on: ${dateStr}`, 10, pageHeight - 10);
            
            pdf.save('model_evaluation.pdf');
        } catch (e) {
            console.error(e);
            alert("Failed to generate PDF.");
        } finally {
            btn.textContent = "Export PDF";
            btn.disabled = false;
        }
    });
});
