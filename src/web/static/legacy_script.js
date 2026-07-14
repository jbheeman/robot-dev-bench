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
        // Initial setup
        toggleMetricsDisplay();
    }

    let dataPrimary = null;
    let dataBaseline = null;
    window.uploadedLogs = []; // Store logs for the session

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
            document.getElementById('drop-zone-b').classList.add('has-file-b');
        } else {
            filePrimary = file;
            document.getElementById('drop-zone').classList.add('has-file');
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
        
        // Save to session history
        window.uploadedLogs.push(data);
        
        return data;
    }

    function renderMetric(el, key, dataPrimary, dataBaseline, suffix='') {
        let val = dataPrimary.metrics[key];
        if (val === undefined || val === null) {
            el.innerHTML = '--' + suffix;
            return;
        }
        let html = val.toFixed(3) + suffix;
        if (dataBaseline && dataBaseline.metrics[key] !== undefined && dataBaseline.metrics[key] !== null) {
            let valB = dataBaseline.metrics[key];
            let delta = val - valB;
            let sign = delta > 0 ? '+' : '';
            html += ` <span style="font-size: 0.7em; margin-left: 8px; color: #9ca3af;">(${sign}${delta.toFixed(3)})</span>`;
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

        // Metrics
        renderMetric(elLdlj, 'smoothness_ldlj', dataPrimary, dataBaseline);
        renderMetric(elSparc, 'smoothness_sparc', dataPrimary, dataBaseline);
        renderMetric(elSymmetry, 'symmetry', dataPrimary, dataBaseline, ' <small>%</small>');
        renderMetric(elPeriodicity, 'periodicity', dataPrimary, dataBaseline);
        renderMetric(elRom, 'rom_utilisation', dataPrimary, dataBaseline, ' <small>rad</small>');
        
        // New Metrics
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
        
        // Hide chart controls (checkboxes) for the PDF
        const chartControls = document.querySelector('.chart-controls');
        const originalDisplay = chartControls ? chartControls.style.display : '';
        if (chartControls) chartControls.style.display = 'none';

        // Clear chart tooltips and SHOW titles for the PDF
        Object.values(charts).forEach(chart => {
            if (chart.tooltip) {
                chart.tooltip.setActiveElements([], {x: 0, y: 0});
                chart.setActiveElements([]);
            }
            if (chart.options.plugins) {
                let titleText = "Telemetry";
                if (chart.data && chart.data.datasets && chart.data.datasets.length > 0) {
                    titleText = chart.data.datasets[0].label.replace(' (Primary)', '');
                }
                chart.options.plugins.title = {
                    display: true,
                    text: titleText,
                    color: '#f8fafc',
                    font: { size: 14, family: "'Outfit', sans-serif" }
                };
            }
            chart.update();
        });
        
        // Wait a tiny bit for charts to re-render with titles
        await new Promise(resolve => setTimeout(resolve, 100));

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
            
            // Restore chart controls
            if (chartControls) chartControls.style.display = originalDisplay;
            
            // Hide titles again
            Object.values(charts).forEach(chart => {
                if (chart.options.plugins && chart.options.plugins.title) {
                    chart.options.plugins.title.display = false;
                    chart.update();
                }
            });
        }
    });

    // --- Log Library Logic ---
    const logLibraryBtn = document.getElementById('log-library-btn');
    const logLibraryModal = document.getElementById('log-library-modal');
    const closeLibraryModalBtn = document.getElementById('close-modal-btn');
    const logList = document.getElementById('log-list');
    const rankLogsBtn = document.getElementById('rank-logs-btn');
    const libraryUploadInput = document.getElementById('library-upload-input');
    const libraryUploadBtn = document.getElementById('library-upload-btn');

    if (libraryUploadBtn && libraryUploadInput) {
        libraryUploadBtn.addEventListener('click', () => {
            libraryUploadInput.click();
        });

        libraryUploadInput.addEventListener('change', async function() {
            if (this.files.length > 0) {
                const originalText = libraryUploadBtn.textContent;
                libraryUploadBtn.textContent = `Uploading ${this.files.length}...`;
                libraryUploadBtn.disabled = true;
                
                try {
                    for (const file of this.files) {
                        await fetchFile(file);
                    }
                    populateLogLibrary();
                } catch (e) {
                    alert('Failed to upload one or more logs: ' + e.message);
                } finally {
                    libraryUploadBtn.textContent = originalText;
                    libraryUploadBtn.disabled = false;
                    this.value = ''; // Reset input
                }
            }
        });
    }

    const leaderboardModal = document.getElementById('leaderboard-modal');
    const closeLeaderboardBtn = document.getElementById('close-leaderboard-btn');
    const leaderboardList = document.getElementById('leaderboard-list');

    function populateLogLibrary() {
        logList.innerHTML = '';
        if (window.uploadedLogs.length === 0) {
            logList.innerHTML = '<p style="color: var(--text-secondary); text-align: center; margin-top: 2rem;">No logs uploaded yet.</p>';
            rankLogsBtn.disabled = true;
            return;
        }

        window.uploadedLogs.forEach((log, index) => {
            const item = document.createElement('div');
            item.className = 'log-item';
            
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.value = index;
            checkbox.addEventListener('change', () => {
                const checked = logList.querySelectorAll('input[type="checkbox"]:checked');
                rankLogsBtn.disabled = checked.length === 0;
                if (checkbox.checked) {
                    item.classList.add('selected');
                } else {
                    item.classList.remove('selected');
                }
            });

            item.addEventListener('click', (e) => {
                if (e.target !== checkbox) {
                    checkbox.checked = !checkbox.checked;
                    checkbox.dispatchEvent(new Event('change'));
                }
            });

            const details = document.createElement('div');
            details.className = 'log-item-details';
            details.innerHTML = `
                <div class="log-item-name">${log.filename}</div>
                <div class="log-item-meta">Ready for evaluation</div>
            `;

            item.appendChild(checkbox);
            item.appendChild(details);
            logList.appendChild(item);
        });
    }

    logLibraryBtn.addEventListener('click', () => {
        populateLogLibrary();
        logLibraryModal.classList.remove('hidden');
    });

    closeLibraryModalBtn.addEventListener('click', () => {
        logLibraryModal.classList.add('hidden');
    });

    let currentSelectedLogs = [];

    function renderLeaderboard(selectedLogs) {
        // Sort descending by score
        selectedLogs.sort((a, b) => b.classification.score - a.classification.score);
        
        // Populate leaderboard
        leaderboardList.innerHTML = '';
        selectedLogs.forEach((log, index) => {
            const item = document.createElement('div');
            item.className = 'leaderboard-item';
            item.innerHTML = `
                <div class="leaderboard-rank">#${index + 1}</div>
                <div class="leaderboard-item-details">
                    <div class="leaderboard-item-name">${log.filename}</div>
                    <div class="leaderboard-item-tier">Task: ${log.task} | Tier: ${log.classification.tier}</div>
                </div>
                <div class="leaderboard-item-score">${log.classification.score.toFixed(3)}</div>
            `;
            leaderboardList.appendChild(item);
        });
    }

    rankLogsBtn.addEventListener('click', () => {
        const checkedBoxes = logList.querySelectorAll('input[type="checkbox"]:checked');
        currentSelectedLogs = Array.from(checkedBoxes).map(cb => window.uploadedLogs[cb.value]);
        
        const leaderboardTaskSelect = document.getElementById('leaderboard-task-select');
        if (leaderboardTaskSelect && currentSelectedLogs.length > 0) {
            leaderboardTaskSelect.value = currentSelectedLogs[0].task;
        }

        renderLeaderboard(currentSelectedLogs);

        // Hide library modal, show leaderboard modal
        logLibraryModal.classList.add('hidden');
        leaderboardModal.classList.remove('hidden');
    });

    const leaderboardTaskSelect = document.getElementById('leaderboard-task-select');
    if (leaderboardTaskSelect) {
        leaderboardTaskSelect.addEventListener('change', async (e) => {
            const newTask = e.target.value;
            try {
                leaderboardTaskSelect.disabled = true;
                for (const log of currentSelectedLogs) {
                    const response = await fetch('/api/reclassify', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            task: newTask,
                            metrics: log.metrics
                        })
                    });
                    
                    if (!response.ok) throw new Error('Reclassification failed');
                    const data = await response.json();
                    
                    // Update log object in place
                    log.task = data.task;
                    log.classification.score = data.score;
                    log.classification.tier = data.tier;
                }
                
                // Re-render with updated data
                renderLeaderboard(currentSelectedLogs);
                populateLogLibrary(); 
            } catch (err) {
                alert('Error reclassifying: ' + err.message);
            } finally {
                leaderboardTaskSelect.disabled = false;
            }
        });
    }

    closeLeaderboardBtn.addEventListener('click', () => {
        leaderboardModal.classList.add('hidden');
    });
});
