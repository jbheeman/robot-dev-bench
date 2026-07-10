// Log Library (log integration) + Multi-Model Evaluation frontend logic.
document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('lib-file-input');
    const chooseBtn = document.getElementById('lib-choose-file-btn');
    const chosenLabel = document.getElementById('lib-chosen-filename');
    const modelNameInput = document.getElementById('lib-model-name');
    const taskSelect = document.getElementById('lib-task-select');
    const registerBtn = document.getElementById('lib-register-btn');
    const registerStatus = document.getElementById('lib-register-status');
    const tableBody = document.getElementById('log-table-body');
    const detailPanel = document.getElementById('validation-detail');

    const mmeTaskSelect = document.getElementById('mme-task-select');
    const mmeRunBtn = document.getElementById('mme-run-btn');
    const mmeStatus = document.getElementById('mme-status');
    const mmeResults = document.getElementById('mme-results');
    const mmeRanking = document.getElementById('mme-ranking');
    const mmeMetricHead = document.getElementById('mme-metric-head');
    const mmeMetricBody = document.getElementById('mme-metric-body');
    const mmeBatchNote = document.getElementById('mme-batch-note');

    if (!tableBody) return; // Section not present

    let selectedFile = null;
    let logs = [];
    let selectedLogIds = new Set();
    let openReportLogId = null;

    const METRIC_LABELS = {
        control_precision: 'Control Precision (RMSE, rad)',
        dynamic_stability: 'Dynamic Stability (rad²)',
        cost_of_transport: 'Cost of Transport',
        system_latency: 'System Latency (s)'
    };

    function esc(s) {
        return String(s).replace(/[&<>"']/g, c => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
        }[c]));
    }

    function setStatus(el, message, kind) {
        el.textContent = message;
        el.classList.remove('hidden');
        el.style.color = kind === 'error' ? '#ef4444' : (kind === 'ok' ? '#10b981' : 'var(--text-secondary)');
    }

    function badge(status) {
        const cls = status === 'valid' ? 'badge-pass' : (status === 'warnings' ? 'badge-warn' : 'badge-fail');
        return `<span class="badge ${cls}">${esc(status || 'unknown')}</span>`;
    }

    function updateRunButton() {
        mmeRunBtn.textContent = `Run Parallel Evaluation (${selectedLogIds.size} selected)`;
        mmeRunBtn.disabled = selectedLogIds.size === 0;
    }

    function updateRegisterButton() {
        registerBtn.disabled = !(selectedFile && modelNameInput.value.trim());
    }

    // --- Log Library ---------------------------------------------------------

    chooseBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', function () {
        if (this.files.length > 0) {
            selectedFile = this.files[0];
            chosenLabel.textContent = selectedFile.name;
        }
        updateRegisterButton();
    });
    modelNameInput.addEventListener('input', updateRegisterButton);

    registerBtn.addEventListener('click', async () => {
        if (!selectedFile || !modelNameInput.value.trim()) return;
        registerBtn.disabled = true;
        setStatus(registerStatus, 'Validating pipeline compatibility...', 'info');
        try {
            const formData = new FormData();
            formData.append('file', selectedFile);
            formData.append('model_name', modelNameInput.value.trim());
            formData.append('task', taskSelect.value);

            const response = await fetch('/api/logs', { method: 'POST', body: formData });
            const data = await response.json();
            if (!response.ok || data.status !== 'success') {
                throw new Error(data.detail || data.message || 'Registration failed');
            }
            const v = data.log.validation_status;
            setStatus(registerStatus,
                `Registered '${data.log.filename}' — validation: ${v}.` +
                (v === 'invalid' ? ' See report for pipeline incompatibilities.' : ''),
                v === 'invalid' ? 'error' : 'ok');
            selectedFile = null;
            fileInput.value = '';
            chosenLabel.textContent = 'No file chosen';
            await loadLogs();
        } catch (err) {
            setStatus(registerStatus, 'Registration failed: ' + err.message, 'error');
        } finally {
            updateRegisterButton();
        }
    });

    async function loadLogs() {
        try {
            const response = await fetch('/api/logs');
            const data = await response.json();
            logs = data.logs || [];
        } catch (err) {
            console.error('Failed to load logs:', err);
            logs = [];
        }
        // Drop selections for logs that no longer exist
        const ids = new Set(logs.map(l => l.id));
        selectedLogIds = new Set([...selectedLogIds].filter(id => ids.has(id)));
        renderLogTable();
        updateRunButton();
    }

    function renderLogTable() {
        if (logs.length === 0) {
            tableBody.innerHTML = '<tr><td colspan="9" class="table-empty">No logs registered yet.</td></tr>';
            detailPanel.classList.add('hidden');
            return;
        }
        tableBody.innerHTML = logs.map(log => `
            <tr>
                <td><input type="checkbox" class="log-select" data-id="${log.id}" ${selectedLogIds.has(log.id) ? 'checked' : ''}></td>
                <td>${log.id}</td>
                <td class="cell-filename" title="${esc(log.filename)}">${esc(log.filename)}</td>
                <td>${esc(log.model_name)}</td>
                <td>${esc(log.task)}</td>
                <td>${log.row_count ?? '--'}</td>
                <td>${esc(log.schema_format || '--')}</td>
                <td>${badge(log.validation_status)}</td>
                <td>
                    <button class="table-btn report-btn" data-id="${log.id}">Report</button>
                    <button class="table-btn delete-btn" data-id="${log.id}">Delete</button>
                </td>
            </tr>
        `).join('');

        tableBody.querySelectorAll('.log-select').forEach(cb => {
            cb.addEventListener('change', () => {
                const id = parseInt(cb.dataset.id, 10);
                if (cb.checked) selectedLogIds.add(id); else selectedLogIds.delete(id);
                updateRunButton();
            });
        });
        tableBody.querySelectorAll('.report-btn').forEach(btn => {
            btn.addEventListener('click', () => toggleReport(parseInt(btn.dataset.id, 10)));
        });
        tableBody.querySelectorAll('.delete-btn').forEach(btn => {
            btn.addEventListener('click', async () => {
                const id = parseInt(btn.dataset.id, 10);
                if (!confirm(`Delete log ${id} and its evaluations?`)) return;
                await fetch(`/api/logs/${id}`, { method: 'DELETE' });
                if (openReportLogId === id) { openReportLogId = null; detailPanel.classList.add('hidden'); }
                await loadLogs();
            });
        });
    }

    function toggleReport(logId) {
        if (openReportLogId === logId) {
            openReportLogId = null;
            detailPanel.classList.add('hidden');
            return;
        }
        const log = logs.find(l => l.id === logId);
        if (!log || !log.validation_report) return;
        openReportLogId = logId;

        const report = log.validation_report;
        const checksHtml = (report.checks || []).map(c => `
            <div class="check-row">
                ${badge(c.status === 'pass' ? 'valid' : (c.status === 'warn' ? 'warnings' : 'invalid'))}
                <strong>${esc(c.name)}</strong>
                <span>${esc(c.detail)}</span>
            </div>
        `).join('');

        const availability = report.metric_availability || {};
        const availHtml = Object.entries(METRIC_LABELS).map(([key, label]) =>
            `<span class="badge ${availability[key] ? 'badge-pass' : 'badge-warn'}">${esc(label.split(' (')[0])}: ${availability[key] ? 'available' : 'missing'}</span>`
        ).join(' ');

        detailPanel.innerHTML = `
            <h4 class="mme-subtitle">Validation Report — Log ${log.id} (${esc(log.filename)})</h4>
            <p class="lib-status">Format: <strong>${esc(report.schema_format)}</strong> ·
               Rows: ${report.row_count} · Duration: ${report.duration_sec != null ? report.duration_sec.toFixed(2) + 's' : '--'} ·
               Joints: ${report.joint_count ?? '--'}</p>
            <p class="lib-status">Scoring metric coverage: ${availHtml}</p>
            ${checksHtml}
        `;
        detailPanel.classList.remove('hidden');
    }

    // --- Multi-Model Evaluation ----------------------------------------------

    mmeRunBtn.addEventListener('click', async () => {
        if (selectedLogIds.size === 0) return;
        mmeRunBtn.disabled = true;
        mmeResults.classList.add('hidden');
        setStatus(mmeStatus, `Running ${selectedLogIds.size} evaluation(s) in parallel...`, 'info');
        try {
            const response = await fetch('/api/evaluate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ log_ids: [...selectedLogIds], task: mmeTaskSelect.value })
            });
            const data = await response.json();
            if (!response.ok || data.status !== 'success') {
                throw new Error(data.detail || data.message || 'Evaluation failed');
            }
            renderEvaluation(data);
            mmeStatus.classList.add('hidden');
        } catch (err) {
            setStatus(mmeStatus, 'Evaluation failed: ' + err.message, 'error');
        } finally {
            updateRunButton();
        }
    });

    function renderEvaluation(data) {
        const cal = data.calibration;
        const okResults = data.results.filter(r => r.status === 'ok');
        const failed = data.results.filter(r => r.status !== 'ok');

        mmeRanking.innerHTML = cal.ranking.map((r, i) => `
            <div class="rank-card">
                <span class="rank-pos">#${i + 1}</span>
                <span class="rank-model">${esc(r.model_name)}</span>
                <span class="rank-score">${r.score.toFixed(3)}</span>
                <span class="rank-tier">${esc(r.tier)}</span>
            </div>
        `).join('') + failed.map(r => `
            <div class="rank-card rank-failed">
                <span class="rank-model">${esc(r.model_name)}</span>
                <span class="rank-tier">FAILED: ${esc(r.error || 'unknown error')}</span>
            </div>
        `).join('');

        // Metric comparison table: rows = scoring metrics, cols = models + stats + reference bounds
        const models = okResults.map(r => r.model_name);
        mmeMetricHead.innerHTML = `<tr>
            <th>Metric</th>
            ${models.map(m => `<th>${esc(m)}</th>`).join('')}
            <th>Mean ± Std</th>
            <th>Best</th>
            <th>Class 1 ideal / Class 3 limit</th>
        </tr>`;

        mmeMetricBody.innerHTML = Object.entries(METRIC_LABELS).map(([key, label]) => {
            const m = cal.metrics[key];
            if (!m) return '';
            const cells = models.map(model => {
                const v = m.values[model];
                const ns = m.normalized_scores[model];
                if (v == null) return '<td class="cell-missing">--</td>';
                const isBest = m.best_model === model;
                return `<td class="${isBest ? 'cell-best' : ''}">${v}<small class="cell-subscore">(${ns != null ? ns.toFixed(2) : '--'})</small></td>`;
            }).join('');
            const stats = m.mean != null ? `${m.mean} ± ${m.std}` : '--';
            const bounds = `${m.reference_bounds.class1_ideal} / ${m.reference_bounds.class3_acceptable}`;
            return `<tr>
                <td>${esc(label)} <small class="cell-subscore">w=${m.weight}</small></td>
                ${cells}
                <td>${stats}</td>
                <td>${m.best_model ? esc(m.best_model) : '--'}</td>
                <td>${bounds}</td>
            </tr>`;
        }).join('');

        const spread = cal.score_spread;
        mmeBatchNote.textContent =
            `Batch ${data.batch_id} · task: ${data.task} · ${cal.num_models} model(s) evaluated` +
            (cal.num_failed ? `, ${cal.num_failed} failed` : '') +
            (spread ? ` · score spread ${spread.min}–${spread.max} (mean ${spread.mean}, std ${spread.std})` : '') +
            '. Cell format: metric value (normalized sub-score 0–1). Results persisted to evaluation history.';
        mmeBatchNote.classList.remove('hidden');
        mmeResults.classList.remove('hidden');
    }

    // Initial load
    loadLogs();
});
