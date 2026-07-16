import re

with open("src/web/static/script.js", "r") as f:
    content = f.read()

# Replace the fetch logic with polling logic
new_fetch = """            const response = await fetch('/api/upload_av', {
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
            const resultData = jobResult;"""

content = re.sub(r'            const response = await fetch\(\'/api/upload_av\', \{.*?const data = await response\.json\(\);\s*if \(data\.status !== \'success\'\) \{\s*throw new Error\(data\.message \|\| \'Upload failed\'\);\s*\}', new_fetch, content, flags=re.DOTALL)

# Now we must change `data.` to `resultData.` for the rest of the parsing
content = content.replace('data.classification', 'resultData.classification')
content = content.replace('data.metrics', 'resultData.metrics')
content = content.replace('data.poses_3d', 'resultData.poses_3d')
content = content.replace('data.valid_mask', 'resultData.valid_mask')

with open("src/web/static/script.js", "w") as f:
    f.write(content)
