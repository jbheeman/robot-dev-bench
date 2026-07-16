import uuid
from typing import Dict, Any

class JobStore:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def create_job(self) -> str:
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {
            "status": "processing",
            "progress": 0.0,
            "message": "Initializing...",
            "result": None,
            "error": None
        }
        return job_id

    def update_job(self, job_id: str, progress: float, message: str):
        if job_id in self.jobs:
            self.jobs[job_id]["progress"] = progress
            self.jobs[job_id]["message"] = message

    def finish_job(self, job_id: str, result: Dict[str, Any]):
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = "success"
            self.jobs[job_id]["progress"] = 1.0
            self.jobs[job_id]["message"] = "Complete"
            self.jobs[job_id]["result"] = result

    def fail_job(self, job_id: str, error_message: str):
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = "error"
            self.jobs[job_id]["message"] = error_message
            self.jobs[job_id]["error"] = error_message

    def get_job(self, job_id: str) -> Dict[str, Any]:
        return self.jobs.get(job_id)
