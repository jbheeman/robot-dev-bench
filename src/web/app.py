import os
import random
import time
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="Unitree G1-Edu Benchmarking Dashboard")

# Define path to static files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

# Mock Data Generator
def generate_mock_results(filename: str):
    # Simulate processing delay
    time.sleep(1.5)
    
    classification_score = random.uniform(0.7, 0.99)
    policy_tier = "Superhuman/Industrial" if classification_score > 0.9 else "Research" if classification_score > 0.8 else "Experimental"
    
    return {
        "filename": filename,
        "metrics": {
            "rmse": round(random.uniform(0.01, 0.1), 4),
            "cot": round(random.uniform(0.1, 0.5), 3),
            "latency_ms": round(random.uniform(1.0, 15.0), 2),
            "stress": round(random.uniform(0.2, 0.8), 3),
            "imu_variance": round(random.uniform(0.001, 0.05), 4)
        },
        "classification": {
            "score": round(classification_score, 3),
            "tier": policy_tier
        },
        "status": "success"
    }

@app.post("/api/upload")
async def upload_log_file(file: UploadFile = File(...)):
    """
    Endpoint to receive a robot log file and return classification results.
    Currently returns mock data since the full classification pipeline is not yet integrated.
    """
    try:
        # We would normally process the file here using the ingestion/processing pipeline
        # file_content = await file.read()
        
        results = generate_mock_results(file.filename)
        return JSONResponse(content=results)
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

# Mount the static directory to serve the frontend
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
