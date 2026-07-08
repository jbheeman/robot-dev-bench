import os
import random
import time
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from src.classification.rules import RuleBasedClassifier

app = FastAPI(title="Unitree G1-Edu Benchmarking Dashboard")

# Define path to static files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

# Initialize our classifier
classifier = RuleBasedClassifier()

# Mock Data Generator
def generate_mock_results(filename: str):
    # Simulate processing delay
    time.sleep(1.5)
    
    # Seed the random number generator with the filename so the same file 
    # always produces the same mock metrics and classification score.
    random.seed(filename)
    
    # Generate random metrics that could fall into any class
    metrics = {
        "rmse": round(random.uniform(0.001, 0.15), 4),
        "cot": round(random.uniform(0.05, 1.0), 3),
        "latency_ms": round(random.uniform(0.5, 30.0), 2),
        "stress": round(random.uniform(0.0, 0.8), 3),
        "imu_variance": round(random.uniform(0.0, 0.08), 4)
    }
    
    # Reset the seed so it doesn't affect other parts of the app if any
    random.seed()
    
    # Use the real classification algorithm to score the mock metrics
    score, tier = classifier.classify(metrics)
    
    return {
        "filename": filename,
        "metrics": metrics,
        "classification": {
            "score": round(score, 3),
            "tier": tier
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
