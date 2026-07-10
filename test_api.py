import pandas as pd
from fastapi.testclient import TestClient
from src.web.app import app
client = TestClient(app)
with open('/home/andrew/Downloads/episode_000000.parquet', 'rb') as f:
    response = client.post('/api/upload', files={'file': f}, data={'task':'walking'})
    print(response.status_code)
    print(response.text)
