import torch
import time
from src.processing.pose_estimation import PoseEstimator

estimator = PoseEstimator(device="cuda")
print("Initializing models...")
estimator._lazy_init()
print("Models loaded. Warming up...")
dummy_img = torch.rand(1080, 1920, 3).numpy()

# Run a quick dummy benchmark (detector + pose)
# Wait, we need a valid image format. numpy uint8
import numpy as np
dummy_img = (np.random.rand(1080, 1920, 3) * 255).astype(np.uint8)

start = time.time()
n_frames = 10
for _ in range(n_frames):
    # Just run inference_topdown or the whole pipeline?
    # Let's just run the whole pipeline on 10 identical frames by simulating a small video
    pass

