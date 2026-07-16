import urllib.request
import cv2
import json

# Download image
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/62/Unitree_G1_Humanoid_Robot.jpg/800px-Unitree_G1_Humanoid_Robot.jpg"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with open('unitree_g1.jpg', 'wb') as f:
    f.write(urllib.request.urlopen(req).read())

from src.processing.pose_estimation import PoseEstimator
print("Running inference...")
estimator = PoseEstimator(device="cuda:0")

frame = cv2.imread('unitree_g1.jpg')
if frame is None:
    print("Failed to read image")
    exit(1)

result = estimator.estimate_from_video("unitree_g1.jpg", max_frames=1)
if len(result.keypoints) == 0:
    print("No poses detected.")
else:
    kpts = result.keypoints[0]
    print(f"Detected {len(kpts)} keypoints.")
    # Draw skeleton
    import math
    for pt in kpts:
        x, y = int(pt[0]), int(pt[1])
        if x > 0 and y > 0:
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    
    cv2.imwrite("g1_pose_output.jpg", frame)
    print("Saved output to g1_pose_output.jpg")
