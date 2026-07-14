import os
import cv2
import numpy as np
import time
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import uvicorn

# Import the tracker we already built
from src.processing.bps_calculator import SimpleTracker

app = FastAPI(title="BPS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/calculate-bps")
async def calculate_bps_endpoint(
    video: UploadFile = File(...),
    min_area: int = Form(500),
    roi_x: int = Form(...),
    roi_y: int = Form(...),
    roi_w: int = Form(...),
    roi_h: int = Form(...),
    start_time: float = Form(0.0),
    end_time: float = Form(-1.0)
):
    try:
        # Save uploaded video to temp file
        ext = os.path.splitext(video.filename)[1]
        if not ext:
            ext = ".mp4"
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_video:
            content = await video.read()
            temp_video.write(content)
            temp_video_path = temp_video.name

        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            os.remove(temp_video_path)
            raise HTTPException(status_code=400, detail="Cannot open video file")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps if fps > 0 else 0

        # Logic for determining the counting line
        if roi_w == 0 or roi_h == 0:
            h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            line_orientation = "horizontal"
            line_pos = h_frame // 2
        else:
            line_orientation = "horizontal"
            line_pos = roi_y + roi_h // 2

        lower_yellow = np.array([18, 90, 90])
        upper_yellow = np.array([45, 255, 255])
        tracker = SimpleTracker(max_distance=300)
        
        # Seek to start time if provided
        if start_time > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
            
        crossing_frames = []
        prev_positions = {}
        # Start frame count from where we seeked
        frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if end_time > 0 and current_time > end_time:
                break
                
            frame_count += 1
            
            # Apply ROI crop for optimization if we want, but since balls might enter from outside,
            # we'll just process the whole frame or crop to a slightly larger area than ROI.
            # To be safe and reuse the exact logic, we process the whole frame.
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            
            # Use an elliptical kernel which is much better for separating round balls
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            centroids = []
            for cnt in contours:
                if cv2.contourArea(cnt) > min_area:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        centroids.append((cx, cy))

            current_tracks = tracker.update(centroids)
            
            for track_id, pos in current_tracks.items():
                if track_id in prev_positions:
                    prev_pos = prev_positions[track_id]
                    if line_orientation == "vertical":
                        if (prev_pos[0] <= line_pos and pos[0] > line_pos) or (prev_pos[0] >= line_pos and pos[0] < line_pos):
                            crossing_frames.append(frame_count)
                    else:
                        if (prev_pos[1] <= line_pos and pos[1] > line_pos) or (prev_pos[1] >= line_pos and pos[1] < line_pos):
                            crossing_frames.append(frame_count)
                            
            prev_positions = current_tracks.copy()

        cap.release()
        os.remove(temp_video_path)

        total_balls = len(crossing_frames)
        avg_bps = 0
        peak_bps = 0
        active_duration = 0

        if total_balls >= 2:
            first_frame = crossing_frames[0]
            last_frame = crossing_frames[-1]
            active_duration = (last_frame - first_frame) / fps
            
            if active_duration > 0:
                avg_bps = total_balls / active_duration
            
        crossing_times = [f / fps for f in crossing_frames]
        max_balls_in_window = 0
        window_duration = 1.0
        
        for i in range(len(crossing_times)):
            start_t = crossing_times[i]
            end_t = start_t + window_duration
            
            count_in_window = 0
            for t in crossing_times[i:]:
                if t <= end_t:
                    count_in_window += 1
                else:
                    break
                    
            if count_in_window > max_balls_in_window:
                max_balls_in_window = count_in_window
        peak_bps = max_balls_in_window

        return {
            "total_balls": total_balls,
            "average_bps": round(avg_bps, 2),
            "peak_bps": peak_bps,
            "video_duration": round(active_duration, 2)
        }

    except Exception as e:
        if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("bps_api:app", host="0.0.0.0", port=8000, reload=True)
