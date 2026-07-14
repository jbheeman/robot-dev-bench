import cv2
import numpy as np
import argparse
import time
from collections import deque

class SimpleTracker:
    def __init__(self, max_distance=50):
        self.max_distance = max_distance
        self.tracks = {}  # id -> current_centroid
        self.next_id = 0

    def update(self, centroids):
        """
        centroids: list of (x,y)
        Returns a dict of id -> (x,y) for the current frame
        """
        new_tracks = {}
        unassigned_centroids = list(centroids)

        for track_id, last_centroid in self.tracks.items():
            if not unassigned_centroids:
                break
            
            # Find closest
            distances = [np.linalg.norm(np.array(last_centroid) - np.array(c)) for c in unassigned_centroids]
            min_dist_idx = np.argmin(distances)
            min_dist = distances[min_dist_idx]

            if min_dist < self.max_distance:
                new_tracks[track_id] = unassigned_centroids[min_dist_idx]
                unassigned_centroids.pop(min_dist_idx)

        # Assign new IDs to remaining
        for c in unassigned_centroids:
            new_tracks[self.next_id] = c
            self.next_id += 1

        self.tracks = new_tracks
        return self.tracks

def main():
    parser = argparse.ArgumentParser(description="Calculate Balls Per Second (BPS) for FRC robot.")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("--min-area", type=int, default=500, help="Minimum contour area to be considered a ball")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {args.video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if fps > 0 else 0

    # Read first frame to select ROI
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return

    print("Please select the Region of Interest (ROI) where balls pass through.")
    print("Press ENTER or SPACE to confirm the ROI. Press 'c' to cancel and use the whole frame.")
    roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")

    # Determine counting line based on ROI shape
    # If the ROI is wider than tall, we use a vertical counting line (assuming horizontal movement).
    # If the ROI is taller than wide, we use a horizontal counting line (assuming vertical movement).
    x, y, w, h = roi
    if w == 0 or h == 0:
        # Default to vertical line in the middle of the frame
        h_frame, w_frame, _ = frame.shape
        line_orientation = "vertical"
        line_pos = w_frame // 2
        print(f"No ROI selected. Using default vertical line at x={line_pos}.")
    elif w > h:
        line_orientation = "vertical"
        line_pos = x + w // 2
        print(f"ROI is wider than tall. Using vertical counting line at x={line_pos}.")
    else:
        line_orientation = "horizontal"
        line_pos = y + h // 2
        print(f"ROI is taller than wide. Using horizontal counting line at y={line_pos}.")

    # "Hella yellow" HSV bounds
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])

    tracker = SimpleTracker(max_distance=100)
    
    # Store crossing events (frame numbers)
    crossing_frames = []
    
    # To keep track of previous positions for crossing detection
    prev_positions = {}

    frame_count = 0
    start_time = time.time()
    
    print("Processing video for max speed...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Apply HSV thresholding
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        centroids = []
        for cnt in contours:
            if cv2.contourArea(cnt) > args.min_area:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centroids.append((cx, cy))

        # Update tracker
        current_tracks = tracker.update(centroids)
        
        # Check for line crossings
        for track_id, pos in current_tracks.items():
            if track_id in prev_positions:
                prev_pos = prev_positions[track_id]
                
                # Check crossing
                if line_orientation == "vertical":
                    # Crossed from left to right or right to left
                    if (prev_pos[0] <= line_pos and pos[0] > line_pos) or (prev_pos[0] >= line_pos and pos[0] < line_pos):
                        crossing_frames.append(frame_count)
                else: # horizontal
                    # Crossed from top to bottom or bottom to top
                    if (prev_pos[1] <= line_pos and pos[1] > line_pos) or (prev_pos[1] >= line_pos and pos[1] < line_pos):
                        crossing_frames.append(frame_count)
                        
        prev_positions = current_tracks.copy()
        
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames...")

    end_time = time.time()
    cap.release()

    total_balls = len(crossing_frames)
    
    print("\n--- BPS Calculation Results ---")
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    print(f"Total Balls Counted: {total_balls}")
    print(f"Video Duration: {video_duration:.2f} seconds")
    
    if total_balls > 0 and video_duration > 0:
        avg_bps = total_balls / video_duration
        print(f"Average BPS: {avg_bps:.2f} balls/sec")
        
        # Calculate Peak BPS over a 1-second rolling window
        # Convert crossing frames to timestamps
        crossing_times = [f / fps for f in crossing_frames]
        
        max_balls_in_window = 0
        window_duration = 1.0 # 1 second
        
        for i in range(len(crossing_times)):
            start_t = crossing_times[i]
            end_t = start_t + window_duration
            
            # Count balls in this 1s window
            count_in_window = 0
            for t in crossing_times[i:]:
                if t <= end_t:
                    count_in_window += 1
                else:
                    break
                    
            if count_in_window > max_balls_in_window:
                max_balls_in_window = count_in_window
                
        print(f"Peak BPS (1-second window): {max_balls_in_window} balls/sec")
    else:
        print("No balls were counted or video duration is zero.")

if __name__ == "__main__":
    main()
