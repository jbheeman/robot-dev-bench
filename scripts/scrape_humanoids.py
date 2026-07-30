import os
import cv2
import yt_dlp
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_video(url, output_path):
    ydl_opts = {
        'format': 'bestvideo[vcodec^=avc1][height<=720]+bestaudio/best[vcodec^=avc1][height<=720]/bestvideo[height<=720]',
        'outtmpl': output_path,
        'merge_output_format': 'mp4',
        'quiet': True,
        'no_warnings': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def extract_frames(video_path, output_dir, prefix, frame_interval=30):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video {video_path}")
        return

    frame_count = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            out_path = os.path.join(output_dir, f"{prefix}_{saved_count:04d}.jpg")
            cv2.imwrite(out_path, frame)
            saved_count += 1
            
        frame_count += 1
        
    cap.release()
    logger.info(f"Extracted {saved_count} frames from {video_path}")

def main():
    parser = argparse.ArgumentParser(description="Scrape and extract frames from humanoid robot videos.")
    parser.add_argument("--urls", nargs="+", help="List of YouTube URLs to download", required=True)
    parser.add_argument("--output_dir", default="data/humanoid_dataset/images", help="Directory to save extracted frames")
    parser.add_argument("--interval", type=int, default=30, help="Extract 1 frame every N frames")
    
    args = parser.parse_args()
    
    temp_dir = "data/humanoid_dataset/temp_videos"
    os.makedirs(temp_dir, exist_ok=True)
    
    for i, url in enumerate(args.urls):
        logger.info(f"Downloading video {i+1}/{len(args.urls)}: {url}")
        video_path = os.path.join(temp_dir, f"video_{i}.mp4")
        try:
            download_video(url, video_path)
            # Use a hash of the URL to prevent overwriting images from previous scrapes
            safe_prefix = f"robot_{abs(hash(url)) % 100000}"
            extract_frames(video_path, args.output_dir, prefix=safe_prefix, frame_interval=args.interval)
        except Exception as e:
            logger.error(f"Failed to process {url}: {e}")
            
    # Cleanup temp videos
    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    os.rmdir(temp_dir)
    
    logger.info(f"All done! Frames saved to {args.output_dir}")

if __name__ == "__main__":
    main()
