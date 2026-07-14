import logging
import numpy as np
from scipy import signal
from typing import Tuple

logger = logging.getLogger(__name__)

def get_video_offset(left_path: str, right_path: str) -> float:
    """
    Computes the time offset (in seconds) between two video files by analyzing 
    their audio tracks and finding the peak cross-correlation.
    
    Returns:
        float: The time offset in seconds. 
               - If positive, left video started EARLIER (clap happens later in left video).
               - If negative, right video started EARLIER.
    """
    try:
        # Compatibility for both MoviePy 1.x and 2.x
        try:
            from moviepy.editor import VideoFileClip
        except ImportError:
            from moviepy import VideoFileClip
    except ImportError:
        logger.warning("moviepy is not installed. Returning 0.0 offset.")
        return 0.0

    logger.info("Extracting audio tracks to synchronize videos...")
    
    try:
        # Load audio tracks
        clip_l = VideoFileClip(left_path)
        clip_r = VideoFileClip(right_path)
        
        if clip_l.audio is None or clip_r.audio is None:
            logger.warning("One or both videos are missing an audio track. Cannot auto-sync. Returning 0.0 offset.")
            return 0.0
            
        # Extract the first N seconds to speed up correlation (e.g., first 60 seconds)
        fs = 44100
        dur_l = min(60.0, clip_l.duration)
        dur_r = min(60.0, clip_r.duration)
        
        # Subclip logic that works in 1.x and 2.x
        if hasattr(clip_l, "subclip"):
            audio_clip_l = clip_l.audio.subclip(0, dur_l)
            audio_clip_r = clip_r.audio.subclip(0, dur_r)
        else:
            audio_clip_l = clip_l.audio.subclipped(0, dur_l)
            audio_clip_r = clip_r.audio.subclipped(0, dur_r)

        audio_l = audio_clip_l.to_soundarray(fps=fs)
        audio_r = audio_clip_r.to_soundarray(fps=fs)
        
        clip_l.close()
        clip_r.close()
        
        # Convert stereo to mono by averaging channels
        if len(audio_l.shape) > 1:
            audio_l = audio_l.mean(axis=1)
        if len(audio_r.shape) > 1:
            audio_r = audio_r.mean(axis=1)
            
        logger.info("Computing audio cross-correlation...")
        
        # Compute cross correlation
        correlation = signal.correlate(audio_l, audio_r, mode="full")
        lags = signal.correlation_lags(audio_l.size, audio_r.size, mode="full")
        
        lag = lags[np.argmax(correlation)]
        
        # Convert lag (in samples) to time (in seconds)
        offset_seconds = lag / fs
        
        logger.info("Computed video offset: %.3f seconds", offset_seconds)
        return offset_seconds
        
    except Exception as e:
        logger.warning(f"Audio synchronization failed: {e}. Returning 0.0 offset.")
        return 0.0
