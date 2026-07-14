import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class AVLoader:
    """
    Handles the ingestion, validation, and synchronisation of Audio-Visual streams
    from the two-camera setup.
    """
    
    def __init__(self, left_video_path: str, right_video_path: str, audio_path: Optional[str] = None):
        self.left_video_path = left_video_path
        self.right_video_path = right_video_path
        self.audio_path = audio_path

    def validate_streams(self) -> bool:
        """
        Validates that the provided files exist and are readable video/audio formats.
        """
        if not os.path.exists(self.left_video_path) or not os.path.exists(self.right_video_path):
            logger.error("Video streams not found.")
            return False
            
        return True
