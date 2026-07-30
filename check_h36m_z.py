import torch
import numpy as np
from src.processing.lifter import Lifter

lifter = Lifter()
lifter.load_checkpoint("third_party/MotionAGFormer/checkpoints/motionagformer-s-h36m.pth.tr")

# Dummy input
window = np.zeros((1, 27, 17, 3), dtype=np.float32)
# We can't really get a meaningful Z from dummy input, it's untrained for zeros.
