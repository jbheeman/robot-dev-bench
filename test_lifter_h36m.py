import sys, os
sys.path.insert(0, os.path.abspath('src'))
from processing.lifter import Lifter
import numpy as np
lifter = Lifter(checkpoint='checkpoints/motionagformer-s-h36m.pth.tr')
win = np.zeros((1, 81, 17, 3), dtype=np.float32)
win[..., 2] = 1.0
win[..., 10, 1] = 0.5
pose = lifter.lift(win)
print("H36M model head Z:", pose[10])
