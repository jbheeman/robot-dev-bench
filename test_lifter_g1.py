import sys, os
sys.path.insert(0, os.path.abspath('src'))
from processing.lifter import Lifter
import numpy as np
lifter = Lifter(checkpoint='checkpoints/motionagformer-s-g1.pth')
win = np.zeros((1, 81, 17, 3), dtype=np.float32)
win[..., 2] = 1.0
win[..., 10, 1] = 0.5
pose = lifter.lift(win)
print("G1 model pose head:", pose[10])
print("G1 model pose pelvis:", pose[0])
print("G1 model pose l_shoulder:", pose[5])
