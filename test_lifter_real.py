import sys, os
sys.path.insert(0, os.path.abspath('src'))
from processing.lifter import Lifter
import numpy as np
import pickle

with open('third_party/MotionAGFormer/data/motion3d/g1_synthetic.pkl', 'rb') as f:
    d = pickle.load(f)

# Get a real sequence from test set
pts2d = d['test']['joint_2d'][0:81] # (81, 17, 3)
pts2d[:, :, :2] = pts2d[:, :, :2] / 500.0 - 1.0 # normalize to [-1, 1]

lifter = Lifter(checkpoint='checkpoints/motionagformer-s-g1.pth')
win = pts2d[None, ...] # (1, 81, 17, 3)
pose = lifter.lift(win)

print("Real input G1 model pose head:", pose[10])
print("Real input G1 model pose pelvis:", pose[0])
print("Real input G1 model pose l_shoulder:", pose[5])
