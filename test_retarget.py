import numpy as np
import yourdfpy
from scipy.spatial.transform import Rotation
import zipfile
import pickle
import os
import tqdm

def convert_h36m_to_g1():
    urdf = yourdfpy.URDF.load("src/web/static/assets/g1_description/g1_29dof.urdf")
    
    # We will sample 10 files for test
    files = ['h3.6m/dataset/S1/walking_1.txt']
    
    with zipfile.ZipFile('/home/andrew/Downloads/archive.zip') as z:
        for fname in files:
            with z.open(fname) as txt:
                lines = [txt.readline().decode('utf-8').strip() for _ in range(5)]
                
    for i, line in enumerate(lines):
        vals = np.array([float(x) for x in line.split(',')])
        
        # J2 is Right Hip
        r_hip_rotvec = vals[2*3 : 2*3+3]
        print("RHip rotvec:", r_hip_rotvec)
        r_hip_euler = Rotation.from_rotvec(r_hip_rotvec).as_euler('xyz')
        print("RHip euler:", r_hip_euler)
        
convert_h36m_to_g1()
