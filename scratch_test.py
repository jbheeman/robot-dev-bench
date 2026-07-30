import numpy as np, zipfile, io, yourdfpy
from scipy.spatial.transform import Rotation
z = zipfile.ZipFile('/home/andrew/Downloads/archive.zip')
lines = io.TextIOWrapper(z.open('h3.6m/dataset/S1/walking_1.txt')).readlines()
vals = np.array([float(x) for x in lines[500].split(',')])

urdf = yourdfpy.URDF.load('src/web/static/assets/g1_description/g1_29dof.urdf')
cfg = {}

H36M_IDX = {
    'RHip': 2, 'RKnee': 3, 'RFoot': 4,
    'LHip': 7, 'LKnee': 8, 'LFoot': 9,
}

def set_joint(h36m_name, urdf_pitch, urdf_roll=None, urdf_yaw=None, hinge=False):
    idx = H36M_IDX[h36m_name]
    # offset by 3 (skip global translation), then local joints start at idx*3
    rotvec = vals[3 + idx*3 : 3 + idx*3+3]
    if hinge:
        cfg[urdf_pitch] = rotvec[0]
    else:
        euler = Rotation.from_rotvec(rotvec).as_euler('xyz')
        if urdf_pitch: cfg[urdf_pitch] = euler[0]
        if urdf_roll: cfg[urdf_roll] = euler[1]
        if urdf_yaw: cfg[urdf_yaw] = euler[2]

set_joint('RHip', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint')
set_joint('RKnee', 'right_knee_joint', hinge=True)
set_joint('LHip', 'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint')
set_joint('LKnee', 'left_knee_joint', hinge=True)

urdf.update_cfg(cfg)
def get_pos(name): return urdf.scene.graph.get(name)[0][:3, 3]

pelvis = get_pos('pelvis')
r_hip = get_pos('right_hip_pitch_link')
r_knee = get_pos('right_knee_link')
r_ank = get_pos('right_ankle_roll_link')

pts3d = np.stack([pelvis, r_hip, r_knee, r_ank])

# Pelvis global rotation is at vals[3:6]
# But H3.6M uses different global axes! 
pelvis_rotvec = vals[3:6]
pelvis_rot = Rotation.from_rotvec(pelvis_rotvec).as_matrix()
pts3d = pts3d @ pelvis_rot.T

print("Pelvis:", pts3d[0])
print("RHip:", pts3d[1])
print("RKnee:", pts3d[2])
print("RAnkle:", pts3d[3])
print("Knee relative to Hip:", pts3d[2] - pts3d[1])
print("Ankle relative to Knee:", pts3d[3] - pts3d[2])
