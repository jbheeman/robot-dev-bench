import yourdfpy
import numpy as np

urdf = yourdfpy.URDF.load("src/web/static/assets/g1_description/g1_29dof.urdf")
urdf.update_cfg(np.zeros(len(urdf.actuated_joints)))
data = urdf.scene.graph.to_edgelist()

# Convert to dict
links = {}
for edge in data:
    node = edge[1]
    mat = urdf.scene.graph.get(node)[0]
    links[node] = {'pos': mat[:3, 3]}

SKEL_LINKS = [
    'pelvis',       # 0
    'left_hip_pitch_link', # 1
    'left_knee_link',      # 2
    'left_ankle_pitch_link',# 3
    'right_hip_pitch_link',# 4
    'right_knee_link',     # 5
    'right_ankle_pitch_link',# 6
    'torso_link',   # 7
    'torso_link', # thorax
    'torso_link', # neck
    'head_link',    # 10
    'left_shoulder_pitch_link', # 11
    'left_elbow_pitch_link', # 12
    'left_elbow_pitch_link', # left wrist (simplified)
    'right_shoulder_pitch_link',# 14
    'right_elbow_pitch_link',# 15
    'right_elbow_pitch_link' # right wrist
]

COCO_SKELETON = [
    [0, 1], [1, 2], [2, 3], # left leg
    [0, 4], [4, 5], [5, 6], # right leg
    [0, 7], [7, 8], [8, 9], [9, 10], # spine
    [8, 11], [11, 12], [12, 13], # left arm
    [8, 14], [14, 15], [15, 16] # right arm
]

print("G1 True Bone Lengths:")
for u, v in COCO_SKELETON:
    child_link = SKEL_LINKS[v]
    if child_link == SKEL_LINKS[u]:
        length = 0.0
    else:
        if child_link in links:
            # The local translation magnitude IS the bone length!
            length = np.linalg.norm(links[child_link]['pos'])
        else:
            length = 0.0
    print(f"({u}, {v}): {length:.4f}")

