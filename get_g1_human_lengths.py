import yourdfpy
import numpy as np

urdf = yourdfpy.URDF.load("src/web/static/assets/g1_description/g1_29dof.urdf")
urdf.update_cfg(np.zeros(len(urdf.actuated_joints)))
data = urdf.scene.graph.to_edgelist()

links = {}
for edge in data:
    node = edge[1]
    mat = urdf.scene.graph.get(node)[0]
    links[node] = {'pos': mat[:3, 3]}

pelvis = np.array([0.0, 0.0, 0.0])
l_hip = links['left_hip_pitch_link']['pos']
l_knee = links['left_knee_link']['pos']
l_ankle = links['left_ankle_pitch_link']['pos']

l_sho = links['left_shoulder_pitch_link']['pos']
l_elb = links['left_elbow_link']['pos']

head = links['head_link']['pos']

# Derived human-like joints
shoulder_midpoint = np.copy(l_sho)
shoulder_midpoint[1] = 0.0 # set lateral Y to 0

# Calculate lengths
pelvis_to_hip = np.linalg.norm(l_hip - pelvis)
hip_to_knee = np.linalg.norm(l_knee - l_hip)
knee_to_ankle = np.linalg.norm(l_ankle - l_knee)

spine_total = np.linalg.norm(shoulder_midpoint - pelvis)
spine_len = spine_total / 2.0
thorax_to_sho = np.linalg.norm(l_sho - shoulder_midpoint)

sho_to_elb = np.linalg.norm(l_elb - l_sho)

thorax_to_neck = np.linalg.norm(head - shoulder_midpoint) / 2.0
neck_to_head = thorax_to_neck

print(f"pelvis_to_hip: {pelvis_to_hip:.4f}")
print(f"hip_to_knee: {hip_to_knee:.4f}")
print(f"knee_to_ankle: {knee_to_ankle:.4f}")
print(f"spine (each half): {spine_len:.4f}")
print(f"thorax_to_sho: {thorax_to_sho:.4f}")
print(f"sho_to_elb: {sho_to_elb:.4f}")
print(f"neck (each half): {neck_to_head:.4f}")
