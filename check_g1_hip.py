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

print(f"pelvis: {pelvis}")
print(f"l_hip: {l_hip}")
print(f"l_knee: {l_knee}")
print(f"l_ankle: {l_ankle}")
