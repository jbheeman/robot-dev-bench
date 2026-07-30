import yourdfpy
import numpy as np
urdf = yourdfpy.URDF.load("src/web/static/assets/g1_description/g1_29dof.urdf")
def get_pos(link_name): return urdf.scene.graph.get(link_name)[0][:3, 3]

print("l_hip_pitch:", get_pos('left_hip_pitch_link'))
print("l_hip_roll:", get_pos('left_hip_roll_link'))
print("l_hip_yaw:", get_pos('left_hip_yaw_link'))
