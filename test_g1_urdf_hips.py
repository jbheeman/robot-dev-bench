import yourdfpy
import numpy as np
urdf = yourdfpy.URDF.load("src/web/static/assets/g1_description/g1_29dof.urdf")
def get_pos(link_name):
    return urdf.scene.graph.get(link_name)[0][:3, 3]

print("l_hip:", get_pos('left_hip_yaw_link'))
print("l_knee:", get_pos('left_knee_link'))
print("l_ank:", get_pos('left_ankle_roll_link'))
