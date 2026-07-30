import yourdfpy
import numpy as np
urdf = yourdfpy.URDF.load("src/web/static/assets/g1_description/g1_29dof.urdf")
def get_pos(link_name): return urdf.scene.graph.get(link_name)[0][:3, 3]
l_sho = get_pos('left_shoulder_yaw_link')
r_sho = get_pos('right_shoulder_yaw_link')
pelvis = get_pos('pelvis')
head = get_pos('head_link')
print("Shoulder width:", np.linalg.norm(l_sho - r_sho))
print("Height (pelvis to head):", np.linalg.norm(pelvis - head))
