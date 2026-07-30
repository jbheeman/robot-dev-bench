import yourdfpy
import numpy as np
urdf = yourdfpy.URDF.load("src/web/static/assets/g1_description/g1_29dof.urdf")
def get_pos(link_name):
    return urdf.scene.graph.get(link_name)[0][:3, 3]

pelvis = get_pos('pelvis')
thorax = pelvis + urdf.scene.graph.get('torso_link')[0][:3, :3] @ np.array([0, 0, 0.20])
spine = (pelvis + thorax) / 2.0
head = pelvis + urdf.scene.graph.get('head_link')[0][:3, :3] @ np.array([0, 0, 0.45])
neck = thorax + urdf.scene.graph.get('torso_link')[0][:3, :3] @ np.array([0, 0, 0.15])
l_sho = get_pos('left_shoulder_yaw_link')
r_sho = get_pos('right_shoulder_yaw_link')

print("pelvis:", pelvis)
print("thorax:", thorax)
print("spine:", spine)
print("neck:", neck)
print("head:", head)
print("l_sho:", l_sho)
print("r_sho:", r_sho)
