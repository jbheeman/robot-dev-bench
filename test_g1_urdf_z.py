import yourdfpy
urdf = yourdfpy.URDF.load("src/web/static/assets/g1_description/g1_29dof.urdf")
def get_pos(name): return urdf.scene.graph.get(name)[0][:3, 3]

print("Pelvis Z:", get_pos('pelvis')[2])
print("L_SHO Z:", get_pos('left_shoulder_pitch_link')[2])
print("R_SHO Z:", get_pos('right_shoulder_pitch_link')[2])
