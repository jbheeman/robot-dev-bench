import yourdfpy
urdf = yourdfpy.URDF.load("src/web/static/assets/g1_description/g1_29dof.urdf")
def get_pos(name): return urdf.scene.graph.get(name)[0][:3, 3]

for node in ['pelvis', 'torso_link', 'head_link']:
    print(node, get_pos(node))
