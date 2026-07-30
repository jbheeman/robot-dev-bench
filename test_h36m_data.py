import pickle
with open('third_party/MotionAGFormer/data/motion3d/h36m_cpn_cam_source.pkl', 'rb') as f:
    d = pickle.load(f)
print("joint3d_image sample:")
print(d['train']['joint3d_image'][0, 0:3])
