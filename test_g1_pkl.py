import pickle
with open('third_party/MotionAGFormer/data/motion3d/g1_synthetic.pkl', 'rb') as f:
    d = pickle.load(f)
print("joint3d_image sample (head joint, 1st frame):")
print(d['train']['joint3d_image'][0, 10])
