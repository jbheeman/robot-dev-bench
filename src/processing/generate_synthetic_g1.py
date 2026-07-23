import numpy as np
import yourdfpy
import pickle
import os
import tqdm
import zipfile
from scipy.spatial.transform import Rotation

def get_h36m_files(zip_path, subset, max_files=20):
    """Get a list of txt files from the archive for a given subset of subjects."""
    files = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        all_files = z.namelist()
        for s in subset:
            s_files = [f for f in all_files if f.startswith(f"h3.6m/dataset/S{s}/") and f.endswith(".txt")]
            files.extend(s_files)
    
    np.random.shuffle(files)
    return files[:max_files]

def generate_dataset():
    urdf = yourdfpy.URDF.load("src/web/static/assets/g1_description/g1_29dof.urdf")
    zip_path = "/home/andrew/Downloads/archive.zip"
    
    train_subjects = [1, 5, 6, 7, 8]
    test_subjects = [9, 11]
    
    train_files = get_h36m_files(zip_path, train_subjects, max_files=100)
    test_files = get_h36m_files(zip_path, test_subjects, max_files=20)
    
    print(f"Found {len(train_files)} train files and {len(test_files)} test files.")
    
    H36M_IDX = {
        'RHip': 2, 'RKnee': 3, 'RFoot': 4,
        'LHip': 7, 'LKnee': 8, 'LFoot': 9,
        'Spine': 12, 'LShoulder': 17, 'LElbow': 19,
        'RShoulder': 25, 'RElbow': 27
    }
    
    def process_files(files, is_train):
        poses_3d = []
        poses_2d = []
        source_names = []
        camera_names = []
        
        cameras = [
            {'name': '54138969', 'az': 0},
            {'name': '55011271', 'az': np.pi/2},
            {'name': '58860488', 'az': np.pi},
            {'name': '60457274', 'az': 3*np.pi/2}
        ]
        
        with zipfile.ZipFile(zip_path, 'r') as z:
            for fname in tqdm.tqdm(files, desc="Processing files"):
                source = fname.split('/')[-1].replace('.txt', '')
                with z.open(fname) as f:
                    lines = f.read().decode('utf-8').strip().split('\n')
                
                lines = lines[::5]
                chunk_size = 81
                if len(lines) < chunk_size: continue
                n_chunks = len(lines) // chunk_size
                lines = lines[:n_chunks * chunk_size]
                
                cam = np.random.choice(cameras)
                cam_dist = 2.5
                cam_height = 1.0
                cam_pos = np.array([cam_dist * np.cos(cam['az']), cam_dist * np.sin(cam['az']), cam_height])
                
                z_axis = -cam_pos / np.linalg.norm(cam_pos)
                x_axis = np.cross(z_axis, np.array([0, 0, 1]))
                x_axis = x_axis / np.linalg.norm(x_axis)
                y_axis = np.cross(z_axis, x_axis)
                R_cam = np.row_stack([x_axis, y_axis, z_axis])
                
                for line in lines:
                    vals = np.array([float(x) for x in line.split(',')])
                    cfg = {}
                    
                    def set_joint(h36m_name, urdf_pitch, urdf_roll=None, urdf_yaw=None, hinge=False):
                        idx = H36M_IDX[h36m_name]
                        rotvec = vals[idx*3 : idx*3+3]
                        if hinge:
                            cfg[urdf_pitch] = rotvec[0]
                        else:
                            euler = Rotation.from_rotvec(rotvec).as_euler('xyz')
                            if urdf_pitch: cfg[urdf_pitch] = euler[0]
                            if urdf_roll: cfg[urdf_roll] = euler[1]
                            if urdf_yaw: cfg[urdf_yaw] = euler[2]
                            
                    set_joint('RHip', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint')
                    set_joint('RKnee', 'right_knee_joint', hinge=True)
                    set_joint('RFoot', 'right_ankle_pitch_joint', 'right_ankle_roll_joint')
                    
                    set_joint('LHip', 'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint')
                    set_joint('LKnee', 'left_knee_joint', hinge=True)
                    set_joint('LFoot', 'left_ankle_pitch_joint', 'left_ankle_roll_joint')
                    
                    set_joint('Spine', 'waist_pitch_joint', 'waist_roll_joint', 'waist_yaw_joint')
                    
                    set_joint('RShoulder', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint')
                    set_joint('RElbow', 'right_elbow_joint', hinge=True)
                    
                    set_joint('LShoulder', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint')
                    set_joint('LElbow', 'left_elbow_joint', hinge=True)
                    
                    for j_name, angle in cfg.items():
                        if j_name in urdf.joint_map:
                            limit = urdf.joint_map[j_name].limit
                            if limit:
                                lower = limit.lower if limit.lower is not None else -3.14
                                upper = limit.upper if limit.upper is not None else 3.14
                                cfg[j_name] = np.clip(angle, lower, upper)
                                
                    urdf.update_cfg(cfg)
                    
                    def get_pos(link_name): return urdf.scene.graph.get(link_name)[0][:3, 3]
                    
                    pelvis = get_pos('pelvis')
                    r_hip = get_pos('right_hip_pitch_link')
                    r_knee = get_pos('right_knee_link')
                    r_ank = get_pos('right_ankle_roll_link')
                    l_hip = get_pos('left_hip_pitch_link')
                    l_knee = get_pos('left_knee_link')
                    l_ank = get_pos('left_ankle_roll_link')
                    l_sho = get_pos('left_shoulder_yaw_link')
                    r_sho = get_pos('right_shoulder_yaw_link')
                    thorax = (l_sho + r_sho) / 2.0
                    spine = (pelvis + thorax) / 2.0
                    head = pelvis + urdf.scene.graph.get('head_link')[0][:3, :3] @ np.array([0, 0, 0.45])
                    neck = thorax + (head - thorax) * 0.4
                    l_elb = get_pos('left_elbow_link')
                    l_wri = get_pos('left_rubber_hand')
                    r_elb = get_pos('right_elbow_link')
                    r_wri = get_pos('right_rubber_hand')
                    
                    pts3d = np.stack([
                        pelvis, r_hip, r_knee, r_ank, l_hip, l_knee, l_ank,
                        spine, thorax, neck, head,
                        l_sho, l_elb, l_wri, r_sho, r_elb, r_wri
                    ])
                    
                    pelvis_rotvec = vals[3:6]
                    pelvis_rot = Rotation.from_rotvec(pelvis_rotvec).as_matrix()
                    pts3d = pts3d @ pelvis_rot.T
                    
                    pts_cam = (pts3d - cam_pos) @ R_cam.T
                    pts2d = (pts_cam[:, :2] / pts_cam[:, 2:3]) * 1000 + 500
                    pts2d += np.random.normal(0, 1.5, pts2d.shape)
                    pts2d_conf = np.hstack([pts2d, np.ones((17, 1), dtype=np.float32)])
                    
                    j3d = np.zeros_like(pts_cam)
                    j3d[:, 0] = pts2d[:, 0]
                    j3d[:, 1] = pts2d[:, 1]
                    j3d[:, 2] = pts_cam[:, 2] * 1000.0  # meters to millimeters
                    
                    poses_3d.append(j3d)
                    poses_2d.append(pts2d_conf)
                    source_names.append(source)
                    camera_names.append(cam['name'])
                    
        return np.array(poses_3d, dtype=np.float32), np.array(poses_2d, dtype=np.float32), source_names, camera_names
        
    print("Generating training data...")
    train_3d, train_2d, train_src, train_cam = process_files(train_files, is_train=True)
    print("Generating testing data...")
    test_3d, test_2d, test_src, test_cam = process_files(test_files, is_train=False)
    
    out_data = {
        'train': {
            'joint_2d': train_2d, 'confidence': train_2d[:, :, 2],
            'joint3d_image': train_3d, 'camera_name': train_cam, 'source': train_src
        },
        'test': {
            'joint_2d': test_2d, 'confidence': test_2d[:, :, 2],
            'joint3d_image': test_3d, 'camera_name': test_cam, 'source': test_src
        }
    }
    
    os.makedirs("third_party/MotionAGFormer/data/motion3d", exist_ok=True)
    with open("third_party/MotionAGFormer/data/motion3d/g1_synthetic.pkl", "wb") as f:
        pickle.dump(out_data, f)
    print("Saved to third_party/MotionAGFormer/data/motion3d/g1_synthetic.pkl")

if __name__ == '__main__':
    generate_dataset()
