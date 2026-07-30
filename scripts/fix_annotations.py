import json
import os

def fix_annotations():
    file_path = 'data/humanoid_dataset/annotations/train.json'
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    with open(file_path, 'r') as f:
        data = json.load(f)

    # Standard COCO keypoint names
    coco_keypoints = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]

    # Mapping from standard COCO index to the user's original index (which has swapped L/R)
    # COCO index -> User index (from original JSON)
    mapping = [
        0,   # 0: nose -> nose (0)
        3,   # 1: left_eye -> right eye (3)
        1,   # 2: right_eye -> left eye (1)
        4,   # 3: left_ear -> right ear (4)
        2,   # 4: right_ear -> left ear (2)
        6,   # 5: left_shoulder -> right shoulder (6)
        5,   # 6: right_shoulder -> left shoulder (5)
        9,   # 7: left_elbow -> right elbow (9)
        7,   # 8: right_elbow -> left elbow (7)
        10,  # 9: left_wrist -> right wrist (10)
        8,   # 10: right_wrist -> left wrist (8)
        12,  # 11: left_hip -> right hip (12)
        11,  # 12: right_hip -> left hip (11)
        14,  # 13: left_knee -> right knee (14)
        13,  # 14: right_knee -> left knee (13)
        16,  # 15: left_ankle -> right foot (16)
        15   # 16: right_ankle -> left foot (15)
    ]

    # Update categories
    data['categories'][0]['keypoints'] = coco_keypoints
    
    # Update each annotation
    for ann in data['annotations']:
        if 'keypoints' in ann and len(ann['keypoints']) == 51:
            old_kpts = ann['keypoints']
            new_kpts = []
            for user_idx in mapping:
                # Append x, y, v for this joint
                base = user_idx * 3
                new_kpts.extend(old_kpts[base:base+3])
            ann['keypoints'] = new_kpts

    # Also fix num_keypoints count if needed (usually 17, should remain same)
    
    # Save back
    with open(file_path, 'w') as f:
        json.dump(data, f)
        
    print("Successfully fixed left/right swap and standardized COCO keypoints format!")

if __name__ == '__main__':
    fix_annotations()
