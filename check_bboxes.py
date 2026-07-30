import json
with open('data/humanoid_dataset/annotations/train.json', 'r') as f:
    d = json.load(f)

bad_count = 0
for ann in d['annotations']:
    kpts = ann['keypoints']
    bbox = ann['bbox'] # [x, y, w, h]
    bx1, by1, bw, bh = bbox
    bx2, by2 = bx1 + bw, by1 + bh
    
    outside = False
    for i in range(0, len(kpts), 3):
        x, y, v = kpts[i], kpts[i+1], kpts[i+2]
        if v > 0:
            if x < bx1 or x > bx2 or y < by1 or y > by2:
                outside = True
                break
    if outside:
        bad_count += 1

print(f"Total annotations: {len(d['annotations'])}")
print(f"Annotations with keypoints OUTSIDE bbox: {bad_count}")
