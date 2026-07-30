import numpy as np

with open('/home/andrew/Downloads/archive.zip', 'rb') as f:
    import zipfile
    with zipfile.ZipFile(f) as z:
        with z.open('h3.6m/dataset/S1/directions_1.txt') as txt:
            lines = [txt.readline().decode('utf-8').strip() for _ in range(5)]
            
for i, line in enumerate(lines):
    vals = np.array([float(x) for x in line.split(',')])
    non_zeros = []
    for j in range(0, 99, 3):
        triplet = vals[j:j+3]
        if np.any(np.abs(triplet) > 1e-5):
            non_zeros.append(f"J{j//3}: {triplet}")
    print(f"Line {i} non-zero joints: {non_zeros}")
