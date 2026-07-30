import pickle
import numpy as np

with open('data/unitree_g1/g1.xml', 'r') as f:
    xml = f.read()

import re
lines = xml.split('\n')
for i, line in enumerate(lines):
    if 'pos=' in line and 'body' in line:
        pass # Not easy to parse xml via regex.

