#!/usr/bin/env python
import numpy as np
import sys

if len(sys.argv) < 2:
    print("Usage: python check_npz_keys.py <fichier.npz>")
    sys.exit(1)

path = sys.argv[1]
data = np.load(path, allow_pickle=True)

print(f"\n📦 Clés dans {path}:\n")
for key in sorted(data.keys()):
    shape = data[key].shape if hasattr(data[key], 'shape') else 'N/A'
    print(f"  {key}: {shape}")
print()
