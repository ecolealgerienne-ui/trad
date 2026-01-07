#!/usr/bin/env python3
import numpy as np
import sys

file_path = sys.argv[1] if len(sys.argv) > 1 else 'data/prepared/dataset_btc_eth_bnb_ada_ltc_macd_dual_binary_kalman.npz'

data = np.load(file_path, allow_pickle=True)
print(f'Fichier: {file_path}')
print(f'\nClés disponibles ({len(data.keys())} total):')
for key in sorted(data.keys()):
    arr = data[key]
    if hasattr(arr, 'shape'):
        print(f'  {key:<30} shape={arr.shape}, dtype={arr.dtype}')
    else:
        print(f'  {key:<30} type={type(arr)}')
