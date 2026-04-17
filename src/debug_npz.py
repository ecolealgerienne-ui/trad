import numpy as np
data = np.load('data/prepared/macd_30m_dataset.npz', allow_pickle=True)
print("Keys:", list(data.keys()))
for k in data.keys():
    v = data[k]
    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    if v.dtype in (np.float32, np.float64):
        valid = ~np.isnan(v)
        print(f"    valid={valid.sum()}, NaN={(~valid).sum()}, min={v[valid].min():.2f}, max={v[valid].max():.2f}")
    elif 'date' in k:
        print(f"    first={v[0]}, last={v[-1]}")
    else:
        print(f"    first 5: {v[:5]}")
