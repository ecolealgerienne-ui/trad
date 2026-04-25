"""Baselines for RSI -> Oracle slope reconstruction.

Computes simple causal predictors and evaluates against y (slope_oracle):
    - identity       : yhat = 0           (no movement)
    - raw_slope      : yhat = RSI[t] - RSI[t-1]
    - ma_slope_K     : yhat = MA_K[t] - MA_K[t-1]   for K in {5, 10, 20}

Metrics per split: MSE, MAE, DirMatch (strict, non-zero sign agreement), Pearson.

Usage:
    python experiments/foundation_finetune/baselines.py
    python experiments/foundation_finetune/baselines.py --data path/to/file.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "foundation" / "rsi_btc_5min_slope.npz"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--ma-windows", nargs="+", type=int, default=[5, 10, 20])
    return p.parse_args()


def predict_identity(X):
    return np.zeros(X.shape[0], dtype=np.float32)


def predict_raw_slope(X):
    return (X[:, -1] - X[:, -2]).astype(np.float32)


def predict_ma_slope(X, K):
    """MA(t) - MA(t-1) where MA is causal mean over last K points."""
    ma_t = X[:, -K:].mean(axis=1)
    ma_tm1 = X[:, -K - 1:-1].mean(axis=1)
    return (ma_t - ma_tm1).astype(np.float32)


def metrics(yhat, y):
    yhat = yhat.astype(np.float64)
    y = y.astype(np.float64)
    mse = float(np.mean((yhat - y) ** 2))
    mae = float(np.mean(np.abs(yhat - y)))

    same_sign = (np.sign(yhat) * np.sign(y)) > 0
    nonzero = (yhat != 0) & (y != 0)
    dir_match = float(same_sign.sum() / max(nonzero.sum(), 1))

    if yhat.std() < 1e-12 or y.std() < 1e-12:
        pearson = 0.0
    else:
        pearson = float(np.corrcoef(yhat, y)[0, 1])

    return {"MSE": mse, "MAE": mae, "DirMatch": dir_match, "Pearson": pearson}


def evaluate_split(X, y, ma_windows):
    rows = []
    rows.append(("identity", metrics(predict_identity(X), y)))
    rows.append(("raw_slope", metrics(predict_raw_slope(X), y)))
    for K in ma_windows:
        if K + 1 > X.shape[1]:
            continue
        rows.append((f"ma_slope_{K}", metrics(predict_ma_slope(X, K), y)))
    return rows


def print_table(split_name, rows):
    print(f"\n=== {split_name.upper()} (n={rows[0][1]['_n']:,}) ===")
    header = f"{'baseline':<14} {'MSE':>10} {'MAE':>10} {'DirMatch':>10} {'Pearson':>10}"
    print(header)
    print("-" * len(header))
    for name, m in rows:
        print(f"{name:<14} {m['MSE']:>10.5f} {m['MAE']:>10.5f} "
              f"{m['DirMatch']:>10.4f} {m['Pearson']:>10.4f}")


def main():
    args = parse_args()
    data_path = Path(args.data)
    print(f"Loading {data_path} ...")
    data = np.load(data_path, allow_pickle=True)
    meta = json.loads(str(data["meta"]))
    print(f"Meta: rsi_period={meta['rsi_period']} window={meta['window']} "
          f"Q={meta['process_var']} R={meta['measure_var']} "
          f"n_total_rsi={meta['n_total_rsi']:,}")

    summary = {}
    for split in ("train", "val", "test"):
        X = data[f"X_{split}"]
        y = data[f"y_{split}"]
        rows = evaluate_split(X, y, args.ma_windows)
        for _, m in rows:
            m["_n"] = len(y)
        print_table(split, rows)
        summary[split] = {name: {k: v for k, v in m.items() if k != "_n"}
                          for name, m in rows}

    out_path = data_path.parent / "baselines_summary.json"
    out_path.write_text(json.dumps({"meta": meta, "summary": summary}, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
