"""Build meta-classifier dataset : Triple Barrier labels + features.

Étape 1 + 2 du plan meta-labeling pour détection de transitions à haute précision.

Pipeline :
  1. Pour chaque indicateur (RSI, MACD, CCI) :
     - Charger dataset npz (val + test)
     - Charger checkpoint et générer prédictions yhat sur val + test
  2. Charger CSV brut BTC pour récupérer OHLCV → calcul volume_ratio, atr_norm, etc.
     Alignement par timestamp avec val/test
  3. Construire les features par timestep :
     PRIMAIRES (par modèle) :
       - yhat_{rsi,macd,cci} (continuous prediction)
       - mag_{rsi,macd,cci} = |yhat|
       - accel_{rsi,macd,cci} = yhat[t] - yhat[t-1]
     DÉRIVÉES (cross-modèle) :
       - confidence_spread = max(|yhat|) - min(|yhat|)
       - confidence_min, confidence_mean, confidence_max
       - n_models_agree_direction (compte de modèles d'accord sur direction majoritaire)
       - mean_direction = sign(mean(yhat))
       - time_since_last_flip (sur mean_direction)
     ORTHOGONALES (vraiment indépendantes du close) :
       - volume_ratio = volume[t] / MA(volume, 20)
       - volume_spike (binaire, volume > 2 * MA)
       - atr_normalized = ATR[t] / close[t]
     RÉGIME (déjà dans dataset) :
       - regime (0/1/2 catégoriel)
       - ts_score, vc_score (continus 0-1)
  4. Calculer Triple Barrier labels (López de Prado) :
     Pour chaque timestep t avec direction = mean_direction[t] :
       - si direction = +1 (long) : label = 1 si close hit +TP avant -SL dans T bars
       - si direction = -1 (short) : label = 1 si close hit -TP avant +SL dans T bars
       - sinon label = 0 (timeout ou SL touché)
  5. Save :
     - X_meta_train (features sur split val), y_meta_train (TB labels)
     - X_meta_test  (features sur split test), y_meta_test
     - direction_train, direction_test (pour analyse)
     - regime/ts/vc per split (stratification)
     - timestamps per split (traçabilité)

Réutilise :
  - load_model, predict_model de evaluate.py
  - calculate_atr_normalized, calculate_volume_ratio, calculate_volume_spike de regime_features.py
  - Datasets existants data/foundation/{rsi,macd,cci}_btc_close_kalman_5min.npz

Usage :
    python experiments/foundation_finetune/build_meta_dataset.py \\
        --rsi-data  data/foundation/rsi_btc_close_kalman_5min.npz \\
        --macd-data data/foundation/macd_btc_close_kalman_5min.npz \\
        --cci-data  data/foundation/cci_btc_close_kalman_5min.npz \\
        --rsi-ckpt  models/specialist_rsi/chronos-t5-tiny_lora.pt \\
        --macd-ckpt models/specialist_macd/chronos-t5-tiny_lora.pt \\
        --cci-ckpt  models/specialist_cci/chronos-t5-tiny_lora.pt \\
        --csv-file  data_trad/BTCUSD_all_5m.csv \\
        --tp 0.005 --sl 0.002 --t-max 6 \\
        --output    data/foundation/meta_btc_close_kalman.npz
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Bridge vers src/ + même dossier (pour evaluate.py)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_utils import load_crypto_data  # noqa: E402
from regime_features import (  # noqa: E402
    calculate_atr_normalized,
    calculate_volume_ratio,
    calculate_volume_spike,
)
from evaluate import load_model, predict_model  # noqa: E402


# =============================================================================
# I/O HELPERS
# =============================================================================

def load_dataset(path: str):
    data = np.load(path, allow_pickle=True)
    meta_key = "meta" if "meta" in data.files else "metadata"
    meta = json.loads(str(data[meta_key]))
    return data, meta


def predict_split(ckpt_path: str, X: np.ndarray, y_mean: float, y_std: float,
                  device: str, batch_size: int, num_workers: int) -> np.ndarray:
    """Charge model, prédit, dénormalise vers slope brute."""
    model, _ = load_model(Path(ckpt_path), device)
    yhat_norm = predict_model(model, X, None, device, batch_size, num_workers)
    yhat_raw = yhat_norm * y_std + y_mean
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return yhat_raw.astype(np.float64)


# =============================================================================
# TRIPLE BARRIER LABELS
# =============================================================================

def triple_barrier_label(close: np.ndarray, direction: np.ndarray,
                          tp: float, sl: float, t_max: int) -> np.ndarray:
    """Labels Triple Barrier (López de Prado).

    Pour chaque timestep t avec direction d ∈ {-1, 0, +1} :
      - d = +1 : label = 1 si close hit close[t]*(1+tp) avant close[t]*(1-sl) dans t_max bars
      - d = -1 : label = 1 si close hit close[t]*(1-tp) avant close[t]*(1+sl) dans t_max bars
      - d = 0  : label = 0 (pas de trade)
      - timeout ou SL hit en premier : label = 0

    Args:
        close: prix close (n,)
        direction: direction proposée à chaque timestep (n,) en {-1,0,+1}
        tp: take-profit threshold (ex. 0.005 = 0.5%)
        sl: stop-loss threshold (ex. 0.002 = 0.2%)
        t_max: horizon maximum en bars

    Returns:
        labels (n,) en {0, 1}
    """
    n = len(close)
    labels = np.zeros(n, dtype=np.int8)

    for t in range(n - 1):
        d = direction[t]
        if d == 0:
            continue
        c_t = close[t]
        if d > 0:
            tp_level = c_t * (1.0 + tp)
            sl_level = c_t * (1.0 - sl)
        else:
            tp_level = c_t * (1.0 - tp)
            sl_level = c_t * (1.0 + sl)

        end = min(t + 1 + t_max, n)
        for k in range(t + 1, end):
            c_k = close[k]
            if d > 0:
                if c_k >= tp_level:
                    labels[t] = 1
                    break
                if c_k <= sl_level:
                    break
            else:
                if c_k <= tp_level:
                    labels[t] = 1
                    break
                if c_k >= sl_level:
                    break
    return labels


# =============================================================================
# FEATURES ENGINEERING
# =============================================================================

def compute_features(yhat_rsi: np.ndarray, yhat_macd: np.ndarray, yhat_cci: np.ndarray,
                      regime: np.ndarray, ts_score: np.ndarray, vc_score: np.ndarray,
                      volume_ratio: np.ndarray, volume_spike: np.ndarray,
                      atr_norm: np.ndarray) -> tuple:
    """Retourne (X_features, feature_names)."""
    yhat = np.stack([yhat_rsi, yhat_macd, yhat_cci], axis=1)  # (n, 3)
    mag = np.abs(yhat)  # (n, 3)
    n = len(yhat_rsi)

    # Acceleration : yhat[t] - yhat[t-1]
    accel = np.zeros_like(yhat)
    accel[1:] = yhat[1:] - yhat[:-1]

    # Cross-modèle
    confidence_spread = mag.max(axis=1) - mag.min(axis=1)  # (n,)
    confidence_min = mag.min(axis=1)
    confidence_mean = mag.mean(axis=1)
    confidence_max = mag.max(axis=1)

    # Direction par modèle, et direction moyenne (vote)
    sign_yhat = np.sign(yhat)  # (n, 3) en {-1, 0, +1}
    mean_yhat = yhat.mean(axis=1)
    mean_direction = np.sign(mean_yhat).astype(np.int8)  # (n,)

    # n_models_agree avec la direction majoritaire
    n_agree = (sign_yhat == mean_direction[:, None]).sum(axis=1).astype(np.int8)  # (n,) en 0..3

    # time_since_last_flip de mean_direction
    time_since_flip = np.zeros(n, dtype=np.int32)
    for t in range(1, n):
        if mean_direction[t] != mean_direction[t - 1]:
            time_since_flip[t] = 0
        else:
            time_since_flip[t] = time_since_flip[t - 1] + 1

    # One-hot regime
    regime_0 = (regime == 0).astype(np.float32)
    regime_1 = (regime == 1).astype(np.float32)
    regime_2 = (regime == 2).astype(np.float32)

    feats = {
        # Primaires per-modèle
        "yhat_rsi": yhat_rsi.astype(np.float32),
        "yhat_macd": yhat_macd.astype(np.float32),
        "yhat_cci": yhat_cci.astype(np.float32),
        "mag_rsi": mag[:, 0].astype(np.float32),
        "mag_macd": mag[:, 1].astype(np.float32),
        "mag_cci": mag[:, 2].astype(np.float32),
        "accel_rsi": accel[:, 0].astype(np.float32),
        "accel_macd": accel[:, 1].astype(np.float32),
        "accel_cci": accel[:, 2].astype(np.float32),
        # Dérivées cross-modèle
        "confidence_spread": confidence_spread.astype(np.float32),
        "confidence_min": confidence_min.astype(np.float32),
        "confidence_mean": confidence_mean.astype(np.float32),
        "confidence_max": confidence_max.astype(np.float32),
        "n_models_agree": n_agree.astype(np.float32),
        "time_since_flip": time_since_flip.astype(np.float32),
        # Orthogonales (vraiment indépendantes)
        "volume_ratio": volume_ratio.astype(np.float32),
        "volume_spike": volume_spike.astype(np.float32),
        "atr_normalized": atr_norm.astype(np.float32),
        # Régime (continus + one-hot)
        "ts_score": ts_score.astype(np.float32),
        "vc_score": vc_score.astype(np.float32),
        "regime_0": regime_0,
        "regime_1": regime_1,
        "regime_2": regime_2,
    }
    feature_names = list(feats.keys())
    X = np.stack([feats[k] for k in feature_names], axis=1)  # (n, F)
    return X, feature_names, mean_direction


# =============================================================================
# ALIGNMENT CSV → DATASET TIMESTAMPS
# =============================================================================

def align_csv_features(csv_path: str, ts_target: np.ndarray) -> dict:
    """Charge CSV, calcule volume/ATR features, aligne sur timestamps cibles.

    Returns dict avec arrays alignés (n_target,):
        - close, volume, volume_ratio, volume_spike, atr_normalized
    """
    df = load_crypto_data(csv_path, asset_name="BTC")
    # Compute features causales sur le full CSV
    df["volume_ratio"] = calculate_volume_ratio(df["volume"], period=20)
    df["volume_spike"] = calculate_volume_spike(df["volume"], window=20)
    df["atr_normalized"] = calculate_atr_normalized(
        df["high"], df["low"], df["close"], period=14
    )
    # Forward-fill NaN du warmup pour ne pas perdre samples
    df["volume_ratio"] = df["volume_ratio"].ffill().fillna(0)
    df["volume_spike"] = df["volume_spike"].ffill().fillna(0)
    df["atr_normalized"] = df["atr_normalized"].ffill().fillna(0)

    # Timestamps en ms (int64)
    ts_csv = pd.to_datetime(df["timestamp"]).astype("int64").values // 10**6

    # Construire un index timestamp → position
    ts_to_idx = {int(t): i for i, t in enumerate(ts_csv.tolist())}
    idx = np.array([ts_to_idx[int(t)] for t in ts_target.tolist()], dtype=np.int64)

    return {
        "close": df["close"].values[idx].astype(np.float64),
        "volume": df["volume"].values[idx].astype(np.float64),
        "volume_ratio": df["volume_ratio"].values[idx].astype(np.float64),
        "volume_spike": df["volume_spike"].values[idx].astype(np.float64),
        "atr_normalized": df["atr_normalized"].values[idx].astype(np.float64),
    }


# =============================================================================
# MAIN
# =============================================================================

def process_split(split_name: str, datasets: dict, ckpts: dict, csv_features: dict,
                   tp: float, sl: float, t_max: int, batch_size: int, num_workers: int,
                   device: str) -> dict:
    """Traite un split (val ou test) : prédictions, features, labels TB.

    Returns dict {X, y, mean_direction, regime, ts_score, vc_score, timestamp,
                  close, feature_names}
    """
    print(f"\n=== {split_name.upper()} ===")
    # Get aligned data per indicator (one ref dataset for metadata)
    rsi_ds = datasets["rsi"]
    macd_ds = datasets["macd"]
    cci_ds = datasets["cci"]

    ts_rsi = np.asarray(rsi_ds[f"timestamp_{split_name}"], dtype=np.int64)
    ts_macd = np.asarray(macd_ds[f"timestamp_{split_name}"], dtype=np.int64)
    ts_cci = np.asarray(cci_ds[f"timestamp_{split_name}"], dtype=np.int64)

    # Intersection des timestamps (alignement strict)
    common_ts = sorted(set(ts_rsi.tolist()) & set(ts_macd.tolist()) & set(ts_cci.tolist()))
    common_ts = np.array(common_ts, dtype=np.int64)
    print(f"  Common timestamps: {len(common_ts):,}")

    def map_idx(ts_arr):
        ts_to_idx = {int(t): i for i, t in enumerate(ts_arr.tolist())}
        return np.array([ts_to_idx[int(t)] for t in common_ts.tolist()], dtype=np.int64)

    idx_rsi = map_idx(ts_rsi)
    idx_macd = map_idx(ts_macd)
    idx_cci = map_idx(ts_cci)

    # Predictions par indicateur (sur leur split puis aligner)
    print("  Inference RSI...")
    yhat_rsi_all = predict_split(
        ckpts["rsi"], rsi_ds[f"X_{split_name}"],
        float(rsi_ds["y_mean"]), float(rsi_ds["y_std"]),
        device, batch_size, num_workers,
    )
    print("  Inference MACD...")
    yhat_macd_all = predict_split(
        ckpts["macd"], macd_ds[f"X_{split_name}"],
        float(macd_ds["y_mean"]), float(macd_ds["y_std"]),
        device, batch_size, num_workers,
    )
    print("  Inference CCI...")
    yhat_cci_all = predict_split(
        ckpts["cci"], cci_ds[f"X_{split_name}"],
        float(cci_ds["y_mean"]), float(cci_ds["y_std"]),
        device, batch_size, num_workers,
    )

    # Aligner sur common_ts
    yhat_rsi = yhat_rsi_all[idx_rsi]
    yhat_macd = yhat_macd_all[idx_macd]
    yhat_cci = yhat_cci_all[idx_cci]

    # Régime/scores (depuis dataset RSI, identiques entre les 3 par construction)
    regime = np.asarray(rsi_ds[f"regime_{split_name}"], dtype=np.int8)[idx_rsi]
    ts_score = np.asarray(rsi_ds[f"ts_score_{split_name}"], dtype=np.float64)[idx_rsi]
    vc_score = np.asarray(rsi_ds[f"vc_score_{split_name}"], dtype=np.float64)[idx_rsi]

    # CSV-aligned features (volume, ATR, close) sur common_ts
    aligned = align_csv_features_split(csv_features, common_ts)
    close = aligned["close"]
    volume_ratio = aligned["volume_ratio"]
    volume_spike = aligned["volume_spike"]
    atr_norm = aligned["atr_normalized"]

    # Features
    print("  Computing features...")
    X, feature_names, mean_direction = compute_features(
        yhat_rsi, yhat_macd, yhat_cci, regime, ts_score, vc_score,
        volume_ratio, volume_spike, atr_norm,
    )

    # Labels Triple Barrier
    print(f"  Computing Triple Barrier labels (TP={tp:.4f}, SL={sl:.4f}, T={t_max})...")
    y = triple_barrier_label(close, mean_direction, tp=tp, sl=sl, t_max=t_max)

    # Stats
    n = len(y)
    pos_rate = y.mean() * 100
    no_trade = (mean_direction == 0).sum()
    print(f"  Samples: {n:,}")
    print(f"  Positive labels (TB winners): {y.sum():,} ({pos_rate:.2f}%)")
    print(f"  No-direction samples: {no_trade} ({no_trade / n * 100:.2f}%)")
    print(f"  Direction distribution: long={int((mean_direction > 0).sum()):,}, "
          f"short={int((mean_direction < 0).sum()):,}, flat={no_trade:,}")

    return {
        "X": X.astype(np.float32),
        "y": y.astype(np.int8),
        "mean_direction": mean_direction,
        "regime": regime,
        "ts_score": ts_score.astype(np.float32),
        "vc_score": vc_score.astype(np.float32),
        "timestamp": common_ts,
        "close": close.astype(np.float32),
        "yhat_rsi": yhat_rsi.astype(np.float32),
        "yhat_macd": yhat_macd.astype(np.float32),
        "yhat_cci": yhat_cci.astype(np.float32),
        "feature_names": feature_names,
    }


def align_csv_features_split(full_csv: dict, target_ts: np.ndarray) -> dict:
    """Sub-aligne les features CSV (déjà alignées sur full target) sur les common_ts."""
    # full_csv est déjà calculé sur tous les timestamps existant dans val ∪ test
    # mais on a besoin de re-mapper pour le split spécifique
    ts_to_idx = {int(t): i for i, t in enumerate(full_csv["_ts"].tolist())}
    idx = np.array([ts_to_idx[int(t)] for t in target_ts.tolist()], dtype=np.int64)
    return {
        "close": full_csv["close"][idx],
        "volume": full_csv["volume"][idx],
        "volume_ratio": full_csv["volume_ratio"][idx],
        "volume_spike": full_csv["volume_spike"][idx],
        "atr_normalized": full_csv["atr_normalized"][idx],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rsi-data", required=True)
    parser.add_argument("--macd-data", required=True)
    parser.add_argument("--cci-data", required=True)
    parser.add_argument("--rsi-ckpt", required=True)
    parser.add_argument("--macd-ckpt", required=True)
    parser.add_argument("--cci-ckpt", required=True)
    parser.add_argument("--csv-file", required=True, help="CSV brut OHLCV pour volume/ATR")
    parser.add_argument("--tp", type=float, default=0.005, help="Take-profit (default 0.5%)")
    parser.add_argument("--sl", type=float, default=0.002, help="Stop-loss (default 0.2%)")
    parser.add_argument("--t-max", type=int, default=6, help="Horizon TB en bars (default 6 = 30min)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output", default=str(ROOT / "data" / "foundation" / "meta_btc_close_kalman.npz"))
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")
    print(f"Triple Barrier: TP={args.tp:.4f} SL={args.sl:.4f} T_max={args.t_max} bars")

    # Charge datasets
    print("\nLoading datasets...")
    datasets = {
        "rsi": load_dataset(args.rsi_data)[0],
        "macd": load_dataset(args.macd_data)[0],
        "cci": load_dataset(args.cci_data)[0],
    }
    ckpts = {"rsi": args.rsi_ckpt, "macd": args.macd_ckpt, "cci": args.cci_ckpt}

    # Calcule features CSV une fois sur la totalité (pour aligner ensuite val + test)
    print("\nLoading CSV + computing volume/ATR features...")
    df = load_crypto_data(args.csv_file, asset_name="BTC")
    df["volume_ratio"] = calculate_volume_ratio(df["volume"], period=20)
    df["volume_spike"] = calculate_volume_spike(df["volume"], window=20)
    df["atr_normalized"] = calculate_atr_normalized(df["high"], df["low"], df["close"], period=14)
    df["volume_ratio"] = df["volume_ratio"].ffill().fillna(0)
    df["volume_spike"] = df["volume_spike"].ffill().fillna(0)
    df["atr_normalized"] = df["atr_normalized"].ffill().fillna(0)
    ts_csv = pd.to_datetime(df["timestamp"]).astype("int64").values // 10**6
    csv_features = {
        "_ts": ts_csv,
        "close": df["close"].values.astype(np.float64),
        "volume": df["volume"].values.astype(np.float64),
        "volume_ratio": df["volume_ratio"].values.astype(np.float64),
        "volume_spike": df["volume_spike"].values.astype(np.float64),
        "atr_normalized": df["atr_normalized"].values.astype(np.float64),
    }
    print(f"  CSV: {len(ts_csv):,} rows")

    # Process val + test
    val = process_split("val", datasets, ckpts, csv_features,
                         tp=args.tp, sl=args.sl, t_max=args.t_max,
                         batch_size=args.batch_size, num_workers=args.num_workers, device=device)
    test = process_split("test", datasets, ckpts, csv_features,
                          tp=args.tp, sl=args.sl, t_max=args.t_max,
                          batch_size=args.batch_size, num_workers=args.num_workers, device=device)

    # Save
    metadata = {
        "config": "Meta-classifier dataset for transition detection",
        "tp": args.tp, "sl": args.sl, "t_max": args.t_max,
        "feature_names": val["feature_names"],
        "n_features": len(val["feature_names"]),
        "n_train_meta (= val of specialists)": len(val["y"]),
        "n_test_meta (= test of specialists)": len(test["y"]),
        "tb_pos_rate_train": float(val["y"].mean()),
        "tb_pos_rate_test": float(test["y"].mean()),
        "ckpts": ckpts,
        "datasets": {"rsi": args.rsi_data, "macd": args.macd_data, "cci": args.cci_data},
        "csv_file": args.csv_file,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    save_dict = {
        "X_meta_train": val["X"],
        "y_meta_train": val["y"],
        "mean_direction_train": val["mean_direction"],
        "regime_train": val["regime"],
        "ts_score_train": val["ts_score"],
        "vc_score_train": val["vc_score"],
        "timestamp_train": val["timestamp"],
        "close_train": val["close"],
        "yhat_rsi_train": val["yhat_rsi"],
        "yhat_macd_train": val["yhat_macd"],
        "yhat_cci_train": val["yhat_cci"],

        "X_meta_test": test["X"],
        "y_meta_test": test["y"],
        "mean_direction_test": test["mean_direction"],
        "regime_test": test["regime"],
        "ts_score_test": test["ts_score"],
        "vc_score_test": test["vc_score"],
        "timestamp_test": test["timestamp"],
        "close_test": test["close"],
        "yhat_rsi_test": test["yhat_rsi"],
        "yhat_macd_test": test["yhat_macd"],
        "yhat_cci_test": test["yhat_cci"],

        "feature_names": np.array(val["feature_names"]),
        "meta": json.dumps(metadata),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **save_dict)
    size_mb = out_path.stat().st_size / (1024**2)
    print(f"\n✓ Saved {out_path} ({size_mb:.1f} MB)")
    print(f"  Features ({len(val['feature_names'])}): {val['feature_names']}")
    print(f"  TB positive rate — train: {val['y'].mean():.3f}  test: {test['y'].mean():.3f}")


if __name__ == "__main__":
    main()
