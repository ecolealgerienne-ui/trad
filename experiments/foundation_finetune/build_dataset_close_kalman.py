"""Build dataset for Kalman_RTS(close) slope prediction from a single indicator.

Setup expérimental (Étape A du plan multi-indicateur):
    Input  : sequence d'un indicateur (RSI, MACD-hist, ou CCI), fenêtre 96 timesteps
    Target : Kalman_RTS(close)[t] - Kalman_RTS(close)[t-1]   (pente smoothed du prix)
    Régime : labels 0/1/2 + scores TS/VC pour stratification de l'analyse

Config 1 — indicateur seul, pas de close brut en input.

Anti-leakage RTS Kalman (CRITIQUE):
    Le RTS smoother est non-causal (utilise le futur). On calcule Kalman PAR SPLIT
    pour qu'aucun smoother de train ne voie val/test, etc.
        kalman_train  = kalman_filter(close[:n_train])
        kalman_val    = kalman_filter(close[:n_train + n_val])
        kalman_test   = kalman_filter(close[:n])

Per-asset (CLAUDE.md règle stricte):
    Indicateurs et Kalman calculés indépendamment par asset, puis concaténés.

Réutilise:
    - src/data_utils.load_crypto_data
    - src/indicators.calculate_rsi / calculate_macd / calculate_cci
    - src/filters.kalman_filter (pykalman, RTS smoother)
    - src/regime_features.calculate_all_regime_features
    - src/regime_labeler.calculate_regime_labels
    - src/constants.* (RSI_PERIOD, MACD_*, CCI_PERIOD, KALMAN_*, splits)

Usage (défaut: BTC seul, comme Phase 1-9):
    python experiments/foundation_finetune/build_dataset_close_kalman.py --indicator rsi
    python experiments/foundation_finetune/build_dataset_close_kalman.py --indicator macd
    python experiments/foundation_finetune/build_dataset_close_kalman.py --indicator cci

Pour scaler à 5 assets ensuite :
    python ... --indicator rsi --assets BTC ETH BNB ADA LTC

Output:
    data/foundation/{indicator}_{assets_tag}_close_kalman_5min.npz
    Ex: rsi_btc_close_kalman_5min.npz   (BTC seul)
        rsi_btc_eth_bnb_ada_ltc_close_kalman_5min.npz   (5 assets)
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# --- Bridge vers src/ ---
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from constants import (
    AVAILABLE_ASSETS_5M,
    CCI_PERIOD,
    KALMAN_MEASURE_VAR,
    KALMAN_PROCESS_VAR,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    RSI_PERIOD,
    TRAIN_SPLIT,
    VAL_SPLIT,
)
from data_utils import load_crypto_data
from filters import kalman_filter
from indicators import calculate_cci, calculate_macd, calculate_rsi
from regime_features import calculate_all_regime_features
from regime_labeler import calculate_regime_labels


# =============================================================================
# CONFIG
# =============================================================================

WINDOW = 96
ASSET_IDS = {"BTC": 0, "ETH": 1, "BNB": 2, "ADA": 3, "LTC": 4}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("build_dataset_close_kalman")
# Réduire verbosité des modules src
logging.getLogger("regime_features").setLevel(logging.WARNING)
logging.getLogger("regime_labeler").setLevel(logging.WARNING)


# =============================================================================
# INDICATEURS
# =============================================================================

def compute_indicator(df: pd.DataFrame, name: str) -> np.ndarray:
    """Calcule l'indicateur demandé sur le DataFrame OHLC."""
    if name == "rsi":
        return calculate_rsi(df["close"], period=RSI_PERIOD)
    if name == "macd":
        # Histogramme = MACD - signal (z-scorable, momentum centré)
        out = calculate_macd(
            df["close"],
            fast_period=MACD_FAST,
            slow_period=MACD_SLOW,
            signal_period=MACD_SIGNAL,
        )
        return out["histogram"]
    if name == "cci":
        return calculate_cci(df["high"], df["low"], df["close"], period=CCI_PERIOD)
    raise ValueError(f"Indicateur inconnu: {name}")


def indicator_params(name: str) -> dict:
    """Paramètres effectifs (pour metadata)."""
    if name == "rsi":
        return {"period": RSI_PERIOD}
    if name == "macd":
        return {
            "fast": MACD_FAST,
            "slow": MACD_SLOW,
            "signal": MACD_SIGNAL,
            "output": "histogram",
        }
    if name == "cci":
        return {"period": CCI_PERIOD, "constant": 0.015}
    raise ValueError(name)


# =============================================================================
# PER-ASSET PROCESSING
# =============================================================================

def process_asset(
    asset: str,
    csv_path: str,
    indicator_name: str,
    process_var: float,
    measure_var: float,
) -> dict:
    """Charge un asset, calcule indicateur, régime, et prépare la base de séquences.

    NOTE: Kalman(close) sera calculé par split en aval (anti-leakage).
    On ne le pré-calcule pas ici sur le full pour éviter toute confusion.

    Returns:
        dict avec arrays per-asset:
          - close (n,)
          - indicator (n,)
          - regime (n,)
          - ts_score, vc_score (n,)
          - timestamp (n,) en ms
    """
    logger.info(f"--- {asset} ---")
    df = load_crypto_data(csv_path, asset_name=asset)

    # Indicateur (calculé sur close ou OHLC selon indicateur)
    df["indicator"] = compute_indicator(df, indicator_name)

    # Features de régime (~23 colonnes ajoutées) puis classification
    df = calculate_all_regime_features(df)
    regime_labels, ts_score, vc_score = calculate_regime_labels(df)
    df["regime"] = regime_labels
    df["ts_score"] = ts_score
    df["vc_score"] = vc_score

    # Drop NaN du warmup (indicateurs + régime features ont des windows)
    required = ["indicator", "regime", "ts_score", "vc_score", "close"]
    n_before = len(df)
    df = df.dropna(subset=required).reset_index(drop=True)
    n_after = len(df)
    logger.info(f"  {asset}: {n_before} → {n_after} rows after warmup drop ({n_after / n_before * 100:.1f}%)")

    # Timestamps en ms (int64) pour traçabilité
    if "timestamp" in df.columns:
        ts_ms = pd.to_datetime(df["timestamp"]).astype("int64").values // 10**6
    else:
        ts_ms = df.index.astype("int64").values // 10**6

    return {
        "asset": asset,
        "asset_id": ASSET_IDS[asset],
        "close": df["close"].astype(np.float64).values,
        "indicator": df["indicator"].astype(np.float64).values,
        "regime": df["regime"].astype(np.int8).values,
        "ts_score": df["ts_score"].astype(np.float32).values,
        "vc_score": df["vc_score"].astype(np.float32).values,
        "timestamp": ts_ms.astype(np.int64),
    }


# =============================================================================
# KALMAN PER SPLIT (anti-leakage)
# =============================================================================

def kalman_close_per_split(
    close: np.ndarray,
    n_train: int,
    n_val: int,
    process_var: float,
    measure_var: float,
) -> tuple:
    """Calcule Kalman_RTS(close) et sa pente par split, sans leakage.

    Anti-leakage:
      - Pour i < n_train        : utilise kalman_filter(close[:n_train])
      - Pour n_train ≤ i < n_tv : utilise kalman_filter(close[:n_tv])
      - Pour i ≥ n_tv           : utilise kalman_filter(close[:n])

    Pente intra-split (cohérence):
      Pour éviter une discontinuité aux frontières (où kalman[i] et kalman[i-1]
      viendraient de deux runs Kalman différents), la pente d'un sample i utilise
      la même Kalman run pour kalman[i] et kalman[i-1] :
        slope[i] = kf_split[i] - kf_split[i-1]   (split = celui contenant i)

      Convention au tout premier sample de chaque split région : pente = 0
      (rare : 1 par région, total ≤ 3 sur ~880k par asset).

    Returns:
        kalman_close (n,) : valeur Kalman per-split (utilisée pour metadata)
        slope        (n,) : pente intra-split, sans cross-Kalman aux frontières
    """
    n = len(close)
    n_tv = n_train + n_val

    kf_train = kalman_filter(close[:n_train], process_variance=process_var, measurement_variance=measure_var)
    kf_tv = kalman_filter(close[:n_tv], process_variance=process_var, measurement_variance=measure_var)
    kf_full = kalman_filter(close[:n], process_variance=process_var, measurement_variance=measure_var)

    # Valeur Kalman per-split (anti-leakage)
    kalman_close = np.empty(n, dtype=np.float64)
    kalman_close[:n_train] = kf_train[:n_train]
    kalman_close[n_train:n_tv] = kf_tv[n_train:n_tv]
    kalman_close[n_tv:] = kf_full[n_tv:]

    # Pente intra-split (utilise la même run pour kalman[i] et kalman[i-1])
    slope = np.zeros(n, dtype=np.float64)
    # Train: slope[i] = kf_train[i] - kf_train[i-1] pour i in [1, n_train)
    slope[1:n_train] = kf_train[1:n_train] - kf_train[:n_train - 1]
    # Val: slope[i] = kf_tv[i] - kf_tv[i-1] pour i in [n_train, n_tv)
    if n_tv > n_train:
        # premier val sample i = n_train : kf_tv[n_train] - kf_tv[n_train-1]
        slope[n_train:n_tv] = kf_tv[n_train:n_tv] - kf_tv[n_train - 1:n_tv - 1]
    # Test: slope[i] = kf_full[i] - kf_full[i-1] pour i in [n_tv, n)
    if n > n_tv:
        slope[n_tv:n] = kf_full[n_tv:n] - kf_full[n_tv - 1:n - 1]

    return kalman_close, slope


# =============================================================================
# SÉQUENCES + SPLITS (per-asset chronologique)
# =============================================================================

def build_sequences_for_asset(
    asset_data: dict,
    window: int,
    train_split: float,
    val_split: float,
    process_var: float,
    measure_var: float,
) -> dict:
    """Construit X, y et metadata pour un asset, avec Kalman per-split.

    Logique:
      n = len(asset)
      n_train = int(n * train_split)
      n_val   = int(n * val_split)
      Kalman calculé per-split sur ces bornes (anti-leakage)
      slope[i] = kalman[i] - kalman[i-1]
      Sequences X[i] = indicator[i-window+1 : i+1]   pour i in [window-1, n-1]
      Target    y[i] = slope[i]
      Le split d'appartenance d'un sample i est défini par i:
        i < n_train               → train
        n_train ≤ i < n_train+n_val → val
        sinon                      → test
    """
    close = asset_data["close"]
    indicator = asset_data["indicator"]
    regime = asset_data["regime"]
    ts_score = asset_data["ts_score"]
    vc_score = asset_data["vc_score"]
    timestamp = asset_data["timestamp"]
    asset_id = asset_data["asset_id"]

    n = len(close)
    if n < window + 100:
        raise ValueError(f"Asset {asset_data['asset']}: trop peu de données ({n} rows)")

    n_train = int(n * train_split)
    n_val = int(n * val_split)
    n_tv = n_train + n_val

    # Kalman per split (anti-leakage) + pente intra-split (cohérente)
    logger.info(f"  Kalman(close) per-split: train[:{n_train}] val[:{n_tv}] test[:{n}]...")
    kalman_close, slope = kalman_close_per_split(
        close, n_train=n_train, n_val=n_val,
        process_var=process_var, measure_var=measure_var,
    )

    # Sequences valides : i in [window-1, n-1]
    idxs = np.arange(window - 1, n, dtype=np.int64)
    n_seq = len(idxs)

    # Construction X (vectorisé via stride/fancy indexing)
    # X[k, j] = indicator[idxs[k] - window + 1 + j]
    # Shape (n_seq, window) en 2D — convention Phase 1-9 / Chronos univariate
    base = idxs - (window - 1)  # (n_seq,)
    offsets = np.arange(window)  # (window,)
    seq_idx = base[:, None] + offsets[None, :]  # (n_seq, window)
    X = indicator[seq_idx].astype(np.float32)  # (n_seq, window) — 2D pour Chronos

    # Cibles et metadata alignées sur idxs (point de prédiction)
    y_raw = slope[idxs].astype(np.float32)
    meta = {
        "timestamp": timestamp[idxs].astype(np.int64),
        "asset_id": np.full(n_seq, asset_id, dtype=np.int8),
        "regime": regime[idxs].astype(np.int8),
        "ts_score": ts_score[idxs].astype(np.float32),
        "vc_score": vc_score[idxs].astype(np.float32),
        "close": close[idxs].astype(np.float32),
        "kalman_close": kalman_close[idxs].astype(np.float32),
    }

    # Split mask basé sur idxs (chronologique)
    train_mask = idxs < n_train
    val_mask = (idxs >= n_train) & (idxs < n_tv)
    test_mask = idxs >= n_tv

    return {
        "X": X,
        "y_raw": y_raw,
        "meta": meta,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
    }


# =============================================================================
# NORMALISATION
# =============================================================================

def zscore(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return ((values - mean) / max(std, 1e-8)).astype(np.float32)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--indicator", required=True, choices=["rsi", "macd", "cci"],
                        help="Indicateur à utiliser comme input (1 channel)")
    parser.add_argument("--assets", nargs="+",
                        default=["BTC"],
                        help="Assets à inclure (défaut: BTC seul, étendre via 'BTC ETH BNB ADA LTC')")
    parser.add_argument("--window", type=int, default=WINDOW,
                        help=f"Taille fenêtre sequence (défaut: {WINDOW})")
    parser.add_argument("--process-var", type=float, default=KALMAN_PROCESS_VAR,
                        help=f"Kalman process variance Q (défaut: {KALMAN_PROCESS_VAR})")
    parser.add_argument("--measure-var", type=float, default=KALMAN_MEASURE_VAR,
                        help=f"Kalman measurement variance R (défaut: {KALMAN_MEASURE_VAR})")
    parser.add_argument("--train-split", type=float, default=TRAIN_SPLIT)
    parser.add_argument("--val-split", type=float, default=VAL_SPLIT)
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "foundation"))
    parser.add_argument("--output-name", default=None,
                        help="Nom de fichier custom (défaut: auto)")
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info(f"BUILD DATASET — indicator={args.indicator} | window={args.window}")
    logger.info(f"Assets: {args.assets}")
    logger.info(f"Splits: train={args.train_split} val={args.val_split} test={1 - args.train_split - args.val_split:.2f}")
    logger.info(f"Kalman: Q={args.process_var} R={args.measure_var}")
    logger.info("=" * 80)

    # Process per asset
    asset_results = []
    for asset in args.assets:
        if asset not in AVAILABLE_ASSETS_5M:
            logger.warning(f"Asset {asset} inconnu, skipping")
            continue
        csv_path = AVAILABLE_ASSETS_5M[asset]
        if not Path(csv_path).exists():
            logger.warning(f"CSV manquant pour {asset}: {csv_path}, skipping")
            continue

        asset_data = process_asset(
            asset=asset,
            csv_path=csv_path,
            indicator_name=args.indicator,
            process_var=args.process_var,
            measure_var=args.measure_var,
        )
        seq = build_sequences_for_asset(
            asset_data=asset_data,
            window=args.window,
            train_split=args.train_split,
            val_split=args.val_split,
            process_var=args.process_var,
            measure_var=args.measure_var,
        )
        n_tr = int(seq["train_mask"].sum())
        n_va = int(seq["val_mask"].sum())
        n_te = int(seq["test_mask"].sum())
        logger.info(f"  {asset}: sequences = {len(seq['X'])} (train {n_tr} | val {n_va} | test {n_te})")
        asset_results.append(seq)

    if not asset_results:
        logger.error("Aucun asset traité.")
        sys.exit(1)

    # Concaténation cross-asset
    logger.info("Concaténation cross-asset...")
    X_all = np.concatenate([s["X"] for s in asset_results], axis=0)
    y_raw_all = np.concatenate([s["y_raw"] for s in asset_results], axis=0)
    meta_keys = list(asset_results[0]["meta"].keys())
    meta_all = {k: np.concatenate([s["meta"][k] for s in asset_results], axis=0) for k in meta_keys}
    train_mask = np.concatenate([s["train_mask"] for s in asset_results], axis=0)
    val_mask = np.concatenate([s["val_mask"] for s in asset_results], axis=0)
    test_mask = np.concatenate([s["test_mask"] for s in asset_results], axis=0)

    logger.info(f"Total sequences: {len(X_all):,}")
    logger.info(f"  Train: {int(train_mask.sum()):,} ({train_mask.mean() * 100:.1f}%)")
    logger.info(f"  Val:   {int(val_mask.sum()):,} ({val_mask.mean() * 100:.1f}%)")
    logger.info(f"  Test:  {int(test_mask.sum()):,} ({test_mask.mean() * 100:.1f}%)")

    # Normalisation z-score (stats sur train uniquement)
    logger.info("Normalisation z-score (stats train uniquement)...")
    X_mean = float(X_all[train_mask].mean())
    X_std = float(X_all[train_mask].std() + 1e-8)
    y_mean = float(y_raw_all[train_mask].mean())
    y_std = float(y_raw_all[train_mask].std() + 1e-8)
    logger.info(f"  X: mean={X_mean:.6f} std={X_std:.6f}")
    logger.info(f"  y: mean={y_mean:.6e} std={y_std:.6e}")

    X_norm = zscore(X_all, X_mean, X_std)
    y_norm = zscore(y_raw_all, y_mean, y_std)

    # Distribution régimes par split
    logger.info("Distribution régimes par split:")
    for split_name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        regimes = meta_all["regime"][mask]
        if len(regimes) == 0:
            continue
        counts = pd.Series(regimes).value_counts(normalize=True).sort_index()
        descr = ", ".join([f"R{r}={p * 100:.1f}%" for r, p in counts.items()])
        logger.info(f"  {split_name}: {descr}")

    # ---------------- Save ----------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output_name:
        out_path = output_dir / args.output_name
    else:
        assets_tag = "_".join(a.lower() for a in args.assets)
        out_path = output_dir / f"{args.indicator}_{assets_tag}_close_kalman_5min.npz"

    metadata = {
        "indicator": args.indicator,
        "indicator_params": indicator_params(args.indicator),
        "timeframe": "5min",
        "window": args.window,
        "kalman_label": "RTS smoothed slope of close (anti-leakage per-split)",
        "kalman_params": {
            "process_var": args.process_var,
            "measure_var": args.measure_var,
        },
        "assets": list(args.assets),
        "asset_ids": ASSET_IDS,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "test_split": 1.0 - args.train_split - args.val_split,
        "split_strategy": "Per-asset chronological. Kalman computed per-split (anti-leakage).",
        "config": "Config 1 - indicator only (no raw close in input)",
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
        "x_norm": {"mean": X_mean, "std": X_std},
        "y_norm": {"mean": y_mean, "std": y_std},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    save_dict = {
        # Inputs/targets normalisés
        "X_train": X_norm[train_mask],
        "X_val": X_norm[val_mask],
        "X_test": X_norm[test_mask],
        "y_train": y_norm[train_mask],
        "y_val": y_norm[val_mask],
        "y_test": y_norm[test_mask],
        # Cibles brutes (avant z-score) pour comparaison FLKS / dénormalisation
        "y_raw_train": y_raw_all[train_mask],
        "y_raw_val": y_raw_all[val_mask],
        "y_raw_test": y_raw_all[test_mask],
        # Stats normalisation (pour dénormaliser les prédictions)
        "X_mean": np.float32(X_mean),
        "X_std": np.float32(X_std),
        "y_mean": np.float32(y_mean),
        "y_std": np.float32(y_std),
        # Metadata JSON (clé "meta" pour compatibilité train.py / evaluate.py Phase 1-9)
        "meta": json.dumps(metadata),
    }
    # Métadonnées par split (timestamp, asset_id, regime, scores, close, kalman_close)
    for split_name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        for k, arr in meta_all.items():
            save_dict[f"{k}_{split_name}"] = arr[mask]

    logger.info(f"Saving → {out_path}...")
    np.savez_compressed(out_path, **save_dict)
    size_mb = out_path.stat().st_size / (1024**2)
    logger.info(f"✓ Saved {out_path} ({size_mb:.1f} MB)")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
