"""Analyse comparative des 3 spécialistes RSI / MACD / CCI sur Kalman_RTS(close).

Étape C du plan multi-indicateur. Suppose que les 3 datasets ont été construits
avec build_dataset_close_kalman.py et que les 3 modèles ont été entraînés
indépendamment via train.py (mêmes hyper-paramètres).

Métriques calculées (sur test set, alignement par timestamp) :

  Per-indicateur :
    - Pearson, DirMatch, MSE, MAE
    - Stratifié par régime (RANGE_LOW_VOL / RANGE_HIGH_VOL / TREND)

  Inter-indicateur (accord) :
    - Pearson pairwise des prédictions
    - Sign agreement pairwise
    - Triple agreement distribution (3/3, 2/3, 0/3, unanimité)
    - Error overlap (Jaccard sur erreurs de signe)
    - Complémentarité (A se trompe / B a raison)

  Per-régime : tout ce qui précède stratifié par régime.

Sortie :
    - Tables texte sur stdout (lecture humaine)
    - JSON complet pour analyse downstream

Réutilise :
    - load_model + predict_model de evaluate.py
    - Datasets construits par build_dataset_close_kalman.py

Usage :
    python experiments/foundation_finetune/analyze_specialists_close_kalman.py \\
        --rsi-data  data/foundation/rsi_btc_close_kalman_5min.npz \\
        --macd-data data/foundation/macd_btc_close_kalman_5min.npz \\
        --cci-data  data/foundation/cci_btc_close_kalman_5min.npz \\
        --rsi-ckpt  models/specialist_rsi/chronos-t5-tiny_lora.pt \\
        --macd-ckpt models/specialist_macd/chronos-t5-tiny_lora.pt \\
        --cci-ckpt  models/specialist_cci/chronos-t5-tiny_lora.pt \\
        --output    results/specialists_analysis.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Bridge vers src/ (au cas où) + même répertoire pour évaluer.py
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import load_model, predict_model  # noqa: E402


REGIME_NAMES = {0: "RANGE_LOW_VOL", 1: "RANGE_HIGH_VOL", 2: "TREND"}


# =============================================================================
# MÉTRIQUES
# =============================================================================

def metrics(yhat: np.ndarray, y: np.ndarray) -> dict:
    """MSE, MAE, DirMatch (sign concordance), Pearson."""
    yhat = np.asarray(yhat, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = int(len(y))
    if n == 0:
        return {"n": 0, "mse": float("nan"), "mae": float("nan"),
                "dirmatch": float("nan"), "pearson": float("nan")}
    mse = float(np.mean((yhat - y) ** 2))
    mae = float(np.mean(np.abs(yhat - y)))
    dirmatch = float(np.mean(np.sign(yhat) == np.sign(y)))
    if y.std() > 1e-12 and yhat.std() > 1e-12:
        pearson = float(np.corrcoef(yhat, y)[0, 1])
    else:
        pearson = float("nan")
    return {"n": n, "mse": mse, "mae": mae, "dirmatch": dirmatch, "pearson": pearson}


def lag_ccf(yhat: np.ndarray, y: np.ndarray, max_lag: int = 5) -> tuple:
    """Cross-correlation par lag, retourne dict {lag: ccf} et best_lag."""
    yhat = np.asarray(yhat, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if y.std() < 1e-12 or yhat.std() < 1e-12:
        return {0: float("nan")}, 0
    ccf = {}
    n = len(y)
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a, b = yhat[:n + lag], y[-lag:]
        elif lag > 0:
            a, b = yhat[lag:], y[:n - lag]
        else:
            a, b = yhat, y
        if a.std() > 1e-12 and b.std() > 1e-12 and len(a) > 1:
            ccf[lag] = float(np.corrcoef(a, b)[0, 1])
        else:
            ccf[lag] = float("nan")
    best_lag = max(ccf, key=lambda k: -float("inf") if np.isnan(ccf[k]) else ccf[k])
    return ccf, best_lag


# =============================================================================
# I/O
# =============================================================================

def load_dataset(path: str):
    data = np.load(path, allow_pickle=True)
    # Backward compat: accept both "meta" (Phase 1-9) and "metadata" (close_kalman builds)
    meta_key = "meta" if "meta" in data.files else "metadata"
    meta = json.loads(str(data[meta_key]))
    return data, meta


def predict_test(ckpt_path: str, data, device: str, batch_size: int, num_workers: int) -> tuple:
    """Charge model, prédit sur test set, dénormalise vers unités slope brutes.

    Returns:
        yhat_raw (N,) : prédiction dénormalisée (mêmes unités que y_raw_test)
        y_raw    (N,) : ground truth (slope brute Kalman_RTS(close))
    """
    model, _ = load_model(Path(ckpt_path), device)
    X_test = data["X_test"]
    has_extras = "extras_test" in data.files
    extras = data["extras_test"] if has_extras else None
    yhat_norm = predict_model(model, X_test, extras, device, batch_size, num_workers)
    # Dénormalisation
    y_mean = float(data["y_mean"])
    y_std = float(data["y_std"])
    yhat_raw = yhat_norm * y_std + y_mean
    y_raw = np.asarray(data["y_raw_test"], dtype=np.float64)
    # Libère GPU
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return yhat_raw.astype(np.float64), y_raw


# =============================================================================
# ALIGNEMENT PAR TIMESTAMP (les 3 datasets peuvent différer en taille)
# =============================================================================

def align_by_timestamp(raw: dict) -> dict:
    """Aligne les prédictions sur les timestamps communs aux 3 indicateurs.

    Args:
        raw: dict {name: {yhat, y_true, ts, regime, asset_id}}

    Returns:
        dict {name: même structure mais aligné sur common_ts}
    """
    ts_sets = [set(d["ts"].tolist()) for d in raw.values()]
    common = sorted(set.intersection(*ts_sets))
    common_ts = np.asarray(common, dtype=np.int64)

    aligned = {}
    for name, d in raw.items():
        ts = d["ts"]
        # Map timestamp → index dans le tableau original
        ts_to_idx = {int(t): i for i, t in enumerate(ts.tolist())}
        idx = np.array([ts_to_idx[int(t)] for t in common_ts], dtype=np.int64)
        aligned[name] = {
            "ts": common_ts,
            "yhat": d["yhat"][idx],
            "y_true": d["y_true"][idx],
            "regime": d["regime"][idx],
            "asset_id": d["asset_id"][idx],
        }

    # Sanity check : y_true doit être quasi-identique entre indicateurs
    # (computed sur même close, mais avec splits potentiellement différents)
    names = list(aligned.keys())
    if len(names) >= 2:
        ref = aligned[names[0]]["y_true"]
        for n in names[1:]:
            other = aligned[n]["y_true"]
            diff = np.abs(ref - other)
            max_diff = float(diff.max()) if len(diff) > 0 else 0.0
            mean_abs = float(np.mean(np.abs(ref))) if len(ref) > 0 else 1.0
            rel = max_diff / max(mean_abs, 1e-12)
            print(f"  y_true sanity {names[0]} vs {n}: "
                  f"max_abs_diff={max_diff:.3e} rel={rel:.3%}")
    return aligned


# =============================================================================
# MÉTRIQUES PER-INDICATEUR (avec stratification régime)
# =============================================================================

def per_indicator_analysis(aligned: dict) -> dict:
    out = {}
    for name, d in aligned.items():
        y_true = d["y_true"]
        yhat = d["yhat"]
        regime = d["regime"]
        ccf, best = lag_ccf(yhat, y_true)
        block = {
            "all": metrics(yhat, y_true),
            "lag_ccf": {"best_lag": int(best), "ccf_at_best": ccf[best],
                        "ccf_full": {int(k): v for k, v in ccf.items()}},
            "by_regime": {},
        }
        for r_id, r_name in REGIME_NAMES.items():
            mask = regime == r_id
            if mask.sum() > 0:
                block["by_regime"][r_name] = metrics(yhat[mask], y_true[mask])
        out[name] = block
    return out


# =============================================================================
# ACCORD INTER-INDICATEUR (pairwise + triple)
# =============================================================================

def pairwise_agreement(aligned: dict, regime_filter: int = None) -> dict:
    """Pour chaque paire d'indicateurs : Pearson(predictions),
    sign agreement, error Jaccard (sign), complémentarité.
    """
    names = list(aligned.keys())
    pairs = [(names[i], names[j]) for i in range(len(names))
             for j in range(i + 1, len(names))]

    out = {}
    for a, b in pairs:
        ya = aligned[a]["yhat"]
        yb = aligned[b]["yhat"]
        truth = aligned[a]["y_true"]  # y_true commun (vérifié par sanity check)
        regime = aligned[a]["regime"]

        if regime_filter is not None:
            mask = regime == regime_filter
            ya, yb, truth = ya[mask], yb[mask], truth[mask]

        n = int(len(ya))
        if n == 0:
            continue

        if ya.std() > 1e-12 and yb.std() > 1e-12:
            pearson = float(np.corrcoef(ya, yb)[0, 1])
        else:
            pearson = float("nan")

        sign_agreement = float(np.mean(np.sign(ya) == np.sign(yb)))

        # Erreurs de signe vs truth
        truth_sign = np.sign(truth)
        # Si truth == 0, on compte comme UP (rare, et négligeable)
        truth_sign[truth_sign == 0] = 1
        err_a = (np.sign(ya) != truth_sign)
        err_b = (np.sign(yb) != truth_sign)
        union = err_a | err_b
        intersect = err_a & err_b
        jaccard = float(intersect.sum() / max(union.sum(), 1))
        comp_a_b = float((err_a & ~err_b).mean())
        comp_b_a = float((err_b & ~err_a).mean())
        complementarity = comp_a_b + comp_b_a

        out[f"{a}_vs_{b}"] = {
            "n": n,
            "pearson_pred": pearson,
            "sign_agreement": sign_agreement,
            "error_jaccard": jaccard,
            "complementarity": complementarity,
            "a_wrong_b_right": comp_a_b,
            "b_wrong_a_right": comp_b_a,
            "err_rate_a": float(err_a.mean()),
            "err_rate_b": float(err_b.mean()),
        }
    return out


def triple_agreement(aligned: dict, regime_filter: int = None) -> dict:
    """Distribution d'accord sur les 3 indicateurs : 3/3, 2/3, 0/3, unanimité,
    et accuracy du vote majoritaire vs truth.
    """
    names = list(aligned.keys())
    if len(names) != 3:
        return None

    signs = np.stack([np.sign(aligned[n]["yhat"]) for n in names], axis=0)  # (3, N)
    truth = aligned[names[0]]["y_true"]
    regime = aligned[names[0]]["regime"]

    if regime_filter is not None:
        mask = regime == regime_filter
        signs = signs[:, mask]
        truth = truth[mask]

    n = int(signs.shape[1])
    if n == 0:
        return None

    n_up = (signs > 0).sum(axis=0)  # (N,) values in {0,1,2,3}
    dist = {
        "n": n,
        "all_up_3_3": float((n_up == 3).mean()),
        "majority_up_2_3": float((n_up == 2).mean()),
        "majority_down_2_3": float((n_up == 1).mean()),
        "all_down_3_3": float((n_up == 0).mean()),
    }
    dist["unanimity_rate"] = dist["all_up_3_3"] + dist["all_down_3_3"]

    # Vote majoritaire : >= 2 UP → +1, sinon -1
    majority_pred = np.where(n_up >= 2, 1, -1)
    truth_sign = np.sign(truth)
    truth_sign[truth_sign == 0] = 1
    dist["majority_vote_dirmatch"] = float((majority_pred == truth_sign).mean())

    # Accuracy moyenne individuelle (pour comparer vs vote majoritaire)
    individual_dirmatches = [(np.sign(aligned[n]["yhat"][regime == regime_filter]
                                       if regime_filter is not None else aligned[n]["yhat"])
                              == truth_sign).mean()
                             for n in names]
    dist["mean_individual_dirmatch"] = float(np.mean(individual_dirmatches))
    dist["majority_vs_individual_delta"] = (
        dist["majority_vote_dirmatch"] - dist["mean_individual_dirmatch"])

    return dist


# =============================================================================
# FORMATAGE TABLES (lecture humaine)
# =============================================================================

def fmt_per_indicator(per_ind: dict) -> str:
    lines = []
    lines.append(f"{'Indicator':<10} {'Regime':<18} {'N':>10} "
                 f"{'Pearson':>10} {'DirMatch':>10} {'MSE':>13} {'MAE':>13} {'best_lag':>10}")
    lines.append("-" * 100)
    for name, d in per_ind.items():
        m = d["all"]
        lag_info = f"lag={d['lag_ccf']['best_lag']:+d}"
        lines.append(f"{name:<10} {'ALL':<18} {m['n']:>10} "
                     f"{m['pearson']:>+10.4f} {m['dirmatch']:>10.4f} "
                     f"{m['mse']:>13.6e} {m['mae']:>13.6e} {lag_info:>10}")
        for r_name, mr in d["by_regime"].items():
            lines.append(f"{'':<10} {r_name:<18} {mr['n']:>10} "
                         f"{mr['pearson']:>+10.4f} {mr['dirmatch']:>10.4f} "
                         f"{mr['mse']:>13.6e} {mr['mae']:>13.6e}")
        lines.append("")
    return "\n".join(lines)


def fmt_pairwise(pairs: dict, label: str) -> str:
    lines = [f"\n--- Pairwise [{label}] ---"]
    lines.append(f"{'Pair':<22} {'N':>10} {'Pearson':>10} "
                 f"{'SignAgr':>10} {'ErrJacc':>10} {'Compl':>10}")
    lines.append("-" * 80)
    for pair, m in pairs.items():
        lines.append(f"{pair:<22} {m['n']:>10} {m['pearson_pred']:>+10.4f} "
                     f"{m['sign_agreement']:>10.4f} {m['error_jaccard']:>10.4f} "
                     f"{m['complementarity']:>10.4f}")
    return "\n".join(lines)


def fmt_triple(triple: dict, label: str) -> str:
    if triple is None:
        return ""
    lines = [f"\n--- Triple [{label}] (n={triple['n']:,}) ---"]
    lines.append(f"  All UP    (3/3): {triple['all_up_3_3'] * 100:>5.1f}%")
    lines.append(f"  Maj UP    (2/3): {triple['majority_up_2_3'] * 100:>5.1f}%")
    lines.append(f"  Maj DOWN  (1/3): {triple['majority_down_2_3'] * 100:>5.1f}%")
    lines.append(f"  All DOWN  (0/3): {triple['all_down_3_3'] * 100:>5.1f}%")
    lines.append(f"  Unanimity rate:  {triple['unanimity_rate'] * 100:>5.1f}%")
    lines.append(f"  Majority vote DirMatch: {triple['majority_vote_dirmatch'] * 100:>5.2f}%  "
                 f"(individual mean: {triple['mean_individual_dirmatch'] * 100:>5.2f}%, "
                 f"delta: {triple['majority_vs_individual_delta'] * 100:+5.2f}%)")
    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rsi-data", required=True, help="Dataset RSI .npz")
    parser.add_argument("--macd-data", required=True, help="Dataset MACD .npz")
    parser.add_argument("--cci-data", required=True, help="Dataset CCI .npz")
    parser.add_argument("--rsi-ckpt", required=True, help="Checkpoint RSI .pt")
    parser.add_argument("--macd-ckpt", required=True, help="Checkpoint MACD .pt")
    parser.add_argument("--cci-ckpt", required=True, help="Checkpoint CCI .pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output", default=str(ROOT / "results" / "specialists_analysis.json"))
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    indicators = {
        "rsi":  (args.rsi_data,  args.rsi_ckpt),
        "macd": (args.macd_data, args.macd_ckpt),
        "cci":  (args.cci_data,  args.cci_ckpt),
    }

    # Prédictions per indicator
    raw = {}
    for name, (data_path, ckpt_path) in indicators.items():
        print(f"\n--- {name.upper()} ---")
        print(f"  dataset : {data_path}")
        data, meta = load_dataset(data_path)
        print(f"  meta    : indicator={meta.get('indicator')} window={meta.get('window')}")
        print(f"  ckpt    : {ckpt_path}")
        yhat, y_true = predict_test(ckpt_path, data, device,
                                    args.batch_size, args.num_workers)
        ts = np.asarray(data["timestamp_test"], dtype=np.int64)
        regime = np.asarray(data["regime_test"], dtype=np.int8)
        asset_id = np.asarray(data["asset_id_test"], dtype=np.int8)
        print(f"  test    : N={len(yhat):,}  "
              f"y_true range=[{y_true.min():.4e}, {y_true.max():.4e}]  "
              f"yhat range=[{yhat.min():.4e}, {yhat.max():.4e}]")
        raw[name] = {"yhat": yhat, "y_true": y_true,
                     "ts": ts, "regime": regime, "asset_id": asset_id}

    # Alignement par timestamp
    print("\n=== ALIGNEMENT PAR TIMESTAMP ===")
    sizes_before = {k: len(v["ts"]) for k, v in raw.items()}
    aligned = align_by_timestamp(raw)
    n_common = len(next(iter(aligned.values()))["ts"])
    print(f"  Common timestamps : {n_common:,}")
    for name, n_orig in sizes_before.items():
        ratio = n_common / n_orig * 100 if n_orig else 0
        print(f"  {name}: {n_orig:,} → {n_common:,} ({ratio:.2f}%)")

    # Per-indicator metrics
    print("\n" + "=" * 100)
    print("PER-INDICATOR METRICS (test set, aligned)")
    print("=" * 100)
    per_ind = per_indicator_analysis(aligned)
    print(fmt_per_indicator(per_ind))

    # Inter-indicator agreement
    print("=" * 100)
    print("INTER-INDICATOR AGREEMENT")
    print("=" * 100)
    pairs_all = pairwise_agreement(aligned)
    print(fmt_pairwise(pairs_all, "ALL"))

    pairs_by_regime = {}
    for r_id, r_name in REGIME_NAMES.items():
        pr = pairwise_agreement(aligned, regime_filter=r_id)
        if pr:
            pairs_by_regime[r_name] = pr
            print(fmt_pairwise(pr, r_name))

    # Triple agreement
    print("\n" + "=" * 100)
    print("TRIPLE AGREEMENT")
    print("=" * 100)
    triple_all = triple_agreement(aligned)
    print(fmt_triple(triple_all, "ALL"))

    triple_by_regime = {}
    for r_id, r_name in REGIME_NAMES.items():
        t = triple_agreement(aligned, regime_filter=r_id)
        if t is not None:
            triple_by_regime[r_name] = t
            print(fmt_triple(t, r_name))

    # Save JSON
    output = {
        "n_common": n_common,
        "sizes_before_align": sizes_before,
        "per_indicator": per_ind,
        "pairwise_all": pairs_all,
        "pairwise_by_regime": pairs_by_regime,
        "triple_all": triple_all,
        "triple_by_regime": triple_by_regime,
        "ckpts": {n: c for n, (_, c) in indicators.items()},
        "datasets": {n: d for n, (d, _) in indicators.items()},
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, default=float))
    print(f"\n✓ Saved → {out_path}")


if __name__ == "__main__":
    main()
