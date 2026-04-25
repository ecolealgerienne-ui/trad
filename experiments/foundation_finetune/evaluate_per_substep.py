"""Per-substep evaluation of Chronos LoRA vs FLKS sub-step.

Replique la metrique de src/signal_processing/flks_substep_convergence.py :
    sign_concordance(slopes_test, slopes_oracle) = % accord de signe vs Oracle
    (excluant NaN et zeros oracle)

Pour chaque sous-pas k=0..5 (= notation step_k de prepare_features_and_labels_progressive,
equivalent k=1..6 de flks_substep_convergence.py), filtre les samples ou step_k == k
et calcule :
    - n samples
    - sign concordance (= DirMatch * 100)
    - Pearson
    - MSE

Comparaison vs FLKS pur :
    Le X[t] (input Chronos) contient deja la slope FLKS au sous-pas k actuel.
    Le baseline "FLKS sub-step" est donc : yhat = X[t][-1] (derniere valeur de la fenetre).
    On compare Chronos vs ce baseline FLKS.

Usage:
    python experiments/foundation_finetune/evaluate_per_substep.py \\
        --data data/foundation/rsi_btc_5min_flks_substep.npz \\
        --ckpt models/foundation_finetune_flks/chronos-t5-tiny_lora.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import ChronosRegressor
from train import SlopeDataset


ROOT = Path(__file__).resolve().parents[2]
EPSILON = 1e-8


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return p.parse_args()


def sign_concordance(yhat, y):
    """% accord de signe (excl NaN et zeros y). Format flks_substep_convergence."""
    yhat = np.asarray(yhat, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = ~np.isnan(yhat) & ~np.isnan(y) & (np.abs(y) > EPSILON)
    n = int(mask.sum())
    if n == 0:
        return float("nan"), 0
    return float(np.mean(np.sign(yhat[mask]) == np.sign(y[mask])) * 100.0), n


def metrics(yhat, y):
    yhat = np.asarray(yhat, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mse = float(np.mean((yhat - y) ** 2))
    mae = float(np.mean(np.abs(yhat - y)))
    pearson = (0.0 if (yhat.std() < 1e-12 or y.std() < 1e-12)
               else float(np.corrcoef(yhat, y)[0, 1]))
    sc, n_sc = sign_concordance(yhat, y)
    return {"n": int(len(y)), "mse": mse, "mae": mae,
            "pearson": pearson, "sign_conc_pct": sc, "n_sc": n_sc}


def load_model(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args_ck = ckpt["args"]
    extra_dim = int(ckpt.get("extra_dim", 0))
    kwargs = dict(model_name=args_ck["model"], head_hidden=args_ck.get("head_hidden", 64),
                  extra_dim=extra_dim, device=device)
    mode = args_ck["mode"]
    if mode == "probing":
        kwargs.update(freeze_backbone=True, use_lora=False)
    elif mode == "lora":
        kwargs.update(freeze_backbone=True, use_lora=True,
                      lora_rank=args_ck.get("lora_rank", 8))
    else:
        kwargs.update(freeze_backbone=False, use_lora=False)

    model = ChronosRegressor(**kwargs).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt, extra_dim


def predict_model(model, X, extras, device, batch_size, num_workers):
    has_extras = extras is not None and model.extra_dim > 0
    ds = SlopeDataset(X, np.zeros(len(X), dtype=np.float32),
                      extras=extras if has_extras else None)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(device == "cuda"))
    preds = []
    with torch.no_grad():
        for batch in loader:
            if has_extras:
                x, ex, _ = batch
                preds.append(model(x.to(device, non_blocking=True),
                                   ex.to(device, non_blocking=True)).cpu().numpy())
            else:
                x, _ = batch
                preds.append(model(x.to(device, non_blocking=True)).cpu().numpy())
    return np.concatenate(preds)


def print_table(rows):
    header = (f"{'predictor':<22} {'n':>8} {'sign_conc%':>11} {'pearson':>9} "
              f"{'mse':>9} {'mae':>9}")
    print(header)
    print("-" * len(header))
    for name, m in rows:
        sc = m["sign_conc_pct"]
        sc_str = f"{sc:>10.2f}%" if not np.isnan(sc) else f"{'nan':>11}"
        print(f"{name:<22} {m['n']:>8,} {sc_str} "
              f"{m['pearson']:>+9.4f} {m['mse']:>9.5f} {m['mae']:>9.5f}")


def main():
    args = parse_args()
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}\nLoading {args.data}...")

    data = np.load(args.data, allow_pickle=True)
    meta = json.loads(str(data["meta"]))
    print(f"  meta: indicator={meta.get('indicator')} tf_minutes={meta.get('tf_minutes')} "
          f"window={meta.get('window')} adaptive={meta.get('adaptive')}")

    if f"step_k_{args.split}" not in data.files:
        print(f"\nERROR: step_k_{args.split} not in npz. Regenerate dataset with the "
              f"updated build_dataset_flks_substep.py (commit 8946254+).")
        sys.exit(1)

    X = data[f"X_{args.split}"]
    y = data[f"y_{args.split}"]
    step_k = data[f"step_k_{args.split}"]
    extras = data[f"extras_{args.split}"] if f"extras_{args.split}" in data.files else None
    print(f"  {args.split}: X={X.shape} y={y.shape} step_k={step_k.shape}  "
          f"k distribution={dict(zip(*np.unique(step_k, return_counts=True)))}")

    # --- Load Chronos checkpoint
    print(f"\nLoading checkpoint {args.ckpt}...")
    model, ckpt, extra_dim = load_model(Path(args.ckpt), device)
    print(f"  trainable={model.count_trainable():,}  extra_dim={extra_dim}  "
          f"epoch={ckpt.get('epoch', '?')}  val_mse={ckpt.get('val_metrics', {}).get('mse', '?')}")

    # --- Predictions
    print(f"\nRunning Chronos inference on {args.split} ({len(y):,} samples)...")
    yhat_chronos = predict_model(model, X, extras, device,
                                 args.batch_size, args.num_workers)

    # --- Baseline FLKS sub-step : la valeur a t (derniere de la fenetre X) IS l'estimation FLKS
    # apres z-score sur train. Donc on de-normalise pour comparer dans la meme echelle que y.
    norm = meta.get("norm_stats", {}).get("slope_progressive")
    if norm is None:
        print("WARNING: meta.norm_stats.slope_progressive missing, FLKS baseline skipped.")
        yhat_flks = None
    else:
        mean_n, std_n = norm
        yhat_flks = X[:, -1].astype(np.float64) * std_n + mean_n  # de-normalize

    # --- Global metrics
    print(f"\n=== GLOBAL ({args.split}, n={len(y):,}) ===")
    rows = [("Chronos LoRA", metrics(yhat_chronos, y))]
    if yhat_flks is not None:
        rows.append(("FLKS (slope_prog)", metrics(yhat_flks, y)))
    print_table(rows)

    # --- Per sub-step metrics (k=0..5 = notation flks_substep_convergence k=1..6)
    unique_k = sorted(np.unique(step_k).tolist())
    print(f"\n=== PER SUB-STEP ({args.split}) — sign concordance vs Oracle ===")
    print(f"   step_k=0 corresponds to FLKS k=1 (first 5min sub-step)")
    print(f"   step_k=5 corresponds to FLKS k=6 (last 5min sub-step in 30min bar)\n")

    header_row = (f"{'k':>3} {'n':>8} {'Chronos sc%':>13} {'Chronos pearson':>16} "
                  f"{'FLKS sc%':>10} {'FLKS pearson':>14}")
    print(header_row)
    print("-" * len(header_row))
    summary = {}
    for k in unique_k:
        mask = (step_k == k)
        n_k = int(mask.sum())
        if n_k < 10:
            continue
        m_c = metrics(yhat_chronos[mask], y[mask])
        m_f = metrics(yhat_flks[mask], y[mask]) if yhat_flks is not None else None
        sc_c = m_c["sign_conc_pct"]
        pe_c = m_c["pearson"]
        if m_f is not None:
            sc_f = m_f["sign_conc_pct"]
            pe_f = m_f["pearson"]
            print(f"{int(k):>3} {n_k:>8,} {sc_c:>12.2f}% {pe_c:>+16.4f} "
                  f"{sc_f:>9.2f}% {pe_f:>+14.4f}")
        else:
            print(f"{int(k):>3} {n_k:>8,} {sc_c:>12.2f}% {pe_c:>+16.4f}")
        summary[int(k)] = {"chronos": m_c, "flks": m_f}

    # --- Save JSON
    out_path = ROOT / "data" / "foundation" / f"per_substep_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "ckpt": str(args.ckpt),
        "split": args.split,
        "global": {name: m for name, m in rows},
        "per_substep": summary,
    }, indent=2, default=float))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
