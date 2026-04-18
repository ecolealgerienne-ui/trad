#!/usr/bin/env python3
"""
Clustering K-means non-supervisé sur les features Kalman AQ-KF du RSI.

Différent de cluster_model_flips.py :
  - Pas de labels (pas de is_profitable_flip) : purement non-supervisé
  - Fit sur features_train UNIQUEMENT
  - Predict sur train/val/test pour usage downstream

Pipeline :
  1. Charge kalman_rsi_features_30m.npz
  2. StandardScaler fit sur features_train
  3. K-means grid (K=5, 7, 10, 15, 20, 25)
  4. Pour chaque K :
     - Fit sur train scaled
     - Predict sur train/val/test
     - Stats descriptives par cluster (taille, centroïdes dénormalisés)
  5. Sauvegarde pickle par K pour usage étape 3 (transitions)

Sortie : models/kalman_clusters/kmeans_rsi_k<K>_30m.pkl
  - scaler : StandardScaler fit val
  - kmeans : KMeans model
  - cluster_ids_train/val/test : (N,) int
  - centroids_denorm : (K, 4) valeurs originales
  - feature_cols, K

Usage :
    python scripts/cluster_kalman_rsi.py
    python scripts/cluster_kalman_rsi.py --ks 5 10 15 20 --seed 42
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
except ImportError as e:
    print(f"❌ sklearn manquant : {e}")
    sys.exit(1)

PREP_DIR = Path('data/prepared')
MODELS_DIR = Path('models/kalman_clusters')


def analyze_clusters(K, labels, features_raw, feature_cols, split_name):
    """Affiche stats descriptives par cluster (pas de labels supervisés)."""
    print(f"\n  [{split_name}] Distribution {K} clusters :")
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    for cid, cnt in zip(unique, counts):
        pct = cnt / total * 100
        # Centroïde dénormalisé = mean des features raw des rows du cluster
        mask = labels == cid
        centroid = features_raw[mask].mean(axis=0)
        bar = '█' * int(pct / 3)
        print(f"    Cluster {int(cid):>2} : {cnt:>6,} ({pct:>5.2f}%)  {bar}")
        row = "      "
        for i, f in enumerate(feature_cols):
            row += f"{f}={centroid[i]:+.4f}  "
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', default='data/prepared/kalman_rsi_features_30m.npz',
                        help='NPZ features généré par prepare_kalman_rsi_features.py')
    parser.add_argument('--ks', type=int, nargs='+',
                        default=[5, 7, 10, 15, 20, 25],
                        help='Grid de K à tester')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-init', type=int, default=10,
                        help='Nombre d\'initialisations K-means (best)')
    args = parser.parse_args()

    print("=" * 100)
    print(f"CLUSTERING K-MEANS NON-SUPERVISÉ — features Kalman AQ-KF du RSI")
    print(f"  NPZ : {args.npz}")
    print(f"  Ks  : {args.ks}  |  seed={args.seed}  |  n_init={args.n_init}")
    print("=" * 100)

    npz_path = Path(args.npz)
    if not npz_path.exists():
        print(f"❌ NPZ introuvable : {npz_path}")
        return

    # Charger features
    print(f"\n[1] Load features ...")
    ds = np.load(npz_path, allow_pickle=True)
    feat_train = ds['features_train']
    feat_val = ds['features_val']
    feat_test = ds['features_test']
    feature_cols = [str(c) for c in ds['feature_cols']]
    tf_minutes = int(ds['tf_minutes'])
    tf_label = f'{tf_minutes}m' if tf_minutes < 60 else '1h'
    print(f"   Features : {feature_cols}")
    print(f"   Train: {feat_train.shape}  |  Val: {feat_val.shape}  |  "
          f"Test: {feat_test.shape}")

    # Check NaN
    for name, arr in [('train', feat_train), ('val', feat_val), ('test', feat_test)]:
        n_nan = int(np.isnan(arr).any(axis=1).sum())
        if n_nan > 0:
            print(f"   ⚠️ {name}: {n_nan} rows avec NaN → clip à 0 pour clustering")
    feat_train = np.where(np.isnan(feat_train), 0.0, feat_train)
    feat_val = np.where(np.isnan(feat_val), 0.0, feat_val)
    feat_test = np.where(np.isnan(feat_test), 0.0, feat_test)

    # StandardScaler fit train
    print(f"\n[2] StandardScaler fit sur train ({len(feat_train):,} rows)")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(feat_train)
    X_val_scaled = scaler.transform(feat_val)
    X_test_scaled = scaler.transform(feat_test)
    print(f"   Scales train : mean={scaler.mean_}")
    print(f"                  std ={scaler.scale_}")

    # Grid K-means
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for K in args.ks:
        print(f"\n{'=' * 100}")
        print(f"K-MEANS K = {K}")
        print(f"{'=' * 100}")

        km = KMeans(n_clusters=K, random_state=args.seed,
                     n_init=args.n_init)
        km.fit(X_train_scaled)
        inertia = km.inertia_
        print(f"  Fit inertia (train) = {inertia:.2f}")

        # Predict sur 3 splits
        labels_train = km.predict(X_train_scaled)
        labels_val = km.predict(X_val_scaled)
        labels_test = km.predict(X_test_scaled)

        # Centroïdes dénormalisés (pour interprétation)
        centroids_denorm = scaler.inverse_transform(km.cluster_centers_)

        # Stats par cluster (train + val + test)
        analyze_clusters(K, labels_train, feat_train, feature_cols, 'TRAIN')
        analyze_clusters(K, labels_val, feat_val, feature_cols, 'VAL')
        analyze_clusters(K, labels_test, feat_test, feature_cols, 'TEST')

        # Centroïdes synthétiques
        print(f"\n  Centroïdes dénormalisés (KMeans) :")
        header = f"    {'Cluster':>8}"
        for f in feature_cols:
            header += f" {f:>14}"
        print(header)
        for k in range(K):
            row = f"    {k:>8}"
            for i in range(len(feature_cols)):
                row += f" {centroids_denorm[k, i]:>+14.4f}"
            print(row)

        # Sauvegarde
        save_path = MODELS_DIR / f'kmeans_rsi_k{K}_{tf_label}.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump({
                'scaler': scaler,
                'kmeans': km,
                'cluster_ids_train': labels_train.astype(np.int16),
                'cluster_ids_val': labels_val.astype(np.int16),
                'cluster_ids_test': labels_test.astype(np.int16),
                'centroids_denorm': centroids_denorm,
                'feature_cols': feature_cols,
                'K': K,
                'inertia': float(inertia),
                'tf_label': tf_label,
                'seed': args.seed,
            }, f)
        print(f"\n  ✅ Sauvé : {save_path}  "
              f"({save_path.stat().st_size / 1024:.1f} KB)")

    # Synthèse inertia (courbe elbow)
    print(f"\n{'=' * 100}")
    print(f"SYNTHÈSE — courbe inertia (elbow) pour choix K")
    print(f"{'=' * 100}")
    print(f"   {'K':>3}  {'Inertia':>14}")
    for K in args.ks:
        pkl = MODELS_DIR / f'kmeans_rsi_k{K}_{tf_label}.pkl'
        with open(pkl, 'rb') as f:
            data = pickle.load(f)
        print(f"   {K:>3}  {data['inertia']:>14.2f}")


if __name__ == '__main__':
    main()
