#!/usr/bin/env python3
"""
Script de validation de dataset .npz

Vérifie:
1. Shapes correctes (X, Y, T, OHLCV)
2. Pas de NaN
3. Distributions (Direction ~50%, Transitions ~15%)
4. Alignement timestamps entre X, Y, T, OHLCV
5. Métadonnées présentes
6. Intégrité OHLCV (comparaison avec CSV brut)
"""

import numpy as np
import pandas as pd
import json
import argparse
import sys
from pathlib import Path

# Ajouter src/ au path pour importer constants
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from constants import AVAILABLE_ASSETS_5M


def validate_dataset(npz_path: str, verbose: bool = True):
    """Valide un dataset .npz généré par prepare_data_direction_only.py"""

    print(f"\n{'='*80}")
    print(f"VALIDATION: {Path(npz_path).name}")
    print('='*80)

    # Charger
    data = np.load(npz_path, allow_pickle=True)

    errors = []
    warnings = []

    # ========================================================================
    # 1. VÉRIFICATION PRÉSENCE DES CLÉS
    # ========================================================================
    required_keys = [
        'X_train', 'Y_train', 'T_train', 'OHLCV_train',
        'X_val', 'Y_val', 'T_val', 'OHLCV_val',
        'X_test', 'Y_test', 'T_test', 'OHLCV_test',
        'metadata'
    ]

    missing_keys = [k for k in required_keys if k not in data]
    if missing_keys:
        errors.append(f"Clés manquantes: {missing_keys}")
        return errors, warnings

    print("✅ Toutes les clés présentes")

    # ========================================================================
    # 2. VÉRIFICATION SHAPES
    # ========================================================================
    print(f"\n{'─'*80}")
    print("SHAPES")
    print('─'*80)

    for split in ['train', 'val', 'test']:
        X = data[f'X_{split}']
        Y = data[f'Y_{split}']
        T = data[f'T_{split}']
        OHLCV = data[f'OHLCV_{split}']

        n = len(X)

        print(f"\n{split.upper()}:")
        print(f"  X: {X.shape}")
        print(f"  Y: {Y.shape}")
        print(f"  T: {T.shape}")
        print(f"  OHLCV: {OHLCV.shape}")

        # Vérifier cohérence longueurs
        if not (len(X) == len(Y) == len(T) == len(OHLCV)):
            errors.append(f"{split}: Longueurs différentes X={len(X)}, Y={len(Y)}, T={len(T)}, OHLCV={len(OHLCV)}")

        # Vérifier dimensions
        if X.ndim != 3:
            errors.append(f"{split}: X doit être 3D, obtenu {X.ndim}D")

        if Y.shape != (n, 3):
            errors.append(f"{split}: Y doit être (n, 3), obtenu {Y.shape}")

        if T.shape != (n, 3):
            errors.append(f"{split}: T doit être (n, 3), obtenu {T.shape}")

        if OHLCV.shape != (n, 7):
            errors.append(f"{split}: OHLCV doit être (n, 7), obtenu {OHLCV.shape}")

    if not errors:
        print("\n✅ Shapes correctes")

    # ========================================================================
    # 3. VÉRIFICATION NaN
    # ========================================================================
    print(f"\n{'─'*80}")
    print("NaN CHECK")
    print('─'*80)

    for split in ['train', 'val', 'test']:
        X = data[f'X_{split}']
        Y = data[f'Y_{split}']
        T = data[f'T_{split}']
        OHLCV = data[f'OHLCV_{split}']

        nan_counts = {
            'X': np.isnan(X).sum(),
            'Y': np.isnan(Y).sum(),
            'T': np.isnan(T).sum(),
            'OHLCV': np.isnan(OHLCV).sum()
        }

        total_nans = sum(nan_counts.values())

        if total_nans > 0:
            errors.append(f"{split}: {total_nans} NaN trouvés - {nan_counts}")
            print(f"❌ {split.upper()}: {total_nans} NaN trouvés")
            for key, count in nan_counts.items():
                if count > 0:
                    print(f"   {key}: {count} NaN")
        else:
            print(f"✅ {split.upper()}: 0 NaN")

    # ========================================================================
    # 4. VÉRIFICATION DISTRIBUTIONS
    # ========================================================================
    print(f"\n{'─'*80}")
    print("DISTRIBUTIONS")
    print('─'*80)

    for split in ['train', 'val', 'test']:
        Y = data[f'Y_{split}']
        T = data[f'T_{split}']

        # Direction (colonne 2 de Y)
        direction = Y[:, 2]
        dir_pct = direction.mean() * 100

        # Transitions (colonne 2 de T)
        transitions = T[:, 2]
        trans_pct = transitions.mean() * 100

        print(f"\n{split.upper()}:")
        print(f"  Direction: {dir_pct:.1f}% UP")
        print(f"  Transitions: {trans_pct:.1f}%")

        # Warning si déséquilibre direction
        if not (45 <= dir_pct <= 55):
            warnings.append(f"{split}: Direction déséquilibrée ({dir_pct:.1f}% UP, attendu ~50%)")

        # Warning si transitions anormales
        if not (10 <= trans_pct <= 20):
            warnings.append(f"{split}: Transitions anormales ({trans_pct:.1f}%, attendu 10-20%)")

    # ========================================================================
    # 5. VÉRIFICATION ALIGNEMENT TIMESTAMPS
    # ========================================================================
    print(f"\n{'─'*80}")
    print("ALIGNEMENT TIMESTAMPS")
    print('─'*80)

    for split in ['train', 'val', 'test']:
        X = data[f'X_{split}']
        Y = data[f'Y_{split}']
        T = data[f'T_{split}']
        OHLCV = data[f'OHLCV_{split}']

        # Extraire timestamps (colonne 0)
        # X: dernière timestep de la séquence (X[:, -1, 0])
        # Y, T, OHLCV: colonne 0
        ts_X = X[:, -1, 0]  # Dernière timestep de chaque séquence
        ts_Y = Y[:, 0]
        ts_T = T[:, 0]
        ts_OHLCV = OHLCV[:, 0]

        # Vérifier égalité
        if not np.allclose(ts_Y, ts_T):
            errors.append(f"{split}: Timestamps Y != T")

        if not np.allclose(ts_Y, ts_OHLCV):
            errors.append(f"{split}: Timestamps Y != OHLCV")

        # Pour X, on ne peut pas vérifier directement car c'est une séquence
        # Mais on peut vérifier que les timestamps sont croissants
        for i in range(len(X)):
            seq_ts = X[i, :, 0]
            if not np.all(seq_ts[:-1] <= seq_ts[1:]):
                errors.append(f"{split}: Timestamps non croissants dans séquence {i}")
                break

        if verbose:
            print(f"\n{split.upper()}:")
            print(f"  Timestamps Y vs T: {'✅ OK' if np.allclose(ts_Y, ts_T) else '❌ ERREUR'}")
            print(f"  Timestamps Y vs OHLCV: {'✅ OK' if np.allclose(ts_Y, ts_OHLCV) else '❌ ERREUR'}")

    if not errors:
        print("\n✅ Alignement timestamps correct")

    # ========================================================================
    # 6. VÉRIFICATION MÉTADONNÉES
    # ========================================================================
    print(f"\n{'─'*80}")
    print("MÉTADONNÉES")
    print('─'*80)

    # Charger metadata (stocké en JSON string dans le .npz)
    metadata_raw = data['metadata']

    # Si c'est un array numpy, extraire l'item
    if hasattr(metadata_raw, 'item'):
        metadata_str = metadata_raw.item()
    else:
        metadata_str = metadata_raw

    # Parser le JSON string en dict
    try:
        if isinstance(metadata_str, str):
            metadata = json.loads(metadata_str)
        elif isinstance(metadata_str, dict):
            metadata = metadata_str  # Déjà un dict
        else:
            raise ValueError(f"Type inattendu: {type(metadata_str)}")
    except Exception as e:
        errors.append(f"Erreur parsing métadonnées: {e}")
        print(f"❌ Métadonnées: erreur parsing ({e})")
        print(f"   Type: {type(metadata_str).__name__}")
        metadata = {}

    # Vérifier les champs requis
    if metadata:
        required_meta = [
            'version', 'architecture', 'indicator', 'assets',
            'filter_type', 'trim_edges', 'labels', 'label_names'
        ]

        missing_meta = [k for k in required_meta if k not in metadata]
        if missing_meta:
            warnings.append(f"Métadonnées manquantes: {missing_meta}")

        if verbose:
            print(f"\nIndicateur: {metadata.get('indicator', 'N/A')}")
            print(f"Assets: {metadata.get('assets', 'N/A')}")
            print(f"Filtre: {metadata.get('filter_type', 'N/A')}")
            print(f"TRIM edges: {metadata.get('trim_edges', 'N/A')}")
            print(f"Labels: {metadata.get('label_names', 'N/A')}")

        if not missing_meta:
            print("\n✅ Métadonnées complètes")

    # ========================================================================
    # 7. VALIDATION INTÉGRITÉ OHLCV (Comparaison avec CSV brut)
    # ========================================================================
    print(f"\n{'─'*80}")
    print("INTÉGRITÉ OHLCV (vs CSV brut)")
    print('─'*80)

    # Récupérer l'asset depuis les métadonnées
    if metadata and 'assets' in metadata:
        assets = metadata['assets']
        if not assets:
            warnings.append("Pas d'assets dans métadonnées, skip validation OHLCV")
        else:
            # Prendre le premier asset (pour BTC uniquement par exemple)
            asset_name = assets[0]

            # Trouver le fichier CSV brut
            if asset_name not in AVAILABLE_ASSETS_5M:
                warnings.append(f"Asset {asset_name} non trouvé dans AVAILABLE_ASSETS_5M")
            else:
                csv_path = AVAILABLE_ASSETS_5M[asset_name]

                try:
                    # Charger CSV brut
                    df_raw = pd.read_csv(csv_path, parse_dates=['timestamp'])
                    df_raw.set_index('timestamp', inplace=True)

                    # Vérifier qu'on a assez de lignes (> 1000)
                    if len(df_raw) < 1000:
                        warnings.append(f"CSV brut trop petit ({len(df_raw)} lignes), skip validation OHLCV")
                    else:
                        print(f"\nAsset: {asset_name}")
                        print(f"CSV brut: {len(df_raw)} lignes")

                        # Prendre 5 timestamps aléatoires (index > 1000)
                        np.random.seed(42)  # Reproductible
                        valid_indices = np.arange(1000, len(df_raw))
                        sample_indices = np.random.choice(valid_indices, size=5, replace=False)
                        sample_indices = sorted(sample_indices)  # Trier pour affichage

                        print(f"Échantillons testés: {sample_indices}")

                        # Pour chaque split, vérifier les 5 timestamps
                        for split in ['train', 'val', 'test']:
                            OHLCV_split = data[f'OHLCV_{split}']

                            # Timestamps du split (colonne 0)
                            ts_split = OHLCV_split[:, 0].astype('datetime64[ns]')

                            found_count = 0
                            mismatches = []

                            for idx in sample_indices:
                                # Valeurs brutes
                                ts_raw = df_raw.index[idx]
                                ohlcv_raw = df_raw.iloc[idx][['open', 'high', 'low', 'close', 'volume']].values

                                # Chercher dans le split par timestamp
                                # Convertir ts_raw en int64 (nanosecondes)
                                ts_raw_int64 = pd.Timestamp(ts_raw).value

                                # Trouver l'index dans le split
                                matches = np.where(OHLCV_split[:, 0] == ts_raw_int64)[0]

                                if len(matches) == 0:
                                    # Pas trouvé (probablement à cause du TRIM)
                                    continue

                                found_count += 1
                                match_idx = matches[0]

                                # Valeurs traitées (colonnes 2-6 = O, H, L, C, V)
                                ohlcv_processed = OHLCV_split[match_idx, 2:]

                                # Comparer avec tolérance
                                if not np.allclose(ohlcv_raw, ohlcv_processed, rtol=1e-6):
                                    mismatches.append({
                                        'idx': idx,
                                        'timestamp': ts_raw,
                                        'raw': ohlcv_raw,
                                        'processed': ohlcv_processed
                                    })

                            # Afficher résultats
                            if found_count == 0:
                                warnings.append(f"{split}: Aucun timestamp trouvé (probablement tous TRIMés)")
                                print(f"  {split.upper()}: ⚠️  0/5 timestamps trouvés (TRIMés?)")
                            elif mismatches:
                                errors.append(f"{split}: {len(mismatches)}/{found_count} valeurs OHLCV différentes!")
                                print(f"  {split.upper()}: ❌ {len(mismatches)}/{found_count} différences")
                                if verbose:
                                    for mm in mismatches[:2]:  # Afficher max 2 exemples
                                        print(f"    Idx {mm['idx']}: {mm['timestamp']}")
                                        print(f"      Raw:       {mm['raw']}")
                                        print(f"      Processed: {mm['processed']}")
                            else:
                                print(f"  {split.upper()}: ✅ {found_count}/5 OK (valeurs identiques)")

                        if not mismatches:
                            print("\n✅ Intégrité OHLCV validée")

                except Exception as e:
                    warnings.append(f"Erreur chargement CSV {csv_path}: {e}")
                    print(f"⚠️  Erreur: {e}")
    else:
        warnings.append("Pas de métadonnées, skip validation OHLCV")

    # ========================================================================
    # RÉSUMÉ
    # ========================================================================
    print(f"\n{'='*80}")
    print("RÉSUMÉ VALIDATION")
    print('='*80)

    if errors:
        print(f"\n❌ {len(errors)} ERREUR(S) CRITIQUE(S):")
        for i, err in enumerate(errors, 1):
            print(f"  {i}. {err}")

    if warnings:
        print(f"\n⚠️  {len(warnings)} WARNING(S):")
        for i, warn in enumerate(warnings, 1):
            print(f"  {i}. {warn}")

    if not errors and not warnings:
        print("\n✅ DATASET VALIDE - Aucune erreur détectée")
    elif not errors:
        print("\n⚠️  DATASET UTILISABLE - Warnings mineurs uniquement")
    else:
        print("\n❌ DATASET INVALIDE - Corrections requises")

    return errors, warnings


def main():
    parser = argparse.ArgumentParser(description='Valider un dataset .npz')
    parser.add_argument('--data', type=str, required=True, help='Chemin vers le .npz')
    parser.add_argument('--quiet', action='store_true', help='Mode silencieux (erreurs uniquement)')

    args = parser.parse_args()

    errors, warnings = validate_dataset(args.data, verbose=not args.quiet)

    # Exit code
    if errors:
        exit(1)
    elif warnings:
        exit(2)
    else:
        exit(0)


if __name__ == '__main__':
    main()
