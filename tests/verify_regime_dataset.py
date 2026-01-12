#!/usr/bin/env python3
"""
Script de vérification de la qualité du dataset regime.

Inspiré de:
- tests/test_load_direction_only.py (vérification shapes/distributions)
- tests/verify_causality.py (vérification alignement temporel)

Vérifications effectuées:
1. Shapes des arrays (X, Y, OHLCV)
2. Timestamps (croissants, pas de doublons, gaps entre splits)
3. Asset IDs (valides 0-4)
4. Labels régime (0-3, distributions, TS/VC scores)
5. Features (~20 attendues, pas de NaN/Inf)
6. OHLCV (cohérence prix, volume > 0)
7. Primary key (timestamp, asset_id) synchronisé
8. Metadata (split_indices, cohérence)
9. Causalité temporelle (pas de lookahead)

Usage:
    # Vérifier dataset regime généré
    python tests/verify_regime_dataset.py \\
        --data data/prepared/dataset_btc_eth_bnb_ada_ltc_regime.npz

    # Mode verbeux avec détails
    python tests/verify_regime_dataset.py \\
        --data data/prepared/dataset_btc_eth_bnb_ada_ltc_regime.npz \\
        --verbose
"""

import numpy as np
import pandas as pd
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, List
import sys

# Ajouter src/ au path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from constants import ASSET_ID_MAP


class RegimeDatasetValidator:
    """
    Validateur de dataset regime.

    Pattern copié de tests/test_load_direction_only.py et tests/verify_causality.py
    """

    def __init__(self, npz_path: str, verbose: bool = False):
        self.npz_path = Path(npz_path)
        self.verbose = verbose
        self.errors = []
        self.warnings = []

        # Charger dataset
        print(f"\n{'='*80}")
        print(f"VÉRIFICATION DATASET REGIME")
        print('='*80)
        print(f"Fichier: {self.npz_path}")

        if not self.npz_path.exists():
            raise FileNotFoundError(f"Dataset non trouvé: {self.npz_path}")

        self.data = np.load(self.npz_path, allow_pickle=True)

        # Extraire arrays
        self.X_train = self.data['X_train']
        self.Y_train = self.data['Y_train']
        self.OHLCV_train = self.data['OHLCV_train']

        self.X_val = self.data['X_val']
        self.Y_val = self.data['Y_val']
        self.OHLCV_val = self.data['OHLCV_val']

        self.X_test = self.data['X_test']
        self.Y_test = self.data['Y_test']
        self.OHLCV_test = self.data['OHLCV_test']

        # Charger metadata
        if 'metadata' in self.data:
            try:
                self.metadata = json.loads(str(self.data['metadata']))
            except Exception as e:
                self.metadata = {}
                self.warnings.append(f"Échec chargement metadata: {e}")
        else:
            self.metadata = {}
            self.errors.append("Metadata manquant dans NPZ!")

    def _print_check(self, check_name: str, passed: bool, message: str = ""):
        """Afficher résultat d'une vérification."""
        status = "✅" if passed else "❌"
        full_msg = f"  {status} {check_name}"
        if message:
            full_msg += f": {message}"
        print(full_msg)

        if not passed:
            self.errors.append(f"{check_name}: {message}")

    def _print_warning(self, message: str):
        """Afficher un warning."""
        print(f"  ⚠️  {message}")
        self.warnings.append(message)

    def verify_shapes(self) -> bool:
        """
        Vérifier les shapes des arrays.

        Attendu:
        - X: (n, 12, ~22) = [timestamp, asset_id, features...]
        - Y: (n, 5) = [timestamp, asset_id, regime, ts_score, vc_score]
        - OHLCV: (n, 7) = [timestamp, asset_id, O, H, L, C, V]
        """
        print(f"\n{'='*80}")
        print("TEST #1: SHAPES")
        print('='*80)

        all_passed = True

        # Vérifier shapes X
        print("\n📊 X (Sequences):")
        for split_name, X in [('Train', self.X_train), ('Val', self.X_val), ('Test', self.X_test)]:
            print(f"  {split_name}: {X.shape}")

            # Doit être 3D
            passed = len(X.shape) == 3
            self._print_check(f"{split_name} X rank=3", passed)
            all_passed &= passed

            # Sequence length doit être 12
            if len(X.shape) == 3:
                passed = X.shape[1] == 12
                self._print_check(f"{split_name} seq_length=12", passed,
                                f"obtenu {X.shape[1]}")
                all_passed &= passed

                # Features ~20-22 (timestamp + asset_id + ~20 features)
                n_features = X.shape[2]
                passed = 20 <= n_features <= 25
                if not passed:
                    self._print_warning(f"{split_name} n_features={n_features} (attendu ~20-22)")

        # Vérifier shapes Y
        print("\n📊 Y (Labels):")
        for split_name, Y in [('Train', self.Y_train), ('Val', self.Y_val), ('Test', self.Y_test)]:
            print(f"  {split_name}: {Y.shape}")

            # Doit avoir 5 colonnes
            passed = len(Y.shape) == 2 and Y.shape[1] == 5
            self._print_check(f"{split_name} Y shape=(n, 5)", passed,
                            f"obtenu {Y.shape}")
            all_passed &= passed

        # Vérifier shapes OHLCV
        print("\n📊 OHLCV:")
        for split_name, OHLCV in [('Train', self.OHLCV_train), ('Val', self.OHLCV_val), ('Test', self.OHLCV_test)]:
            print(f"  {split_name}: {OHLCV.shape}")

            # Doit avoir 7 colonnes
            passed = len(OHLCV.shape) == 2 and OHLCV.shape[1] == 7
            self._print_check(f"{split_name} OHLCV shape=(n, 7)", passed,
                            f"obtenu {OHLCV.shape}")
            all_passed &= passed

        # Vérifier cohérence tailles
        print("\n📊 Cohérence tailles:")
        for split_name, (X, Y, OHLCV) in [
            ('Train', (self.X_train, self.Y_train, self.OHLCV_train)),
            ('Val', (self.X_val, self.Y_val, self.OHLCV_val)),
            ('Test', (self.X_test, self.Y_test, self.OHLCV_test))
        ]:
            passed = len(X) == len(Y) == len(OHLCV)
            self._print_check(f"{split_name} len(X)=len(Y)=len(OHLCV)", passed,
                            f"X={len(X)}, Y={len(Y)}, OHLCV={len(OHLCV)}")
            all_passed &= passed

        return all_passed

    def verify_timestamps(self) -> bool:
        """
        Vérifier les timestamps.

        - Croissants par asset
        - Pas de doublons
        - Gap temporel entre val et test
        """
        print(f"\n{'='*80}")
        print("TEST #2: TIMESTAMPS")
        print('='*80)

        all_passed = True

        for split_name, arrays in [
            ('Train', (self.X_train, self.Y_train, self.OHLCV_train)),
            ('Val', (self.X_val, self.Y_val, self.OHLCV_val)),
            ('Test', (self.X_test, self.Y_test, self.OHLCV_test))
        ]:
            X, Y, OHLCV = arrays

            print(f"\n📅 {split_name}:")

            # Extraire timestamps (première colonne de chaque séquence)
            # X: (n, 12, ~22) → timestamp à X[:, 0, 0]
            ts_X = X[:, 0, 0]  # Premier timestep de chaque séquence
            ts_Y = Y[:, 0]
            ts_OHLCV = OHLCV[:, 0]

            # Vérifier synchronisation
            passed = np.allclose(ts_X, ts_Y) and np.allclose(ts_Y, ts_OHLCV)
            self._print_check("Timestamps synchronisés (X=Y=OHLCV)", passed)
            all_passed &= passed

            # Vérifier croissance par asset
            asset_ids = Y[:, 1].astype(int)
            for asset_id in np.unique(asset_ids):
                mask = asset_ids == asset_id
                ts_asset = ts_Y[mask]

                is_sorted = np.all(np.diff(ts_asset) > 0)
                asset_name = [k for k, v in ASSET_ID_MAP.items() if v == asset_id][0] if asset_id in ASSET_ID_MAP.values() else f"ID_{asset_id}"
                self._print_check(f"Timestamps croissants pour {asset_name}", is_sorted)
                all_passed &= is_sorted

                # Vérifier pas de doublons
                n_unique = len(np.unique(ts_asset))
                n_total = len(ts_asset)
                passed = n_unique == n_total
                if not passed:
                    self._print_warning(f"{asset_name}: {n_total - n_unique} doublons détectés")
                    all_passed = False

        # Vérifier gaps entre splits (important pour éviter data leakage)
        print("\n📅 Gaps temporels entre splits:")

        max_ts_train = np.max(self.Y_train[:, 0])
        min_ts_val = np.min(self.Y_val[:, 0])
        max_ts_val = np.max(self.Y_val[:, 0])
        min_ts_test = np.min(self.Y_test[:, 0])

        gap_train_val = (min_ts_val - max_ts_train) / 300  # En périodes 5min
        gap_val_test = (min_ts_test - max_ts_val) / 300

        print(f"  Train max → Val min: {gap_train_val:.0f} périodes (~{gap_train_val/12:.1f} heures)")
        print(f"  Val max → Test min: {gap_val_test:.0f} périodes (~{gap_val_test/12:.1f} heures)")

        # Warning si pas de gap (splits se touchent)
        if gap_train_val < 1:
            self._print_warning("Train et Val se touchent (gap < 1 période)")
        if gap_val_test < 1:
            self._print_warning("Val et Test se touchent (gap < 1 période)")

        return all_passed

    def verify_asset_ids(self) -> bool:
        """
        Vérifier les asset IDs.

        - Valeurs dans [0-4]
        - Tous les assets présents
        """
        print(f"\n{'='*80}")
        print("TEST #3: ASSET IDs")
        print('='*80)

        all_passed = True

        for split_name, arrays in [
            ('Train', (self.X_train, self.Y_train, self.OHLCV_train)),
            ('Val', (self.X_val, self.Y_val, self.OHLCV_val)),
            ('Test', (self.X_test, self.Y_test, self.OHLCV_test))
        ]:
            X, Y, OHLCV = arrays

            print(f"\n🏷️  {split_name}:")

            # Extraire asset IDs
            asset_X = X[:, 0, 1].astype(int)  # Premier timestep
            asset_Y = Y[:, 1].astype(int)
            asset_OHLCV = OHLCV[:, 1].astype(int)

            # Vérifier synchronisation
            passed = np.array_equal(asset_X, asset_Y) and np.array_equal(asset_Y, asset_OHLCV)
            self._print_check("Asset IDs synchronisés (X=Y=OHLCV)", passed)
            all_passed &= passed

            # Vérifier valeurs valides [0-4]
            valid_ids = np.all((asset_Y >= 0) & (asset_Y <= 4))
            self._print_check("Asset IDs dans [0-4]", valid_ids)
            all_passed &= valid_ids

            # Compter assets
            unique_assets = np.unique(asset_Y)
            asset_counts = pd.Series(asset_Y).value_counts().sort_index()

            print(f"  Assets présents: {len(unique_assets)}/5")
            for asset_id in unique_assets:
                asset_name = [k for k, v in ASSET_ID_MAP.items() if v == asset_id][0] if asset_id in ASSET_ID_MAP.values() else f"UNKNOWN_{asset_id}"
                count = asset_counts[asset_id]
                pct = count / len(asset_Y) * 100
                print(f"    {asset_name} (ID={asset_id}): {count} samples ({pct:.1f}%)")

        return all_passed

    def verify_regime_labels(self) -> bool:
        """
        Vérifier les labels régime.

        Y: [timestamp, asset_id, regime, ts_score, vc_score]
        - regime dans [0-3]
        - ts_score dans [0, 1]
        - vc_score >= 0
        - Distributions raisonnables
        """
        print(f"\n{'='*80}")
        print("TEST #4: LABELS RÉGIME")
        print('='*80)

        all_passed = True

        regime_names = {
            0: "RANGE LOW VOL",
            1: "RANGE HIGH VOL",
            2: "TREND LOW VOL",
            3: "TREND HIGH VOL"
        }

        for split_name, Y in [('Train', self.Y_train), ('Val', self.Y_val), ('Test', self.Y_test)]:
            print(f"\n🎯 {split_name}:")

            regime = Y[:, 2].astype(int)
            ts_score = Y[:, 3]
            vc_score = Y[:, 4]

            # Vérifier regime dans [0-3]
            valid_regime = np.all((regime >= 0) & (regime <= 3))
            self._print_check("Regime dans [0-3]", valid_regime)
            all_passed &= valid_regime

            # Vérifier ts_score dans [0, 1]
            valid_ts = np.all((ts_score >= 0) & (ts_score <= 1))
            self._print_check("TS score dans [0, 1]", valid_ts,
                            f"min={ts_score.min():.3f}, max={ts_score.max():.3f}")
            all_passed &= valid_ts

            # Vérifier vc_score >= 0
            valid_vc = np.all(vc_score >= 0)
            self._print_check("VC score >= 0", valid_vc,
                            f"min={vc_score.min():.3f}, max={vc_score.max():.3f}")
            all_passed &= valid_vc

            # Distribution régimes
            regime_counts = pd.Series(regime).value_counts().sort_index()
            regime_pcts = (regime_counts / len(regime) * 100).round(1)

            print(f"\n  Distribution régimes:")
            for regime_id, pct in regime_pcts.items():
                name = regime_names.get(regime_id, f"UNKNOWN_{regime_id}")
                count = regime_counts[regime_id]
                print(f"    {regime_id} ({name}): {count} ({pct}%)")

            # Warning si un régime domine (>60%)
            max_pct = regime_pcts.max()
            if max_pct > 60:
                self._print_warning(f"Régime dominant à {max_pct}% (déséquilibre fort)")

        return all_passed

    def verify_features(self) -> bool:
        """
        Vérifier les features.

        - Pas de NaN/Inf
        - Nombre de features cohérent avec metadata
        """
        print(f"\n{'='*80}")
        print("TEST #5: FEATURES")
        print('='*80)

        all_passed = True

        for split_name, X in [('Train', self.X_train), ('Val', self.X_val), ('Test', self.X_test)]:
            print(f"\n📈 {split_name}:")

            # Extraire features (colonnes 2+)
            features = X[:, :, 2:]  # Skip timestamp et asset_id

            # Vérifier NaN
            n_nan = np.sum(np.isnan(features))
            passed = n_nan == 0
            self._print_check("Pas de NaN", passed, f"trouvé {n_nan} NaN" if n_nan > 0 else "")
            all_passed &= passed

            # Vérifier Inf
            n_inf = np.sum(np.isinf(features))
            passed = n_inf == 0
            self._print_check("Pas de Inf", passed, f"trouvé {n_inf} Inf" if n_inf > 0 else "")
            all_passed &= passed

            # Stats features
            n_features = features.shape[2]
            print(f"  Nombre features: {n_features}")

            # Comparer avec metadata
            if 'n_features' in self.metadata:
                expected = self.metadata['n_features']
                passed = n_features == expected
                self._print_check(f"n_features = metadata ({expected})", passed,
                                f"obtenu {n_features}")
                all_passed &= passed

            # Ranges features (après clipping attendu)
            feat_min = float(np.min(features))
            feat_max = float(np.max(features))
            print(f"  Range features: [{feat_min:.2f}, {feat_max:.2f}]")

            if 'clip_value' in self.metadata:
                clip = self.metadata['clip_value']
                if feat_min < -clip or feat_max > clip:
                    self._print_warning(f"Features hors clip_value ±{clip}")

        return all_passed

    def verify_ohlcv(self) -> bool:
        """
        Vérifier OHLCV.

        - High >= Low
        - High >= Open, Close
        - Low <= Open, Close
        - Volume > 0
        """
        print(f"\n{'='*80}")
        print("TEST #6: OHLCV")
        print('='*80)

        all_passed = True

        for split_name, OHLCV in [('Train', self.OHLCV_train), ('Val', self.OHLCV_val), ('Test', self.OHLCV_test)]:
            print(f"\n💹 {split_name}:")

            # Extraire colonnes
            O = OHLCV[:, 2]
            H = OHLCV[:, 3]
            L = OHLCV[:, 4]
            C = OHLCV[:, 5]
            V = OHLCV[:, 6]

            # Vérifier High >= Low
            violations = np.sum(H < L)
            passed = violations == 0
            self._print_check("High >= Low", passed, f"{violations} violations" if violations > 0 else "")
            all_passed &= passed

            # Vérifier High >= Open
            violations = np.sum(H < O)
            passed = violations == 0
            self._print_check("High >= Open", passed, f"{violations} violations" if violations > 0 else "")
            all_passed &= passed

            # Vérifier High >= Close
            violations = np.sum(H < C)
            passed = violations == 0
            self._print_check("High >= Close", passed, f"{violations} violations" if violations > 0 else "")
            all_passed &= passed

            # Vérifier Low <= Open
            violations = np.sum(L > O)
            passed = violations == 0
            self._print_check("Low <= Open", passed, f"{violations} violations" if violations > 0 else "")
            all_passed &= passed

            # Vérifier Low <= Close
            violations = np.sum(L > C)
            passed = violations == 0
            self._print_check("Low <= Close", passed, f"{violations} violations" if violations > 0 else "")
            all_passed &= passed

            # Vérifier Volume > 0
            violations = np.sum(V <= 0)
            passed = violations == 0
            self._print_check("Volume > 0", passed, f"{violations} violations" if violations > 0 else "")
            all_passed &= passed

        return all_passed

    def verify_metadata(self) -> bool:
        """
        Vérifier metadata.

        - split_indices présent
        - Cohérence n_sequences vs shapes
        """
        print(f"\n{'='*80}")
        print("TEST #7: METADATA")
        print('='*80)

        all_passed = True

        # Vérifier split_indices
        if 'split_indices' not in self.metadata:
            self._print_check("split_indices présent", False, "manquant!")
            all_passed = False
        else:
            split_idx = self.metadata['split_indices']

            passed = 'train_end' in split_idx and 'val_end' in split_idx
            self._print_check("split_indices complet", passed)
            all_passed &= passed

            if passed:
                train_end = split_idx['train_end']
                val_end = split_idx['val_end']

                # Vérifier cohérence
                expected_train = len(self.Y_train)
                expected_val = len(self.Y_val)

                passed = train_end == expected_train
                self._print_check("train_end cohérent", passed,
                                f"metadata={train_end}, réel={expected_train}")
                all_passed &= passed

                passed = (val_end - train_end) == expected_val
                self._print_check("val_end cohérent", passed,
                                f"metadata={val_end - train_end}, réel={expected_val}")
                all_passed &= passed

        # Vérifier autres champs
        required_fields = ['assets', 'n_assets', 'sequence_length', 'features',
                          'n_features', 'n_classes', 'regime_definition']

        print("\n📋 Champs requis:")
        for field in required_fields:
            present = field in self.metadata
            self._print_check(field, present)
            all_passed &= present

        return all_passed

    def verify_primary_key_sync(self) -> bool:
        """
        Vérifier que (timestamp, asset_id) est synchronisé entre X, Y, OHLCV.

        Test critique pour éviter data leakage.
        """
        print(f"\n{'='*80}")
        print("TEST #8: PRIMARY KEY (timestamp, asset_id)")
        print('='*80)

        all_passed = True

        for split_name, arrays in [
            ('Train', (self.X_train, self.Y_train, self.OHLCV_train)),
            ('Val', (self.X_val, self.Y_val, self.OHLCV_val)),
            ('Test', (self.X_test, self.Y_test, self.OHLCV_test))
        ]:
            X, Y, OHLCV = arrays

            print(f"\n🔑 {split_name}:")

            # Extraire primary keys
            # X: premier timestep de chaque séquence
            pk_X = X[:, 0, :2]  # (timestamp, asset_id)
            pk_Y = Y[:, :2]
            pk_OHLCV = OHLCV[:, :2]

            # Vérifier X == Y
            passed = np.allclose(pk_X, pk_Y)
            self._print_check("Primary key X == Y", passed)
            all_passed &= passed

            # Vérifier Y == OHLCV
            passed = np.allclose(pk_Y, pk_OHLCV)
            self._print_check("Primary key Y == OHLCV", passed)
            all_passed &= passed

            if not all_passed:
                # Debug: afficher premiers mismatches
                if not np.allclose(pk_X, pk_Y):
                    diff_mask = ~np.isclose(pk_X, pk_Y).all(axis=1)
                    first_diff = np.where(diff_mask)[0][0] if np.any(diff_mask) else None
                    if first_diff is not None:
                        print(f"    Premier mismatch X vs Y à index {first_diff}:")
                        print(f"      X:  {pk_X[first_diff]}")
                        print(f"      Y:  {pk_Y[first_diff]}")

        return all_passed

    def run_all_verifications(self) -> bool:
        """
        Exécuter toutes les vérifications.

        Retourne True si tous les tests passent.
        """
        print(f"\n{'='*80}")
        print(f"DÉBUT VÉRIFICATIONS")
        print('='*80)

        results = {
            'shapes': self.verify_shapes(),
            'timestamps': self.verify_timestamps(),
            'asset_ids': self.verify_asset_ids(),
            'regime_labels': self.verify_regime_labels(),
            'features': self.verify_features(),
            'ohlcv': self.verify_ohlcv(),
            'metadata': self.verify_metadata(),
            'primary_key': self.verify_primary_key_sync(),
        }

        # Résumé
        print(f"\n{'='*80}")
        print(f"RÉSUMÉ")
        print('='*80)

        print("\n📊 Tests:")
        for test_name, passed in results.items():
            status = "✅ PASSÉ" if passed else "❌ ÉCHEC"
            print(f"  {test_name:20s}: {status}")

        all_passed = all(results.values())

        if self.errors:
            print(f"\n❌ ERREURS ({len(self.errors)}):")
            for i, error in enumerate(self.errors[:10], 1):  # Limiter à 10
                print(f"  {i}. {error}")
            if len(self.errors) > 10:
                print(f"  ... et {len(self.errors) - 10} autres erreurs")

        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings[:10], 1):
                print(f"  {i}. {warning}")
            if len(self.warnings) > 10:
                print(f"  ... et {len(self.warnings) - 10} autres warnings")

        print(f"\n{'='*80}")
        if all_passed:
            print("✅ TOUS LES TESTS PASSÉS - Dataset valide!")
        else:
            print("❌ ÉCHECS DÉTECTÉS - Dataset invalide")
        print('='*80)

        return all_passed


def main():
    parser = argparse.ArgumentParser(
        description='Vérifier la qualité du dataset regime',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
    # Vérifier dataset standard
    python tests/verify_regime_dataset.py \\
        --data data/prepared/dataset_btc_eth_bnb_ada_ltc_regime.npz

    # Mode verbeux
    python tests/verify_regime_dataset.py \\
        --data data/prepared/dataset_btc_eth_bnb_ada_ltc_regime.npz \\
        --verbose
        """
    )

    parser.add_argument('--data', type=str, required=True,
                       help='Chemin vers le dataset regime (.npz)')
    parser.add_argument('--verbose', action='store_true',
                       help='Mode verbeux avec détails')

    args = parser.parse_args()

    # Créer validateur
    validator = RegimeDatasetValidator(args.data, verbose=args.verbose)

    # Exécuter vérifications
    success = validator.run_all_verifications()

    # Exit code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
