#!/usr/bin/env python3
"""
Script de test: Validation de l'intégration des métadonnées dans les datasets.

Teste:
1. Structure des matrices (shapes, colonnes)
2. Alignement des clés primaires (timestamp, asset_id)
3. Cohérence des données OHLCV
4. Navigation entre matrices
5. Extraction des features pour ML

Usage:
    # Test rapide avec 1000 samples par asset
    python tests/test_metadata_integration.py --quick

    # Test complet avec tous les assets
    python tests/test_metadata_integration.py --assets BTC ETH
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import argparse
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Asset ID mapping (doit correspondre à prepare_data_direction_only.py)
ASSET_ID_MAP = {
    'BTC': 0,
    'ETH': 1,
    'BNB': 2,
    'ADA': 3,
    'LTC': 4
}

ASSET_ID_REVERSE = {v: k for k, v in ASSET_ID_MAP.items()}


class TestResult:
    """Résultat d'un test."""
    def __init__(self, name: str):
        self.name = name
        self.passed = True
        self.errors = []

    def fail(self, error: str):
        self.passed = False
        self.errors.append(error)

    def __str__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        result = f"{status} - {self.name}"
        if self.errors:
            result += "\n  Erreurs:"
            for err in self.errors:
                result += f"\n    - {err}"
        return result


def test_structure(data: dict, indicator: str) -> TestResult:
    """Test 1: Vérifier la structure des matrices."""
    test = TestResult("Structure des matrices")

    required_keys = [
        'X_train', 'Y_train', 'T_train', 'OHLCV_train',
        'X_val', 'Y_val', 'T_val', 'OHLCV_val',
        'X_test', 'Y_test', 'T_test', 'OHLCV_test',
        'metadata'
    ]

    for key in required_keys:
        if key not in data:
            test.fail(f"Clé manquante: {key}")

    if not test.passed:
        return test

    # Vérifier shapes
    for split in ['train', 'val', 'test']:
        X = data[f'X_{split}']
        Y = data[f'Y_{split}']
        T = data[f'T_{split}']
        OHLCV = data[f'OHLCV_{split}']

        # Toutes doivent avoir le même nombre de samples
        n_x = len(X)
        n_y = len(Y)
        n_t = len(T)
        n_ohlcv = len(OHLCV)

        if not (n_x == n_y == n_t == n_ohlcv):
            test.fail(f"{split}: Longueurs incohérentes - X:{n_x}, Y:{n_y}, T:{n_t}, OHLCV:{n_ohlcv}")

        # Vérifier dimensions
        if len(X.shape) != 3:
            test.fail(f"{split}: X doit avoir 3 dimensions, a {len(X.shape)}")

        if len(Y.shape) != 2 or Y.shape[1] != 3:
            test.fail(f"{split}: Y doit être (n, 3), est {Y.shape}")

        if len(T.shape) != 2 or T.shape[1] != 3:
            test.fail(f"{split}: T doit être (n, 3), est {T.shape}")

        if len(OHLCV.shape) != 2 or OHLCV.shape[1] != 7:
            test.fail(f"{split}: OHLCV doit être (n, 7), est {OHLCV.shape}")

        # Vérifier n_features dans X
        expected_features = {
            'rsi': 3,   # timestamp, asset_id, c_ret
            'macd': 3,  # timestamp, asset_id, c_ret
            'cci': 5    # timestamp, asset_id, h_ret, l_ret, c_ret
        }

        expected = expected_features.get(indicator, 3)
        if X.shape[2] != expected:
            test.fail(f"{split}: X shape[2] attendu {expected}, obtenu {X.shape[2]} (indicateur: {indicator})")

    return test


def test_alignment(data: dict) -> TestResult:
    """Test 2: Vérifier l'alignement des clés primaires (timestamp, asset_id)."""
    test = TestResult("Alignement des clés primaires")

    for split in ['train', 'val', 'test']:
        X = data[f'X_{split}']
        Y = data[f'Y_{split}']
        T = data[f'T_{split}']
        OHLCV = data[f'OHLCV_{split}']

        # Tester sur 100 samples aléatoires
        n_samples = len(X)
        test_indices = np.random.choice(n_samples, min(100, n_samples), replace=False)

        for i in test_indices:
            # Extraire timestamps
            ts_x = X[i, -1, 0]  # Dernier timestep de la séquence
            ts_y = Y[i, 0]
            ts_t = T[i, 0]
            ts_ohlcv = OHLCV[i, 0]

            # Extraire asset_ids
            aid_x = X[i, -1, 1]
            aid_y = Y[i, 1]
            aid_t = T[i, 1]
            aid_ohlcv = OHLCV[i, 1]

            # Vérifier alignement timestamps
            if not (ts_x == ts_y == ts_t == ts_ohlcv):
                test.fail(f"{split}[{i}]: Timestamps désalignés - X:{ts_x}, Y:{ts_y}, T:{ts_t}, OHLCV:{ts_ohlcv}")
                break

            # Vérifier alignement asset_ids
            if not (aid_x == aid_y == aid_t == aid_ohlcv):
                test.fail(f"{split}[{i}]: Asset IDs désalignés - X:{aid_x}, Y:{aid_y}, T:{aid_t}, OHLCV:{aid_ohlcv}")
                break

            # Vérifier que asset_id est valide
            if int(aid_x) not in ASSET_ID_REVERSE:
                test.fail(f"{split}[{i}]: Asset ID invalide: {aid_x}")
                break

    return test


def test_ohlcv_coherence(data: dict) -> TestResult:
    """Test 3: Vérifier la cohérence des données OHLCV."""
    test = TestResult("Cohérence des données OHLCV")

    for split in ['train', 'val', 'test']:
        OHLCV = data[f'OHLCV_{split}']

        # Tester sur quelques samples
        test_indices = np.random.choice(len(OHLCV), min(50, len(OHLCV)), replace=False)

        for i in test_indices:
            open_price = float(OHLCV[i, 2])
            high_price = float(OHLCV[i, 3])
            low_price = float(OHLCV[i, 4])
            close_price = float(OHLCV[i, 5])
            volume = float(OHLCV[i, 6])

            # Vérifier que high >= low
            if high_price < low_price:
                test.fail(f"{split}[{i}]: High ({high_price}) < Low ({low_price})")
                break

            # Vérifier que open/close sont entre low et high
            if not (low_price <= open_price <= high_price):
                test.fail(f"{split}[{i}]: Open ({open_price}) hors range [{low_price}, {high_price}]")
                break

            if not (low_price <= close_price <= high_price):
                test.fail(f"{split}[{i}]: Close ({close_price}) hors range [{low_price}, {high_price}]")
                break

            # Vérifier que les prix sont positifs
            if open_price <= 0 or high_price <= 0 or low_price <= 0 or close_price <= 0:
                test.fail(f"{split}[{i}]: Prix négatif ou nul détecté")
                break

            # Vérifier que le volume est positif
            if volume < 0:
                test.fail(f"{split}[{i}]: Volume négatif: {volume}")
                break

    return test


def test_labels_range(data: dict) -> TestResult:
    """Test 4: Vérifier que les labels sont dans les ranges attendus."""
    test = TestResult("Range des labels")

    for split in ['train', 'val', 'test']:
        Y = data[f'Y_{split}']
        T = data[f'T_{split}']

        # Y[:, 2] = direction (0 ou 1)
        directions = Y[:, 2].astype(int)
        unique_dirs = np.unique(directions)

        if not set(unique_dirs).issubset({0, 1}):
            test.fail(f"{split}: Direction contient des valeurs invalides: {unique_dirs}")

        # T[:, 2] = is_transition (0 ou 1)
        transitions = T[:, 2].astype(int)
        unique_trans = np.unique(transitions)

        if not set(unique_trans).issubset({0, 1}):
            test.fail(f"{split}: Transitions contiennent des valeurs invalides: {unique_trans}")

    return test


def test_ml_extraction(data: dict, indicator: str) -> TestResult:
    """Test 5: Vérifier l'extraction des features pour ML."""
    test = TestResult("Extraction features pour ML")

    try:
        X_train_full = data['X_train']
        Y_train_full = data['Y_train']

        # Extraire sans métadonnées
        X_train_ml = X_train_full[:, :, 2:]  # Supprimer timestamp et asset_id
        Y_train_ml = Y_train_full[:, 2:]     # Supprimer timestamp et asset_id

        # Vérifier shapes
        expected_features = {
            'rsi': 1,
            'macd': 1,
            'cci': 3
        }

        expected = expected_features.get(indicator, 1)
        if X_train_ml.shape[2] != expected:
            test.fail(f"Features ML: attendu {expected}, obtenu {X_train_ml.shape[2]}")

        if Y_train_ml.shape[1] != 1:
            test.fail(f"Labels ML: attendu shape (n, 1), obtenu {Y_train_ml.shape}")

        # Vérifier qu'il n'y a pas de NaN
        if np.isnan(X_train_ml.astype(float)).any():
            test.fail("Features ML contiennent des NaN")

        if np.isnan(Y_train_ml.astype(float)).any():
            test.fail("Labels ML contiennent des NaN")

    except Exception as e:
        test.fail(f"Erreur lors de l'extraction: {str(e)}")

    return test


def test_chronological_order(data: dict) -> TestResult:
    """Test 6: Vérifier l'ordre chronologique des timestamps."""
    test = TestResult("Ordre chronologique")

    for split in ['train', 'val', 'test']:
        Y = data[f'Y_{split}']

        timestamps = pd.to_datetime(Y[:, 0])

        # Vérifier que les timestamps sont triés (par asset)
        # Grouper par asset_id
        asset_ids = Y[:, 1].astype(int)

        for asset_id in np.unique(asset_ids):
            mask = asset_ids == asset_id
            ts_asset = timestamps[mask]

            if not ts_asset.is_monotonic_increasing:
                test.fail(f"{split}: Timestamps non triés pour asset {ASSET_ID_REVERSE.get(asset_id, asset_id)}")
                break

    return test


def run_all_tests(data_path: str) -> dict:
    """Exécute tous les tests sur un dataset."""
    logger.info(f"📂 Chargement: {data_path}")

    try:
        data = np.load(data_path, allow_pickle=True)
    except Exception as e:
        logger.error(f"❌ Erreur de chargement: {str(e)}")
        return None

    # Extraire l'indicateur du nom de fichier
    filename = Path(data_path).name
    if 'rsi' in filename.lower():
        indicator = 'rsi'
    elif 'macd' in filename.lower():
        indicator = 'macd'
    elif 'cci' in filename.lower():
        indicator = 'cci'
    else:
        indicator = 'unknown'

    logger.info(f"🎯 Indicateur détecté: {indicator.upper()}")

    # Exécuter les tests
    tests = [
        test_structure(data, indicator),
        test_alignment(data),
        test_ohlcv_coherence(data),
        test_labels_range(data),
        test_ml_extraction(data, indicator),
        test_chronological_order(data),
    ]

    # Afficher résultats
    logger.info("\n" + "="*80)
    logger.info("RÉSULTATS DES TESTS")
    logger.info("="*80)

    all_passed = True
    for test in tests:
        logger.info(f"\n{test}")
        if not test.passed:
            all_passed = False

    logger.info("\n" + "="*80)
    if all_passed:
        logger.info("✅ TOUS LES TESTS RÉUSSIS")
    else:
        logger.info("❌ CERTAINS TESTS ONT ÉCHOUÉ")
    logger.info("="*80)

    return {'passed': all_passed, 'tests': tests}


def main():
    parser = argparse.ArgumentParser(
        description="Test de validation de l'intégration des métadonnées"
    )

    parser.add_argument(
        '--quick', action='store_true',
        help='Test rapide: génère dataset avec 1000 samples et teste'
    )

    parser.add_argument(
        '--assets', '-a', type=str, nargs='+',
        default=['BTC'],
        help='Assets à utiliser pour le test rapide'
    )

    parser.add_argument(
        '--data', type=str,
        help='Chemin vers un dataset .npz existant à tester'
    )

    parser.add_argument(
        '--filter', type=str, default='kalman',
        choices=['kalman', 'octave'],
        help='Filtre à utiliser pour le test rapide'
    )

    args = parser.parse_args()

    if args.quick:
        # Mode test rapide: générer un petit dataset
        logger.info("🚀 MODE TEST RAPIDE")
        logger.info("="*80)
        logger.info(f"Assets: {', '.join(args.assets)}")
        logger.info(f"Filtre: {args.filter}")
        logger.info(f"Max samples: 1000 par asset")
        logger.info("="*80)

        # Importer et lancer la préparation
        from prepare_data_direction_only import prepare_and_save_all

        try:
            output_paths = prepare_and_save_all(
                assets=args.assets,
                filter_type=args.filter,
                max_samples=1000
            )

            logger.info("\n✅ Génération terminée")
            logger.info(f"Fichiers générés: {len(output_paths)}")

            # Tester chaque fichier généré
            all_passed = True
            for indicator, path in output_paths.items():
                logger.info(f"\n{'='*80}")
                logger.info(f"TEST: {indicator.upper()}")
                logger.info('='*80)

                result = run_all_tests(path)
                if result and not result['passed']:
                    all_passed = False

            if all_passed:
                logger.info("\n" + "="*80)
                logger.info("✅ VALIDATION COMPLÈTE RÉUSSIE")
                logger.info("="*80)
                return 0
            else:
                logger.error("\n" + "="*80)
                logger.error("❌ VALIDATION ÉCHOUÉE")
                logger.error("="*80)
                return 1

        except Exception as e:
            logger.error(f"❌ Erreur lors de la génération: {str(e)}")
            import traceback
            traceback.print_exc()
            return 1

    elif args.data:
        # Mode test sur fichier existant
        if not Path(args.data).exists():
            logger.error(f"❌ Fichier introuvable: {args.data}")
            return 1

        result = run_all_tests(args.data)
        return 0 if result and result['passed'] else 1

    else:
        logger.error("❌ Spécifier --quick ou --data <fichier.npz>")
        return 1


if __name__ == '__main__':
    sys.exit(main())
