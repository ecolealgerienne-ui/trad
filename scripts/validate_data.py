#!/usr/bin/env python3
"""
Validation et correction de la qualité des données OHLCV.

Vérifications:
    1. Gaps temporels (timestamps manquants)
    2. Valeurs NaN
    3. Doublons
    4. Cohérence OHLC (High >= Low, High >= Open/Close, etc.)
    5. Continuité des prix (Open ≈ Previous Close)
    6. Outliers / sauts de prix anormaux
    7. Volume (zéro, négatif, anomalies)

Usage:
    # Vérifier tous les fichiers
    python scripts/validate_data.py

    # Vérifier un fichier spécifique
    python scripts/validate_data.py --file ../data_trad/BTCUSD_all_5m.csv

    # Corriger les problèmes automatiquement
    python scripts/validate_data.py --fix

    # Rapport détaillé
    python scripts/validate_data.py --verbose
"""

import argparse
import logging
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = Path('../data_trad')
INTERVALS = {
    '1m': timedelta(minutes=1),
    '5m': timedelta(minutes=5),
    '15m': timedelta(minutes=15),
    '30m': timedelta(minutes=30),
    '1h': timedelta(hours=1),
    '4h': timedelta(hours=4),
    '1d': timedelta(days=1),
}

# Seuils pour la détection d'anomalies
MAX_PRICE_CHANGE_PCT = 20.0  # Changement max en % entre 2 bougies
MAX_GAP_RATIO = 10  # Nombre max de bougies manquantes consécutives à combler


class DataQualityReport:
    """Rapport de qualité des données."""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.rows_original = 0
        self.rows_final = 0
        self.issues = []
        self.fixes = []
        self.warnings = []

    def add_issue(self, category: str, count: int, details: str = ""):
        if count > 0:
            self.issues.append({
                'category': category,
                'count': count,
                'details': details
            })

    def add_fix(self, category: str, count: int, details: str = ""):
        if count > 0:
            self.fixes.append({
                'category': category,
                'count': count,
                'details': details
            })

    def add_warning(self, message: str):
        self.warnings.append(message)

    def is_valid(self) -> bool:
        return len(self.issues) == 0

    def print_report(self, verbose: bool = False):
        print(f"\n{'=' * 70}")
        print(f"📊 RAPPORT: {self.filepath.name}")
        print(f"{'=' * 70}")
        print(f"Lignes originales: {self.rows_original:,}")
        print(f"Lignes finales:    {self.rows_final:,}")

        if self.issues:
            print(f"\n❌ PROBLÈMES DÉTECTÉS ({len(self.issues)}):")
            for issue in self.issues:
                print(f"   • {issue['category']}: {issue['count']:,}")
                if verbose and issue['details']:
                    print(f"     {issue['details']}")

        if self.fixes:
            print(f"\n✅ CORRECTIONS APPLIQUÉES ({len(self.fixes)}):")
            for fix in self.fixes:
                print(f"   • {fix['category']}: {fix['count']:,}")
                if verbose and fix['details']:
                    print(f"     {fix['details']}")

        if self.warnings:
            print(f"\n⚠️  AVERTISSEMENTS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   • {warning}")

        if self.is_valid() and not self.fixes:
            print(f"\n✅ Données valides - Aucun problème détecté")

        print(f"{'=' * 70}")


def detect_interval(df: pd.DataFrame) -> timedelta:
    """Détecte l'intervalle des données."""
    if len(df) < 2:
        return timedelta(minutes=5)

    # Calculer les différences entre timestamps
    diffs = df['timestamp'].diff().dropna()
    most_common = diffs.mode().iloc[0]

    return most_common


def check_gaps(df: pd.DataFrame, interval: timedelta) -> Tuple[List, int]:
    """Détecte les gaps temporels."""
    gaps = []
    expected_diff = interval

    for i in range(1, len(df)):
        actual_diff = df['timestamp'].iloc[i] - df['timestamp'].iloc[i - 1]

        if actual_diff > expected_diff:
            missing_count = int(actual_diff / expected_diff) - 1
            gaps.append({
                'index': i,
                'start': df['timestamp'].iloc[i - 1],
                'end': df['timestamp'].iloc[i],
                'missing': missing_count
            })

    total_missing = sum(g['missing'] for g in gaps)
    return gaps, total_missing


def check_duplicates(df: pd.DataFrame) -> int:
    """Compte les doublons."""
    return df.duplicated(subset='timestamp').sum()


def check_nan_values(df: pd.DataFrame) -> Dict[str, int]:
    """Compte les valeurs NaN par colonne."""
    nan_counts = {}
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            count = df[col].isna().sum()
            if count > 0:
                nan_counts[col] = count
    return nan_counts


def check_ohlc_consistency(df: pd.DataFrame) -> Dict[str, int]:
    """Vérifie la cohérence OHLC."""
    issues = {}

    # High doit être >= Open, Close, Low
    high_low = (df['high'] < df['low']).sum()
    if high_low > 0:
        issues['high < low'] = high_low

    high_open = (df['high'] < df['open']).sum()
    if high_open > 0:
        issues['high < open'] = high_open

    high_close = (df['high'] < df['close']).sum()
    if high_close > 0:
        issues['high < close'] = high_close

    # Low doit être <= Open, Close
    low_open = (df['low'] > df['open']).sum()
    if low_open > 0:
        issues['low > open'] = low_open

    low_close = (df['low'] > df['close']).sum()
    if low_close > 0:
        issues['low > close'] = low_close

    return issues


def check_price_continuity(df: pd.DataFrame) -> Tuple[int, List]:
    """Vérifie la continuité des prix (Open ≈ Previous Close)."""
    if len(df) < 2:
        return 0, []

    # Calculer le gap entre Open et Previous Close
    df_check = df.copy()
    df_check['prev_close'] = df_check['close'].shift(1)
    df_check['gap_pct'] = abs(df_check['open'] - df_check['prev_close']) / df_check['prev_close'] * 100

    # Ignorer la première ligne
    df_check = df_check.iloc[1:]

    # Détecter les gaps significatifs (> 1%)
    large_gaps = df_check[df_check['gap_pct'] > 1.0]

    return len(large_gaps), large_gaps.index.tolist()


def check_price_outliers(df: pd.DataFrame) -> Tuple[int, List]:
    """Détecte les sauts de prix anormaux."""
    if len(df) < 2:
        return 0, []

    # Calculer le changement en %
    df_check = df.copy()
    df_check['change_pct'] = df_check['close'].pct_change().abs() * 100

    # Détecter les changements > seuil
    outliers = df_check[df_check['change_pct'] > MAX_PRICE_CHANGE_PCT]

    return len(outliers), outliers.index.tolist()


def check_volume(df: pd.DataFrame) -> Dict[str, int]:
    """Vérifie les anomalies de volume."""
    issues = {}

    # Volume zéro
    zero_vol = (df['volume'] == 0).sum()
    if zero_vol > 0:
        issues['volume = 0'] = zero_vol

    # Volume négatif
    neg_vol = (df['volume'] < 0).sum()
    if neg_vol > 0:
        issues['volume < 0'] = neg_vol

    return issues


def fill_gaps(df: pd.DataFrame, interval: timedelta, max_gap: int = MAX_GAP_RATIO) -> Tuple[pd.DataFrame, int]:
    """Comble les gaps temporels par forward-fill."""
    gaps, _ = check_gaps(df, interval)

    if not gaps:
        return df, 0

    filled_count = 0
    new_rows = []

    for gap in gaps:
        if gap['missing'] <= max_gap:
            # Créer les timestamps manquants
            start = gap['start']
            for i in range(1, gap['missing'] + 1):
                new_ts = start + (interval * i)
                new_rows.append({'timestamp': new_ts})
                filled_count += 1

    if new_rows:
        # Ajouter les nouvelles lignes
        df_new = pd.DataFrame(new_rows)
        df = pd.concat([df, df_new], ignore_index=True)
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Forward-fill les valeurs OHLCV
        df = df.ffill()

    return df, filled_count


def fix_nan_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Corrige les valeurs NaN par interpolation/forward-fill."""
    nan_before = df.isna().sum().sum()

    # Interpolation pour OHLC
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear')
            df[col] = df[col].ffill().bfill()

    # Forward-fill pour volume
    if 'volume' in df.columns:
        df['volume'] = df['volume'].ffill().bfill()
        df['volume'] = df['volume'].fillna(0)

    nan_after = df.isna().sum().sum()
    fixed = nan_before - nan_after

    return df, fixed


def fix_ohlc_consistency(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Corrige les incohérences OHLC."""
    fixed = 0

    # High doit être le max
    mask = df['high'] < df[['open', 'close']].max(axis=1)
    if mask.any():
        df.loc[mask, 'high'] = df.loc[mask, ['open', 'close', 'high']].max(axis=1)
        fixed += mask.sum()

    # Low doit être le min
    mask = df['low'] > df[['open', 'close']].min(axis=1)
    if mask.any():
        df.loc[mask, 'low'] = df.loc[mask, ['open', 'close', 'low']].min(axis=1)
        fixed += mask.sum()

    # High >= Low
    mask = df['high'] < df['low']
    if mask.any():
        # Inverser
        df.loc[mask, ['high', 'low']] = df.loc[mask, ['low', 'high']].values
        fixed += mask.sum()

    return df, fixed


def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Supprime les doublons."""
    before = len(df)
    df = df.drop_duplicates(subset='timestamp', keep='first')
    removed = before - len(df)
    return df, removed


def validate_file(filepath: Path, fix: bool = False, verbose: bool = False) -> DataQualityReport:
    """Valide un fichier de données."""
    report = DataQualityReport(filepath)

    # Charger les données
    try:
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
    except Exception as e:
        report.add_issue('Erreur chargement', 1, str(e))
        return report

    report.rows_original = len(df)

    # Détecter l'intervalle
    interval = detect_interval(df)
    interval_name = None
    for name, delta in INTERVALS.items():
        if delta == interval:
            interval_name = name
            break

    if interval_name:
        logger.info(f"  Intervalle détecté: {interval_name}")
    else:
        report.add_warning(f"Intervalle non standard: {interval}")

    # === VÉRIFICATIONS ===

    # 1. Doublons
    dup_count = check_duplicates(df)
    report.add_issue('Doublons', dup_count)

    if fix and dup_count > 0:
        df, removed = remove_duplicates(df)
        report.add_fix('Doublons supprimés', removed)

    # 2. Gaps temporels
    gaps, missing = check_gaps(df, interval)
    report.add_issue('Timestamps manquants', missing,
                     f"{len(gaps)} gaps détectés" if gaps else "")

    if fix and missing > 0:
        df, filled = fill_gaps(df, interval)
        report.add_fix('Timestamps comblés (forward-fill)', filled)

    # 3. Valeurs NaN
    nan_counts = check_nan_values(df)
    total_nan = sum(nan_counts.values())
    report.add_issue('Valeurs NaN', total_nan,
                     str(nan_counts) if nan_counts else "")

    if fix and total_nan > 0:
        df, fixed_nan = fix_nan_values(df)
        report.add_fix('NaN corrigés (interpolation)', fixed_nan)

    # 4. Cohérence OHLC
    ohlc_issues = check_ohlc_consistency(df)
    total_ohlc = sum(ohlc_issues.values())
    report.add_issue('Incohérences OHLC', total_ohlc,
                     str(ohlc_issues) if ohlc_issues else "")

    if fix and total_ohlc > 0:
        df, fixed_ohlc = fix_ohlc_consistency(df)
        report.add_fix('OHLC corrigés', fixed_ohlc)

    # 5. Continuité des prix
    gap_count, _ = check_price_continuity(df)
    if gap_count > len(df) * 0.01:  # Plus de 1% de gaps
        report.add_warning(f"Gaps prix Open/Close: {gap_count} ({gap_count/len(df)*100:.1f}%)")

    # 6. Outliers
    outlier_count, outlier_idx = check_price_outliers(df)
    if outlier_count > 0:
        report.add_warning(f"Sauts de prix > {MAX_PRICE_CHANGE_PCT}%: {outlier_count}")

    # 7. Volume
    vol_issues = check_volume(df)
    for issue, count in vol_issues.items():
        report.add_warning(f"{issue}: {count}")

    report.rows_final = len(df)

    # Sauvegarder si corrections
    if fix and (report.fixes or report.rows_final != report.rows_original):
        # Backup
        backup_path = filepath.with_suffix('.csv.bak')
        if not backup_path.exists():
            import shutil
            shutil.copy(filepath, backup_path)
            logger.info(f"  Backup créé: {backup_path}")

        # Sauvegarder
        df.to_csv(filepath, index=False)
        logger.info(f"  Fichier mis à jour: {filepath}")

    return report


def validate_all(data_dir: Path, fix: bool = False, verbose: bool = False) -> List[DataQualityReport]:
    """Valide tous les fichiers CSV dans un dossier."""
    reports = []

    csv_files = sorted(data_dir.glob("*.csv"))

    if not csv_files:
        logger.warning(f"Aucun fichier CSV trouvé dans {data_dir}")
        return reports

    print(f"\n{'=' * 70}")
    print(f"VALIDATION DES DONNÉES - {len(csv_files)} fichiers")
    print(f"{'=' * 70}")

    for filepath in csv_files:
        if filepath.suffix == '.bak':
            continue

        logger.info(f"\n📂 {filepath.name}")
        report = validate_file(filepath, fix=fix, verbose=verbose)
        reports.append(report)
        report.print_report(verbose=verbose)

    # Résumé
    print(f"\n{'=' * 70}")
    print("RÉSUMÉ")
    print(f"{'=' * 70}")

    valid = sum(1 for r in reports if r.is_valid())
    with_issues = len(reports) - valid

    print(f"Fichiers valides:      {valid}/{len(reports)}")
    print(f"Fichiers avec erreurs: {with_issues}/{len(reports)}")

    if fix:
        fixed = sum(1 for r in reports if r.fixes)
        print(f"Fichiers corrigés:     {fixed}/{len(reports)}")

    return reports


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validation et correction de la qualité des données OHLCV'
    )

    parser.add_argument(
        '--file', '-f',
        type=Path,
        help='Fichier spécifique à valider'
    )

    parser.add_argument(
        '--dir', '-d',
        type=Path,
        default=DATA_DIR,
        help=f'Dossier contenant les CSV (défaut: {DATA_DIR})'
    )

    parser.add_argument(
        '--fix',
        action='store_true',
        help='Corriger automatiquement les problèmes'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Afficher les détails'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Résoudre chemin
    data_dir = args.dir
    if not data_dir.is_absolute():
        data_dir = Path(__file__).parent.parent / args.dir.name

    if args.file:
        filepath = args.file
        if not filepath.is_absolute():
            filepath = Path(__file__).parent.parent / args.file

        if not filepath.exists():
            logger.error(f"Fichier non trouvé: {filepath}")
            return

        report = validate_file(filepath, fix=args.fix, verbose=args.verbose)
        report.print_report(verbose=args.verbose)

    else:
        validate_all(data_dir, fix=args.fix, verbose=args.verbose)


if __name__ == '__main__':
    main()
