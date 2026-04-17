#!/usr/bin/env python3
"""
Compare Standard vs AQ-KF — même signal ou complémentaires ?
=============================================================

Lit le CSV FLKS features et compare :
1. Corrélation des pentes (Std vs AQ)
2. Accord de signe (% du temps où les 2 disent pareil)
3. Quand ils désaccordent, qui a raison ? (vs oracle)
4. Matrice de confusion croisée

Usage:
    python src/signal_processing/compare_std_vs_aq.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    csv_path = 'data/prepared/BTCUSD_flks_features.csv'
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path, parse_dates=['datetime']).set_index('datetime')
    print(f"  {len(df):,} rows")

    # Work at 30min closures (where slopes are computed, not forward-filled)
    # Detect closures: rows where slope values change
    df_30m = df.dropna(subset=['std_t1_slope', 'aq_t1_slope', 'oracle_slope_macd_30m'])

    # Deduplicate to 30min by taking last value per 30min bucket
    df_30m = df_30m.resample('30min').last().dropna(subset=['std_t1_slope'])
    n = len(df_30m)

    TRIM = 100
    s = TRIM
    e = n - TRIM
    n_eval = e - s

    print(f"  30min closures: {n:,} | Eval [{s}:{e}] = {n_eval:,}")

    methods = [('t1', 'T1 (0 pas)')]
    for k in range(1, 7):
        methods.append((f'k{k}', f'k={k} ({k*5}min)'))

    oracle = df_30m['oracle_slope_macd_30m'].values

    print(f"\n{'=' * 80}")
    print(f"  COMPARAISON STANDARD vs AQ-KF")
    print(f"{'=' * 80}")

    # --- 1. Corrélation des pentes ---
    print(f"\n  1. CORRÉLATION DES PENTES (Pearson)")
    print(f"  {'Méthode':<20} {'Corr Std↔AQ':>12} {'Corr Std↔Orc':>13} {'Corr AQ↔Orc':>12}")
    print(f"  {'-' * 59}")

    for key, label in methods:
        std_sl = df_30m[f'std_{key}_slope'].values[s:e]
        aq_sl = df_30m[f'aq_{key}_slope'].values[s:e]
        orc_sl = oracle[s:e]

        mask = ~np.isnan(std_sl) & ~np.isnan(aq_sl) & ~np.isnan(orc_sl)
        if mask.sum() < 10:
            continue

        corr_sa = np.corrcoef(std_sl[mask], aq_sl[mask])[0, 1]
        corr_so = np.corrcoef(std_sl[mask], orc_sl[mask])[0, 1]
        corr_ao = np.corrcoef(aq_sl[mask], orc_sl[mask])[0, 1]

        print(f"  {label:<20} {corr_sa:>11.4f} {corr_so:>12.4f} {corr_ao:>11.4f}")

    # --- 2. Accord de signe ---
    print(f"\n  2. ACCORD DE SIGNE (Std vs AQ)")
    print(f"  {'Méthode':<20} {'Accord':>8} {'Désaccord':>10} {'Std raison':>11} {'AQ raison':>10}")
    print(f"  {'-' * 61}")

    for key, label in methods:
        std_sl = df_30m[f'std_{key}_slope'].values[s:e]
        aq_sl = df_30m[f'aq_{key}_slope'].values[s:e]
        orc_sl = oracle[s:e]

        mask = ~np.isnan(std_sl) & ~np.isnan(aq_sl) & ~np.isnan(orc_sl)
        if mask.sum() < 10:
            continue

        std_sign = np.sign(std_sl[mask])
        aq_sign = np.sign(aq_sl[mask])
        orc_sign = np.sign(orc_sl[mask])

        accord = np.mean(std_sign == aq_sign) * 100
        desaccord = 100 - accord

        # When they disagree, who is right?
        disagree = std_sign != aq_sign
        n_disagree = disagree.sum()
        if n_disagree > 0:
            std_right = np.mean(std_sign[disagree] == orc_sign[disagree]) * 100
            aq_right = np.mean(aq_sign[disagree] == orc_sign[disagree]) * 100
        else:
            std_right = aq_right = 0

        print(f"  {label:<20} {accord:>7.1f}% {desaccord:>9.1f}% {std_right:>10.1f}% {aq_right:>9.1f}%")

    # --- 3. Analyse aux transitions oracle ---
    print(f"\n  3. AUX TRANSITIONS ORACLE — qui détecte quoi ?")

    orc_labels = oracle[s:e]
    transitions = np.zeros(n_eval, dtype=bool)
    for i in range(1, n_eval):
        if not np.isnan(orc_labels[i]) and not np.isnan(orc_labels[i-1]):
            if np.sign(orc_labels[i]) != np.sign(orc_labels[i-1]) and np.sign(orc_labels[i]) != 0 and np.sign(orc_labels[i-1]) != 0:
                transitions[i] = True
    n_trans = transitions.sum()
    print(f"  Transitions: {n_trans:,}")

    print(f"\n  {'Méthode':<20} {'Std ok':>7} {'AQ ok':>7} {'Les 2 ok':>9} {'Aucun ok':>9} {'AQ seul':>8} {'Std seul':>9}")
    print(f"  {'-' * 71}")

    for key, label in methods:
        std_sl = df_30m[f'std_{key}_slope'].values[s:e]
        aq_sl = df_30m[f'aq_{key}_slope'].values[s:e]

        std_ok = np.sign(std_sl[transitions]) == np.sign(orc_labels[transitions])
        aq_ok = np.sign(aq_sl[transitions]) == np.sign(orc_labels[transitions])

        both_ok = (std_ok & aq_ok).sum()
        none_ok = (~std_ok & ~aq_ok).sum()
        aq_only = (~std_ok & aq_ok).sum()
        std_only = (std_ok & ~aq_ok).sum()

        n_t = transitions.sum()
        print(f"  {label:<20} {std_ok.sum():>6} {aq_ok.sum():>6} {both_ok:>8} "
              f"{none_ok:>8} {aq_only:>7} {std_only:>8}")

    print(f"\n  {'-' * 71}")

    # --- 4. Résumé ---
    std_t1 = df_30m['std_t1_slope'].values[s:e]
    aq_t1 = df_30m['aq_t1_slope'].values[s:e]
    mask = ~np.isnan(std_t1) & ~np.isnan(aq_t1)
    corr = np.corrcoef(std_t1[mask], aq_t1[mask])[0, 1]

    print(f"\n  RÉSUMÉ:")
    if corr > 0.95:
        print(f"  Corrélation T1 = {corr:.4f} → TRÈS CORRÉLÉS (même signal)")
        print(f"  → Utiliser les 2 en features n'apporte probablement PAS d'info")
    elif corr > 0.80:
        print(f"  Corrélation T1 = {corr:.4f} → CORRÉLÉS mais avec différences")
        print(f"  → Les 2 en features peut apporter un léger gain")
    else:
        print(f"  Corrélation T1 = {corr:.4f} → COMPLÉMENTAIRES")
        print(f"  → Les 2 en features devrait apporter de l'info supplémentaire")

    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
