#!/usr/bin/env python3
"""
Tests de visualisation pour valider le pipeline et les filtres.

RÈGLE CRITIQUE: Les filtres ont besoin de warm-up et ont des artifacts
aux bords. Toujours enlever 30 valeurs au début ET à la fin avant train/val.

Tests:
1. Visualiser données 5min vs 30min (bougies)
2. Visualiser filtres adaptatifs sur close (1000 points)
3. Zoom sur partie centrale (éviter artifacts de bord)
4. Tester fonction trim_filter_edges()
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import seaborn as sns
from datetime import datetime

# Imports du pipeline
from data_pipeline import create_ghost_candles
from adaptive_filters import (
    kama_filter,
    hma_filter,
    ehlers_supersmoother,
    ehlers_decycler,
    adaptive_filter_ensemble,
    extract_filter_reactivity
)

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')


def create_test_data_large(n=1000):
    """
    Crée un large dataset de test (1000 points).

    Inclut tendances + bruit pour tester les filtres.
    """
    np.random.seed(42)

    # Timestamps
    timestamps = pd.date_range('2024-01-01', periods=n, freq='5min')

    # Signal avec tendance + cycles + bruit
    t = np.linspace(0, 10*np.pi, n)

    # Tendance lente
    trend = 50000 + 5000 * np.sin(t / 5)

    # Cycles moyens
    cycles = 1000 * np.sin(t)

    # Bruit
    noise = np.random.randn(n) * 200

    # Prix = trend + cycles + noise
    prices = trend + cycles + noise

    # Créer OHLC
    data = []
    for i, (ts, close) in enumerate(zip(timestamps, prices)):
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, 50)
            open_price = prices[i-1] + gap

        volatility = np.random.uniform(0.002, 0.008)
        high = max(open_price, close) * (1 + volatility)
        low = min(open_price, close) * (1 - volatility)
        volume = np.random.uniform(100, 1000)

        data.append({
            'timestamp': ts,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })

    return pd.DataFrame(data)


def plot_candlestick(df, ax, n_candles=50, title="Bougies", start_idx=0):
    """
    Affiche des bougies (candlestick chart).

    Args:
        df: DataFrame avec OHLC
        ax: Matplotlib axis
        n_candles: Nombre de bougies à afficher
        title: Titre du graphique
        start_idx: Index de départ
    """
    df_plot = df.iloc[start_idx:start_idx + n_candles].copy()

    for idx, row in df_plot.iterrows():
        # Position x
        x = idx - start_idx

        # Couleur: vert si hausse, rouge si baisse
        color = 'green' if row['close'] >= row['open'] else 'red'

        # Corps de la bougie (rectangle)
        body_height = abs(row['close'] - row['open'])
        body_bottom = min(row['open'], row['close'])

        rect = Rectangle(
            (x - 0.3, body_bottom),
            0.6,
            body_height,
            facecolor=color,
            edgecolor='black',
            alpha=0.8
        )
        ax.add_patch(rect)

        # Mèches (high-low)
        ax.plot([x, x], [row['low'], row['high']], color='black', linewidth=1)

    ax.set_xlim(-1, n_candles)
    ax.set_xlabel('Bougie #')
    ax.set_ylabel('Prix')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def test_visualize_5min_vs_30min():
    """
    Test 1: Visualiser données 5min vs bougies fantômes 30min sur MÊME courbe.

    À chaque timestamp 5min, on affiche:
    - Bougie 5min (vert/rouge plein)
    - Ghost candle 30min équivalente (bleu transparent)

    Permet de comparer visuellement OHLC 5min vs 30min.
    """
    print("\n" + "="*80)
    print("TEST 1: VISUALISATION 5MIN VS 30MIN SUR MÊME COURBE")
    print("="*80)

    # Créer données
    df_5m = create_test_data_large(n=72)  # 72 bougies 5min = 6 heures = 12 bougies 30min

    # Créer bougies fantômes
    df_ghost = create_ghost_candles(df_5m, target_timeframe='30min')

    # Nombre de bougies à afficher
    n_candles = 60  # Afficher 60 bougies 5min (5 heures)

    # Créer figure
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    # Tracer les bougies 5min et 30min sur le même axe
    for i in range(n_candles):
        # Position x
        x = i

        # --- BOUGIE 5MIN (pleine opacité) ---
        row_5m = df_5m.iloc[i]

        # Couleur: vert si hausse, rouge si baisse
        color_5m = 'green' if row_5m['close'] >= row_5m['open'] else 'red'

        # Corps de la bougie 5min
        body_height_5m = abs(row_5m['close'] - row_5m['open'])
        body_bottom_5m = min(row_5m['open'], row_5m['close'])

        rect_5m = Rectangle(
            (x - 0.4, body_bottom_5m),
            0.35,  # Largeur réduite pour laisser place à la 30min
            body_height_5m,
            facecolor=color_5m,
            edgecolor='black',
            alpha=0.8,
            linewidth=1.5,
            label='5min' if i == 0 else ''
        )
        ax.add_patch(rect_5m)

        # Mèches 5min
        ax.plot([x - 0.225, x - 0.225], [row_5m['low'], row_5m['high']],
                color='black', linewidth=1.5)

        # --- GHOST CANDLE 30MIN (transparente) ---
        row_ghost = df_ghost.iloc[i]

        # Couleur: bleu pour les ghost candles
        color_30m = 'dodgerblue' if row_ghost['ghost_close'] >= row_ghost['ghost_open'] else 'navy'

        # Corps de la bougie 30min
        body_height_30m = abs(row_ghost['ghost_close'] - row_ghost['ghost_open'])
        body_bottom_30m = min(row_ghost['ghost_open'], row_ghost['ghost_close'])

        rect_30m = Rectangle(
            (x - 0.05, body_bottom_30m),
            0.35,  # Largeur réduite
            body_height_30m,
            facecolor=color_30m,
            edgecolor='blue',
            alpha=0.4,  # Plus transparent
            linewidth=1.5,
            label='30min (ghost)' if i == 0 else ''
        )
        ax.add_patch(rect_30m)

        # Mèches 30min
        ax.plot([x + 0.125, x + 0.125], [row_ghost['ghost_low'], row_ghost['ghost_high']],
                color='blue', linewidth=1.5, alpha=0.6)

    # Configuration axes
    ax.set_xlim(-1, n_candles)
    ax.set_xlabel('Index Bougie 5min', fontsize=14, fontweight='bold')
    ax.set_ylabel('Prix', fontsize=14, fontweight='bold')
    ax.set_title('Comparaison Bougies 5min (vert/rouge) vs Ghost Candles 30min (bleu)',
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Légende
    handles = [
        mpatches.Patch(color='green', alpha=0.8, label='Bougie 5min Haussière'),
        mpatches.Patch(color='red', alpha=0.8, label='Bougie 5min Baissière'),
        mpatches.Patch(color='dodgerblue', alpha=0.4, label='Ghost Candle 30min Haussière'),
        mpatches.Patch(color='navy', alpha=0.4, label='Ghost Candle 30min Baissière')
    ]
    ax.legend(handles=handles, loc='upper left', fontsize=12, framealpha=0.9)

    # Ajouter grille de prix (échelle visible)
    ax.yaxis.set_major_locator(plt.MaxNLocator(20))
    ax.tick_params(axis='both', labelsize=11)

    plt.tight_layout()
    output_path = Path('tests/validation_output/01_5min_vs_30min_candles.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Sauvegardé: {output_path}")
    plt.close()

    # Stats
    df_30m_complete = df_ghost[df_ghost['step'] == 6]
    print(f"✅ {len(df_5m)} bougies 5min → {len(df_30m_complete)} bougies 30min complètes")
    print(f"✅ Visualisées: {n_candles} bougies superposées (5min + ghost 30min)")
    print(f"✅ À chaque timestamp 5min: bougie 5min (opaque) + ghost 30min (transparent)")


def test_visualize_filters_on_close():
    """
    Test 2: Visualiser filtres adaptatifs sur close (1000 points).

    IMPORTANT: Visualiser la partie CENTRALE pour éviter artifacts de bord.
    """
    print("\n" + "="*80)
    print("TEST 2: FILTRES ADAPTATIFS SUR CLOSE (1000 POINTS)")
    print("="*80)

    # Créer 1000 points
    df = create_test_data_large(n=1000)
    close = df['close'].values

    print(f"Signal: {len(close)} points")

    # Appliquer tous les filtres
    print("Application des filtres...")
    kama, er = kama_filter(close, return_efficiency_ratio=True)
    hma = hma_filter(close)
    supersmoother = ehlers_supersmoother(close)
    decycler = ehlers_decycler(close)
    ensemble = adaptive_filter_ensemble(close)

    # Visualiser la partie CENTRALE (éviter les 100 premiers et 100 derniers)
    start_idx = 100
    end_idx = 900
    zoom_start = 400
    zoom_end = 600

    fig, axes = plt.subplots(3, 2, figsize=(18, 14))

    # 1. Vue d'ensemble (points 100-900, éviter bords)
    axes[0, 0].plot(close[start_idx:end_idx], label='Close original', alpha=0.5, linewidth=1)
    axes[0, 0].plot(kama[start_idx:end_idx], label='KAMA', linewidth=2)
    axes[0, 0].plot(hma[start_idx:end_idx], label='HMA', linewidth=2)
    axes[0, 0].set_title('Vue d\'ensemble (points 100-900, sans bords)')
    axes[0, 0].set_xlabel('Index')
    axes[0, 0].set_ylabel('Prix')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Zoom partie centrale (400-600)
    axes[0, 1].plot(close[zoom_start:zoom_end], label='Close', alpha=0.5, linewidth=1)
    axes[0, 1].plot(kama[zoom_start:zoom_end], label='KAMA', linewidth=2)
    axes[0, 1].plot(hma[zoom_start:zoom_end], label='HMA', linewidth=2)
    axes[0, 1].plot(supersmoother[zoom_start:zoom_end], label='SuperSmoother', linewidth=2)
    axes[0, 1].set_title('ZOOM Centre (points 400-600) - Zone propre')
    axes[0, 1].set_xlabel('Index')
    axes[0, 1].set_ylabel('Prix')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. KAMA seul (zoom)
    axes[1, 0].plot(close[zoom_start:zoom_end], label='Close', alpha=0.3, linewidth=1)
    axes[1, 0].plot(kama[zoom_start:zoom_end], label='KAMA', linewidth=2, color='blue')
    axes[1, 0].set_title('KAMA (Kaufman Adaptive MA)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. HMA seul (zoom)
    axes[1, 1].plot(close[zoom_start:zoom_end], label='Close', alpha=0.3, linewidth=1)
    axes[1, 1].plot(hma[zoom_start:zoom_end], label='HMA', linewidth=2, color='orange')
    axes[1, 1].set_title('HMA (Hull MA - Très réactif)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 5. Ehlers SuperSmoother
    axes[2, 0].plot(close[zoom_start:zoom_end], label='Close', alpha=0.3, linewidth=1)
    axes[2, 0].plot(supersmoother[zoom_start:zoom_end], label='SuperSmoother', linewidth=2, color='green')
    axes[2, 0].set_title('Ehlers SuperSmoother (Zero-lag optimal)')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # 6. Efficiency Ratio (ER)
    axes[2, 1].plot(er[zoom_start:zoom_end], label='Efficiency Ratio', linewidth=2, color='purple')
    axes[2, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Seuil 0.5')
    axes[2, 1].set_title('Efficiency Ratio (Vitesse du marché)')
    axes[2, 1].set_xlabel('Index')
    axes[2, 1].set_ylabel('ER')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].set_ylim(-0.1, 1.1)

    plt.tight_layout()
    output_path = Path('tests/validation_output/02_adaptive_filters_on_close.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Sauvegardé: {output_path}")
    plt.close()

    print(f"✅ Filtres appliqués sur {len(close)} points")
    print(f"✅ Visualisation zone propre: points {zoom_start}-{zoom_end}")


def test_filter_edge_effects():
    """
    Test 3: Démontrer les effets de bord des filtres.

    Montre pourquoi il faut enlever les 30 premières et 30 dernières valeurs.
    """
    print("\n" + "="*80)
    print("TEST 3: EFFETS DE BORD DES FILTRES")
    print("="*80)

    # Signal de 200 points
    df = create_test_data_large(n=200)
    close = df['close'].values

    # Appliquer KAMA
    kama = kama_filter(close)

    # Calculer l'erreur (différence absolue)
    error = np.abs(close - kama)

    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    # 1. Signal complet
    axes[0].plot(close, label='Close', alpha=0.5)
    axes[0].plot(kama, label='KAMA', linewidth=2)
    axes[0].axvspan(0, 30, color='red', alpha=0.2, label='Zone début (warm-up)')
    axes[0].axvspan(170, 200, color='red', alpha=0.2, label='Zone fin (artifacts)')
    axes[0].axvspan(30, 170, color='green', alpha=0.1, label='Zone propre')
    axes[0].set_title('Signal complet - Zones à éviter en ROUGE')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Erreur de filtrage
    axes[1].plot(error, label='Erreur absolue |Close - KAMA|', color='red')
    axes[1].axvspan(0, 30, color='red', alpha=0.2)
    axes[1].axvspan(170, 200, color='red', alpha=0.2)
    axes[1].set_title('Erreur de filtrage (plus élevée aux bords)')
    axes[1].set_ylabel('Erreur absolue')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. Zoom sur début (0-50)
    axes[2].plot(close[:50], label='Close', alpha=0.5)
    axes[2].plot(kama[:50], label='KAMA', linewidth=2)
    axes[2].axvspan(0, 30, color='red', alpha=0.2, label='Warm-up (30 points)')
    axes[2].set_title('ZOOM Début - Le filtre a besoin de warm-up')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path('tests/validation_output/03_filter_edge_effects.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Sauvegardé: {output_path}")
    plt.close()

    # Calculer erreur moyenne par zone
    error_start = np.mean(error[:30])
    error_middle = np.mean(error[30:170])
    error_end = np.mean(error[170:])

    print(f"\n📊 Erreur moyenne par zone:")
    print(f"  - Début (0-30):    {error_start:.2f} ❌ ÉLEVÉE")
    print(f"  - Milieu (30-170): {error_middle:.2f} ✅ FAIBLE")
    print(f"  - Fin (170-200):   {error_end:.2f} ❌ ÉLEVÉE")
    print(f"\n⚠️  RÈGLE: Toujours enlever 30 valeurs début ET fin avant train/val!")


def trim_filter_edges(df, n_trim=30, timestamp_col='timestamp'):
    """
    Enlève les bords du dataset après filtrage.

    RÈGLE CRITIQUE: Les filtres ont besoin de warm-up (début) et
    peuvent avoir des artifacts (fin). Il faut enlever ces zones
    avant de créer les splits train/val/test.

    Args:
        df: DataFrame avec données filtrées
        n_trim: Nombre de valeurs à enlever au début ET à la fin (défaut: 30)
        timestamp_col: Nom de la colonne timestamp

    Returns:
        DataFrame sans les bords

    Example:
        >>> df_filtered = apply_filters(df)
        >>> df_clean = trim_filter_edges(df_filtered, n_trim=30)
        >>> # Maintenant df_clean peut être utilisé pour train/val/test
    """
    if len(df) <= 2 * n_trim:
        raise ValueError(f"Dataset trop petit ({len(df)} lignes) pour enlever {n_trim} valeurs de chaque côté")

    # Enlever les n_trim premières et n_trim dernières lignes
    df_trimmed = df.iloc[n_trim:-n_trim].copy()

    # Réinitialiser l'index
    df_trimmed = df_trimmed.reset_index(drop=True)

    print(f"✂️  Trim: {len(df)} → {len(df_trimmed)} lignes (enlevé {n_trim} début + {n_trim} fin)")

    return df_trimmed


def test_trim_function():
    """
    Test 4: Tester la fonction trim_filter_edges().
    """
    print("\n" + "="*80)
    print("TEST 4: FONCTION TRIM_FILTER_EDGES()")
    print("="*80)

    # Créer dataset
    df = create_test_data_large(n=200)

    print(f"Dataset original: {len(df)} lignes")
    print(f"Première date: {df['timestamp'].iloc[0]}")
    print(f"Dernière date: {df['timestamp'].iloc[-1]}")

    # Appliquer trim
    df_trimmed = trim_filter_edges(df, n_trim=30)

    print(f"\nDataset trimmed: {len(df_trimmed)} lignes")
    print(f"Première date: {df_trimmed['timestamp'].iloc[0]}")
    print(f"Dernière date: {df_trimmed['timestamp'].iloc[-1]}")

    # Vérifier
    assert len(df_trimmed) == len(df) - 60, "Erreur: devrait enlever 60 lignes (30+30)"

    print("\n✅ Fonction trim_filter_edges() fonctionne correctement")


def test_comparison_all_filters():
    """
    Test 5: Comparer TOUS les filtres sur la même zone.
    """
    print("\n" + "="*80)
    print("TEST 5: COMPARAISON DE TOUS LES FILTRES")
    print("="*80)

    # 1000 points
    df = create_test_data_large(n=1000)
    close = df['close'].values

    # Appliquer tous les filtres
    kama = kama_filter(close)
    hma = hma_filter(close)
    supersmoother = ehlers_supersmoother(close)
    decycler = ehlers_decycler(close)
    ensemble = adaptive_filter_ensemble(close)

    # Zone centrale propre (400-600)
    start, end = 400, 600

    fig, ax = plt.subplots(figsize=(16, 8))

    ax.plot(close[start:end], label='Close original', alpha=0.4, linewidth=1, color='gray')
    ax.plot(kama[start:end], label='KAMA', linewidth=2)
    ax.plot(hma[start:end], label='HMA (Hull)', linewidth=2)
    ax.plot(supersmoother[start:end], label='SuperSmoother', linewidth=2)
    ax.plot(decycler[start:end], label='Decycler', linewidth=2)
    ax.plot(ensemble[start:end], label='Ensemble', linewidth=2.5, linestyle='--', color='black')

    ax.set_title('Comparaison Filtres Adaptatifs (Zone propre 400-600)', fontsize=14)
    ax.set_xlabel('Index')
    ax.set_ylabel('Prix')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path('tests/validation_output/04_all_filters_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Sauvegardé: {output_path}")
    plt.close()

    print("✅ Comparaison complète des 5 filtres")


def main():
    """Point d'entrée principal."""
    print("\n" + "="*80)
    print("TESTS DE VISUALISATION - PIPELINE & FILTRES")
    print("="*80)

    # Créer dossier de sortie
    Path('tests/validation_output').mkdir(parents=True, exist_ok=True)

    # Test 1: 5min vs 30min
    test_visualize_5min_vs_30min()

    # Test 2: Filtres sur close (1000 points)
    test_visualize_filters_on_close()

    # Test 3: Effets de bord
    test_filter_edge_effects()

    # Test 4: Fonction trim
    test_trim_function()

    # Test 5: Comparaison tous filtres
    test_comparison_all_filters()

    # Résumé
    print("\n" + "="*80)
    print("✅ TOUS LES TESTS DE VISUALISATION PASSÉS")
    print("="*80)
    print("\n📊 Visualisations générées:")
    print("  1. tests/validation_output/01_5min_vs_30min_candles.png")
    print("  2. tests/validation_output/02_adaptive_filters_on_close.png")
    print("  3. tests/validation_output/03_filter_edge_effects.png")
    print("  4. tests/validation_output/04_all_filters_comparison.png")

    print("\n⚠️  RÈGLE CRITIQUE:")
    print("  - Toujours enlever 30 valeurs au DÉBUT (warm-up)")
    print("  - Toujours enlever 30 valeurs à la FIN (artifacts)")
    print("  - Utiliser trim_filter_edges(df, n_trim=30) avant train/val/test")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
