"""
Test de l'implémentation de l'hysteresis dans state_machine_v2.

Ce script génère des données synthétiques pour tester que l'hysteresis
réduit bien le nombre de trades en évitant les flips constants.
"""

import numpy as np
import sys
from pathlib import Path

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from state_machine_v2 import run_state_machine_v2


def generate_synthetic_data(n_samples=1000, noise_level=0.1):
    """
    Génère des données synthétiques pour tester l'hysteresis.

    Le signal MACD oscille autour de 0.5 avec du bruit.
    Cela devrait générer beaucoup de trades sans hysteresis,
    mais beaucoup moins avec hysteresis.
    """
    # Signal de base: sinusoïde lente autour de 0.5
    t = np.linspace(0, 4 * np.pi, n_samples)
    base_signal = 0.5 + 0.3 * np.sin(t)

    # Ajouter du bruit
    noise = np.random.randn(n_samples) * noise_level
    macd_prob = np.clip(base_signal + noise, 0, 1)

    # Volatilité constante élevée (pour toujours passer le gate)
    volatility = np.ones(n_samples) * 0.002  # 0.2% (> threshold)

    # Rendements aléatoires (pour le calcul PnL)
    returns = np.random.randn(n_samples) * 0.001

    return macd_prob, volatility, returns


def test_hysteresis():
    """
    Compare les résultats avec et sans hysteresis.
    """
    print("="*80)
    print("TEST HYSTERESIS - Données Synthétiques")
    print("="*80)

    # Générer données
    n_samples = 1000
    macd_prob, volatility, returns = generate_synthetic_data(n_samples)

    print(f"\n📊 Données générées:")
    print(f"   Samples: {n_samples}")
    print(f"   MACD prob: mean={macd_prob.mean():.3f}, std={macd_prob.std():.3f}")
    print(f"   Oscillations autour de 0.5 avec bruit")

    # Test 1: Sans hysteresis (baseline)
    print("\n" + "="*80)
    print("TEST 1: SANS HYSTERESIS (baseline)")
    print("="*80)

    _, stats_baseline = run_state_machine_v2(
        macd_prob=macd_prob,
        volatility=volatility,
        returns=returns,
        vol_threshold=0.0013,
        direction_threshold=0.5,
        fees=0.001,  # 0.1%
        verbose=True
    )

    # Test 2: Hysteresis léger (0.45 - 0.55)
    print("\n" + "="*80)
    print("TEST 2: HYSTERESIS LÉGER (0.45 - 0.55)")
    print("="*80)

    _, stats_light = run_state_machine_v2(
        macd_prob=macd_prob,
        volatility=volatility,
        returns=returns,
        vol_threshold=0.0013,
        direction_threshold=0.5,
        fees=0.001,
        hysteresis_high=0.55,
        hysteresis_low=0.45,
        verbose=True
    )

    # Test 3: Hysteresis standard (0.4 - 0.6)
    print("\n" + "="*80)
    print("TEST 3: HYSTERESIS STANDARD (0.4 - 0.6)")
    print("="*80)

    _, stats_standard = run_state_machine_v2(
        macd_prob=macd_prob,
        volatility=volatility,
        returns=returns,
        vol_threshold=0.0013,
        direction_threshold=0.5,
        fees=0.001,
        hysteresis_high=0.6,
        hysteresis_low=0.4,
        verbose=True
    )

    # Test 4: Hysteresis fort (0.35 - 0.65)
    print("\n" + "="*80)
    print("TEST 4: HYSTERESIS FORT (0.35 - 0.65)")
    print("="*80)

    _, stats_strong = run_state_machine_v2(
        macd_prob=macd_prob,
        volatility=volatility,
        returns=returns,
        vol_threshold=0.0013,
        direction_threshold=0.5,
        fees=0.001,
        hysteresis_high=0.65,
        hysteresis_low=0.35,
        verbose=True
    )

    # Comparaison
    print("\n" + "="*80)
    print("COMPARAISON DES RÉSULTATS")
    print("="*80)

    print(f"\n{'Configuration':<20} {'Trades':>10} {'Réduction':>12} {'PnL Net':>12}")
    print("-" * 60)

    baseline_trades = stats_baseline['n_trades']
    baseline_pnl = stats_baseline['total_pnl_after_fees']

    configs = [
        ('Baseline (0.5)', stats_baseline),
        ('Léger (0.45-0.55)', stats_light),
        ('Standard (0.4-0.6)', stats_standard),
        ('Fort (0.35-0.65)', stats_strong),
    ]

    for name, stats in configs:
        trades = stats['n_trades']
        pnl = stats['total_pnl_after_fees']
        reduction = (1 - trades / baseline_trades) * 100 if baseline_trades > 0 else 0

        print(f"{name:<20} {trades:>10,} {reduction:>11.1f}% {pnl*100:>11.2f}%")

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    print("""
L'hysteresis devrait:
✅ Réduire significativement le nombre de trades
✅ Améliorer le PnL net (moins de frais)
✅ Éviter les oscillations inutiles autour de 0.5

Si ces résultats sont observés, l'implémentation est correcte!
""")


if __name__ == '__main__':
    test_hysteresis()
