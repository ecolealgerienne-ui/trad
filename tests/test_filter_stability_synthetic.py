"""
Test du script de stabilité avec données synthétiques.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from test_filter_stability import test_filter_stability


def generate_synthetic_price(n_samples=5000):
    """Génère un prix synthétique avec tendance + bruit."""
    t = np.linspace(0, 100, n_samples)

    # Tendance + cycles + bruit
    trend = 100 + 0.5 * t
    cycle1 = 10 * np.sin(2 * np.pi * t / 500)
    cycle2 = 5 * np.sin(2 * np.pi * t / 100)
    noise = np.random.randn(n_samples) * 2

    price = trend + cycle1 + cycle2 + noise

    return price


def main():
    print("="*80)
    print("TEST DE STABILITÉ - DONNÉES SYNTHÉTIQUES")
    print("="*80)

    # Générer données
    print("\n📊 Génération de données synthétiques...")
    n_samples = 5000
    close = generate_synthetic_price(n_samples)

    print(f"   Samples: {n_samples:,}")
    print(f"   Close: min={close.min():.2f}, max={close.max():.2f}, "
          f"mean={close.mean():.2f}")

    # Test avec Octave20
    print(f"\n{'='*80}")
    print("TEST 1: OCTAVE20 (step=0.20)")
    print(f"{'='*80}")

    results_octave20 = test_filter_stability(
        data=close,
        filter_name='octave',
        window_size=100,
        n_samples=200,
        step=0.20,
        order=3
    )

    # Test avec Octave25 pour comparaison
    print(f"\n{'='*80}")
    print("TEST 2: OCTAVE25 (step=0.25)")
    print(f"{'='*80}")

    results_octave25 = test_filter_stability(
        data=close,
        filter_name='octave',
        window_size=100,
        n_samples=200,
        step=0.25,
        order=3
    )

    # Comparaison
    print(f"\n{'='*80}")
    print("COMPARAISON")
    print(f"{'='*80}")

    print(f"\nConcordance:")
    print(f"   Octave20 (step=0.20): {results_octave20['concordance']:.2f}%")
    print(f"   Octave25 (step=0.25): {results_octave25['concordance']:.2f}%")

    print("\nInterprétation:")
    print("- Concordance ~100% = filtre parfaitement stable (même label en fenêtre qu'en global)")
    print("- Concordance ~95%  = très stable (quelques différences aux bords)")
    print("- Concordance <90%  = instable (sensible à la taille de fenêtre)")


if __name__ == '__main__':
    main()
