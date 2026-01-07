"""
Script de vérification de causalité stricte - Kalman vs Octave.

Objectif: Prouver que le lag +1 observé (Kalman Force anticipe Octave)
est dû à une propriété structurelle des filtres, PAS à un data leakage.

Vigilance Critique Expert 2:
> "Bien vérifier que le lag +1 Kalman n'utilise aucune info future indirecte."

Théorie attendue:
- Kalman.filter(): Filtre CAUSAL (online, utilise uniquement passé)
- Octave.filtfilt(): Filtre NON-CAUSAL (bidirectionnel, utilise futur)
→ Lag +1 = Octave attend confirmation bidirectionnelle, Kalman réagit plus tôt

Tests effectués:
1. Vérifier que Kalman[t] ne dépend que de features[:t]
2. Vérifier alignement temporal strict
3. Prouver absence de lookahead bias
4. Documenter propriétés mathématiques des filtres

Usage:
    # Vérifier causalité sur MACD (indicateur pivot)
    python tests/verify_causality.py \
        --data-kalman data/prepared/dataset_btc_eth_bnb_ada_ltc_macd_dual_binary_kalman.npz \
        --data-octave data/prepared/dataset_btc_eth_bnb_ada_ltc_macd_dual_binary_octave20.npz \
        --split test

    # Vérifier tous les indicateurs
    for ind in rsi cci macd; do
        python tests/verify_causality.py \
            --data-kalman data/prepared/dataset_btc_eth_bnb_ada_ltc_${ind}_dual_binary_kalman.npz \
            --data-octave data/prepared/dataset_btc_eth_bnb_ada_ltc_${ind}_dual_binary_octave20.npz \
            --split test
    done
"""

import numpy as np
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import sys

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from filters import kalman_filter, signal_filtfilt


def load_dataset(path: str, split: str = 'test') -> Tuple[np.ndarray, np.ndarray, Optional[Dict]]:
    """
    Charge un dataset dual-binary et retourne X, Y pour le split demandé.

    Réutilisé de: src/compare_filters.py
    """
    data = np.load(path, allow_pickle=True)

    X = data[f'X_{split}']
    Y = data[f'Y_{split}']

    # Charger métadonnées
    metadata = None
    if 'metadata' in data:
        try:
            metadata = json.loads(str(data['metadata']))
        except:
            pass

    return X, Y, metadata


def verify_feature_alignment(X_kalman: np.ndarray, X_octave: np.ndarray) -> Dict:
    """
    Vérifier que les features sont IDENTIQUES entre Kalman et Octave.

    Si features différent → les filtres ont accès à des données différentes → problème!
    """
    print("\n" + "="*80)
    print("TEST #1: ALIGNEMENT FEATURES (X)")
    print("="*80)

    # Vérifier shapes
    if X_kalman.shape != X_octave.shape:
        return {
            'passed': False,
            'error': f'Shapes différentes: Kalman {X_kalman.shape} vs Octave {X_octave.shape}',
            'test': 'Feature Alignment'
        }

    # Comparer features (devraient être identiques)
    diff = np.abs(X_kalman - X_octave)
    is_identical = np.allclose(X_kalman, X_octave, rtol=1e-10, atol=1e-10)

    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    n_different = int(np.sum(diff > 1e-10))
    pct_different = float(n_different / diff.size * 100)

    result = {
        'passed': is_identical,
        'test': 'Feature Alignment',
        'shape': X_kalman.shape,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'n_different': n_different,
        'pct_different': pct_different,
    }

    # Afficher résultats
    print(f"\n   Shape: {result['shape']}")
    print(f"   Features identiques: {'✅ OUI' if result['passed'] else '❌ NON'}")
    print(f"   Max diff: {result['max_diff']:.2e}")
    print(f"   Mean diff: {result['mean_diff']:.2e}")

    if result['passed']:
        print("\n   ✅ VALIDATION: Les deux datasets ont les MÊMES features")
        print("      → Les filtres ont accès aux MÊMES données brutes")
        print("      → Pas de différence d'input qui pourrait expliquer le lag")
    else:
        print(f"\n   ❌ ÉCHEC: {result['pct_different']:.2f}% des features diffèrent")
        print("      → Les filtres ont accès à des données DIFFÉRENTES")
        print("      → Problème de génération des datasets!")

    return result


def verify_temporal_ordering(Y_kalman: np.ndarray, Y_octave: np.ndarray) -> Dict:
    """
    Vérifier que le lag observé est cohérent avec la théorie.

    Théorie:
    - Kalman (causal) devrait réagir AVANT Octave (non-causal)
    - Lag +1 = Kalman en avance (normal)
    - Lag négatif = Kalman en retard (PROBLÈME - lookahead bias?)
    """
    print("\n" + "="*80)
    print("TEST #2: ORDRE TEMPOREL - Lag Kalman vs Octave")
    print("="*80)

    # Séparer Direction et Force
    dir_kalman = Y_kalman[:, 0]
    force_kalman = Y_kalman[:, 1]
    dir_octave = Y_octave[:, 0]
    force_octave = Y_octave[:, 1]

    # Mesurer lag optimal pour Force (découverte #1: lag +1)
    lag_range = range(-5, 6)
    concordances = []

    for lag in lag_range:
        if lag < 0:
            # Kalman en avance
            k_shifted = force_kalman[:lag]
            o_shifted = force_octave[-lag:]
        elif lag > 0:
            # Kalman en retard (ATTENTION!)
            k_shifted = force_kalman[lag:]
            o_shifted = force_octave[:-lag]
        else:
            k_shifted = force_kalman
            o_shifted = force_octave

        n_same = np.sum(k_shifted == o_shifted)
        concordance = float(n_same / len(k_shifted) * 100)
        concordances.append(concordance)

    # Trouver lag optimal
    best_idx = np.argmax(concordances)
    optimal_lag = lag_range[best_idx]
    max_concordance = concordances[best_idx]
    concordance_lag0 = concordances[5]  # lag=0 à index 5

    result = {
        'test': 'Temporal Ordering',
        'optimal_lag': optimal_lag,
        'max_concordance': max_concordance,
        'concordance_lag0': concordance_lag0,
        'all_lags': list(lag_range),
        'all_concordances': concordances,
    }

    # Vérifier si lag est positif (Kalman en retard = PROBLÈME)
    if optimal_lag > 0:
        result['passed'] = False
        result['error'] = f"Lag positif ({optimal_lag}) = Kalman EN RETARD sur Octave → Lookahead bias possible!"
    elif optimal_lag == 0:
        result['passed'] = True
        result['warning'] = "Lag 0 = Synchronisés (pas d'anticipation, mais pas de problème)"
    else:
        result['passed'] = True
        result['note'] = f"Lag négatif ({optimal_lag}) = Kalman EN AVANCE (attendu pour filtre causal)"

    # Afficher résultats
    print(f"\n   Lag optimal Force: {optimal_lag}")
    print(f"   Concordance max: {max_concordance:.1f}% (à lag {optimal_lag})")
    print(f"   Concordance lag=0: {concordance_lag0:.1f}%")

    print("\n   Concordances par lag:")
    for lag, conc in zip(lag_range, concordances):
        marker = "🎯" if lag == optimal_lag else "  "
        direction = "Kalman RETARD" if lag > 0 else ("Synchro" if lag == 0 else "Kalman AVANCE")
        print(f"      {marker} Lag {lag:+2d}: {conc:5.1f}% ({direction})")

    if result['passed']:
        if optimal_lag < 0:
            print("\n   ✅ VALIDATION: Kalman EN AVANCE (lag négatif)")
            print("      → Cohérent avec filtre CAUSAL (réagit avant le filtre non-causal)")
            print("      → PAS de lookahead bias détecté")
        else:
            print("\n   ✅ VALIDATION: Lag 0 (synchronisés)")
            print("      → Pas d'anticipation, mais pas de problème de causalité")
    else:
        print(f"\n   ❌ ALERTE: Lag POSITIF ({optimal_lag})")
        print("      → Kalman EN RETARD sur Octave = ANORMAL pour filtre causal")
        print("      → Possible lookahead bias (Kalman utilise info future?)")

    return result


def verify_kalman_causality_property() -> Dict:
    """
    Vérifier mathématiquement que Kalman.filter() est causal.

    Test: Appliquer Kalman sur signal synthétique, vérifier que filtered[t]
    ne change PAS si on ajoute des données après t.
    """
    print("\n" + "="*80)
    print("TEST #3: PROPRIÉTÉ MATHÉMATIQUE - Kalman.filter() est-il causal?")
    print("="*80)

    # Créer signal synthétique
    np.random.seed(42)
    signal_full = np.cumsum(np.random.randn(1000)) + 100  # Random walk

    # Appliquer Kalman sur signal complet
    filtered_full = kalman_filter(signal_full, process_variance=0.01)

    # Appliquer Kalman sur signal tronqué (jusqu'à t=500)
    signal_partial = signal_full[:500]
    filtered_partial = kalman_filter(signal_partial, process_variance=0.01)

    # Comparer filtered_partial avec filtered_full[:500]
    # Si Kalman est causal: filtered_partial == filtered_full[:500]
    # Si Kalman utilise le futur: filtered_partial != filtered_full[:500]

    diff = np.abs(filtered_partial - filtered_full[:500])
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    is_causal = np.allclose(filtered_partial, filtered_full[:500], rtol=1e-10, atol=1e-10)

    result = {
        'test': 'Kalman Causality Property',
        'passed': is_causal,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
    }

    print(f"\n   Signal complet: {len(signal_full)} samples")
    print(f"   Signal tronqué: {len(signal_partial)} samples")
    print(f"   Max diff: {max_diff:.2e}")
    print(f"   Mean diff: {mean_diff:.2e}")

    if result['passed']:
        print("\n   ✅ VALIDATION: Kalman.filter() est CAUSAL")
        print("      → filtered[t] ne dépend QUE de signal[:t+1]")
        print("      → Ajouter des données futures ne change PAS le passé")
        print("      → Propriété mathématique confirmée (Kalman 1960)")
    else:
        print(f"\n   ❌ ÉCHEC: Kalman.filter() semble NON-CAUSAL")
        print(f"      → Max diff: {max_diff:.2e} (devrait être ~0)")
        print("      → Implémentation incorrecte de pykalman?")

    return result


def verify_octave_noncausal_property() -> Dict:
    """
    Vérifier mathématiquement que Octave.filtfilt() est NON-CAUSAL.

    Test: Appliquer filtfilt sur signal synthétique, vérifier que filtered[t]
    CHANGE si on ajoute des données après t.
    """
    print("\n" + "="*80)
    print("TEST #4: PROPRIÉTÉ MATHÉMATIQUE - Octave.filtfilt() est-il non-causal?")
    print("="*80)

    # Créer signal synthétique
    np.random.seed(42)
    signal_full = np.cumsum(np.random.randn(1000)) + 100

    # Appliquer Octave sur signal complet
    filtered_full = signal_filtfilt(signal_full, step=0.2, order=3)

    # Appliquer Octave sur signal tronqué
    signal_partial = signal_full[:500]
    filtered_partial = signal_filtfilt(signal_partial, step=0.2, order=3)

    # Comparer: Si NON-CAUSAL, filtered_partial != filtered_full[:500]
    diff = np.abs(filtered_partial - filtered_full[:500])
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    is_different = max_diff > 1e-6  # Devrait être différent

    result = {
        'test': 'Octave Non-Causality Property',
        'passed': is_different,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
    }

    print(f"\n   Signal complet: {len(signal_full)} samples")
    print(f"   Signal tronqué: {len(signal_partial)} samples")
    print(f"   Max diff: {max_diff:.2e}")
    print(f"   Mean diff: {mean_diff:.2e}")

    if result['passed']:
        print("\n   ✅ VALIDATION: Octave.filtfilt() est NON-CAUSAL")
        print("      → filtered[t] DÉPEND de signal futur (bidirectionnel)")
        print("      → Ajouter des données futures CHANGE le passé")
        print("      → Confirme documentation: filtfilt utilise le futur")
    else:
        print(f"\n   ❌ ÉCHEC: Octave.filtfilt() semble causal")
        print(f"      → Max diff: {max_diff:.2e} (devrait être significatif)")
        print("      → Implémentation incorrecte?")

    return result


def verify_lag_interpretation(optimal_lag: int) -> Dict:
    """
    Interpréter le lag observé selon la théorie des filtres.
    """
    print("\n" + "="*80)
    print("TEST #5: INTERPRÉTATION THÉORIQUE - Lag +1 Kalman vs Octave")
    print("="*80)

    result = {
        'test': 'Lag Interpretation',
        'optimal_lag': optimal_lag,
    }

    print(f"\n   Lag observé: {optimal_lag}")

    if optimal_lag < 0:
        # Kalman en avance (attendu)
        abs_lag = abs(optimal_lag)
        result['passed'] = True
        result['interpretation'] = f"Kalman détecte {abs_lag} période(s) AVANT Octave"
        result['theory'] = "Kalman (causal) réagit plus tôt que Octave (non-causal)"
        result['trading_implication'] = f"Signal d'anticipation de {abs_lag * 5}min"

        print(f"\n   Interprétation: Kalman détecte {abs_lag} période(s) AVANT Octave")
        print(f"      → Kalman (filtre causal): Réagit immédiatement aux changements")
        print(f"      → Octave (filtre non-causal): Attend confirmation bidirectionnelle")
        print(f"      → Lag négatif = ATTENDU pour cette architecture")

        print(f"\n   ✅ COHÉRENCE THÉORIQUE:")
        print(f"      - Kalman.filter(): Online filtering (causal)")
        print(f"      - Octave.filtfilt(): Bidirectionnel (non-causal)")
        print(f"      → Kalman anticipe de {abs_lag * 5}min (lag {optimal_lag})")

        print(f"\n   💡 TRADING INSIGHT:")
        print(f"      - Kalman Force = Early Warning System")
        print(f"      - Octave Force = Confirmation ({abs_lag * 5}min plus tard)")
        print(f"      - Pas de data leakage, c'est une propriété structurelle")

    elif optimal_lag == 0:
        result['passed'] = True
        result['interpretation'] = "Kalman et Octave synchronisés"
        result['theory'] = "Pas d'anticipation détectable"

        print(f"\n   Interprétation: Kalman et Octave synchronisés")
        print(f"      → Pas d'anticipation détectable")
        print(f"      → Pas de problème de causalité")

    else:
        # Kalman en retard (PROBLÈME)
        result['passed'] = False
        result['interpretation'] = f"Kalman détecte {optimal_lag} période(s) APRÈS Octave"
        result['theory'] = "ANORMAL - Kalman causal devrait être en avance"
        result['error'] = "Possible lookahead bias dans Kalman"

        print(f"\n   ⚠️ ALERTE: Kalman détecte {optimal_lag} période(s) APRÈS Octave")
        print(f"      → ANORMAL pour un filtre causal")
        print(f"      → Kalman devrait réagir AVANT Octave (non-causal)")
        print(f"      → Possible lookahead bias (Kalman utilise info future?)")

        print(f"\n   ❌ INCOHÉRENCE THÉORIQUE:")
        print(f"      - Kalman.filter() devrait être causal")
        print(f"      - Octave.filtfilt() est non-causal")
        print(f"      → Lag positif = IMPOSSIBLE sans data leakage")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Vérifier causalité stricte Kalman vs Octave (Vigilance Expert #1)"
    )
    parser.add_argument('--data-kalman', type=str, required=True,
                       help='Dataset Kalman (.npz)')
    parser.add_argument('--data-octave', type=str, required=True,
                       help='Dataset Octave (.npz)')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Split à analyser (défaut: test)')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("VÉRIFICATION CAUSALITÉ STRICTE - Kalman vs Octave")
    print("="*80)
    print(f"\nDataset Kalman: {args.data_kalman}")
    print(f"Dataset Octave: {args.data_octave}")
    print(f"Split: {args.split}")

    # Charger datasets
    print("\n📂 Chargement datasets...")
    X_kalman, Y_kalman, meta_kalman = load_dataset(args.data_kalman, args.split)
    X_octave, Y_octave, meta_octave = load_dataset(args.data_octave, args.split)

    print(f"   Kalman: X={X_kalman.shape}, Y={Y_kalman.shape}")
    print(f"   Octave: X={X_octave.shape}, Y={Y_octave.shape}")

    # Résultats
    results = {}

    # Test #1: Alignement features
    results['feature_alignment'] = verify_feature_alignment(X_kalman, X_octave)

    # Test #2: Ordre temporel (lag)
    results['temporal_ordering'] = verify_temporal_ordering(Y_kalman, Y_octave)
    optimal_lag = results['temporal_ordering']['optimal_lag']

    # Test #3: Propriété mathématique Kalman (causal)
    results['kalman_causality'] = verify_kalman_causality_property()

    # Test #4: Propriété mathématique Octave (non-causal)
    results['octave_noncausality'] = verify_octave_noncausal_property()

    # Test #5: Interprétation lag
    results['lag_interpretation'] = verify_lag_interpretation(optimal_lag)

    # Résumé final
    print("\n" + "="*80)
    print("RÉSUMÉ FINAL - VALIDATION CAUSALITÉ")
    print("="*80)

    all_passed = all(r.get('passed', False) for r in results.values())

    print("\n📊 Résultats par test:")
    for test_name, result in results.items():
        status = "✅ PASS" if result.get('passed', False) else "❌ FAIL"
        print(f"   {status} - {result['test']}")
        if 'error' in result:
            print(f"         ⚠️  {result['error']}")
        elif 'warning' in result:
            print(f"         ⚠️  {result['warning']}")

    print("\n" + "="*80)
    if all_passed:
        print("✅ VALIDATION GLOBALE: CAUSALITÉ STRICTE CONFIRMÉE")
        print("="*80)
        print("\n💡 Conclusions:")
        print("   1. ✅ Features identiques (pas de différence d'input)")
        print("   2. ✅ Kalman.filter() est CAUSAL (propriété mathématique)")
        print("   3. ✅ Octave.filtfilt() est NON-CAUSAL (utilise futur)")
        print(f"   4. ✅ Lag {optimal_lag} cohérent avec théorie des filtres")
        print("   5. ✅ PAS de lookahead bias détecté")

        print("\n🎯 Réponse à la Vigilance Expert #2:")
        print("   > 'Bien vérifier que le lag +1 Kalman n'utilise aucune info future indirecte.'")
        print("\n   ✅ VALIDÉ: Kalman n'utilise AUCUNE info future")
        print("   ✅ Lag observé = Propriété structurelle (causal vs non-causal)")
        print("   ✅ Architecture Multi-Capteurs validée (Early Warning + Confirmation)")

    else:
        print("❌ VALIDATION GLOBALE: PROBLÈME DE CAUSALITÉ DÉTECTÉ")
        print("="*80)
        print("\n⚠️  ALERTE: Tests échoués:")
        for test_name, result in results.items():
            if not result.get('passed', False):
                print(f"   - {result['test']}")
                if 'error' in result:
                    print(f"     {result['error']}")

        print("\n🚨 ACTIONS REQUISES:")
        print("   1. Vérifier implémentation Kalman dans prepare_data*.py")
        print("   2. Auditer génération labels (pas de lookahead?)")
        print("   3. Revalider architecture avant implémentation")


if __name__ == '__main__':
    main()
