"""
Analyse des erreurs de prédiction par Step Index.

Objectif: Identifier si les erreurs sont concentrées sur certains Steps
(début vs fin de bougie 30min).

Usage:
    python src/analyze_errors.py --data <dataset.npz>
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import argparse
from collections import defaultdict

logger = logging.getLogger(__name__)

from constants import BATCH_SIZE, BEST_MODEL_PATH
from model import create_model
from train import IndicatorDataset
from prepare_data import load_prepared_data


def analyze_errors_by_step(model, dataloader, device, step_index_col=6):
    """
    Analyse les erreurs par Step Index.

    Args:
        model: Modèle entraîné
        dataloader: DataLoader du test set
        device: Device (cuda/cpu)
        step_index_col: Index de la colonne Step Index (défaut: 6 pour 7 features)

    Returns:
        Dict avec statistiques par Step Index
    """
    model.eval()

    # Statistiques par step et par indicateur
    stats = defaultdict(lambda: {
        'total': 0,
        'correct': 0,
        'errors': 0,
        'by_indicator': {
            'RSI': {'total': 0, 'correct': 0, 'errors': 0},
            'CCI': {'total': 0, 'correct': 0, 'errors': 0},
            'MACD': {'total': 0, 'correct': 0, 'errors': 0}
        }
    })

    indicator_names = ['RSI', 'CCI', 'MACD']

    with torch.no_grad():
        for X_batch, Y_batch in dataloader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            # Prédictions
            outputs = model(X_batch)
            preds = (outputs >= 0.5).long()
            targets = Y_batch.long()

            # Extraire le Step Index du dernier timestep
            # X shape: (batch, 12, 7) -> dernier timestep: (batch, 7)
            step_values = X_batch[:, -1, step_index_col].cpu().numpy()

            # Convertir step (0.0-1.0) en catégorie (1-6)
            # 0.0->1, 0.2->2, 0.4->3, 0.6->4, 0.8->5, 1.0->6
            step_categories = np.round(step_values * 5 + 1).astype(int)
            step_categories = np.clip(step_categories, 1, 6)

            # Analyser chaque sample
            for i in range(len(X_batch)):
                step = step_categories[i]

                for j, name in enumerate(indicator_names):
                    pred = preds[i, j].item()
                    target = targets[i, j].item()
                    is_correct = (pred == target)

                    stats[step]['total'] += 1
                    stats[step]['by_indicator'][name]['total'] += 1

                    if is_correct:
                        stats[step]['correct'] += 1
                        stats[step]['by_indicator'][name]['correct'] += 1
                    else:
                        stats[step]['errors'] += 1
                        stats[step]['by_indicator'][name]['errors'] += 1

    return dict(stats)


def print_analysis(stats):
    """Affiche l'analyse des erreurs."""

    print("\n" + "="*80)
    print("ANALYSE DES ERREURS PAR STEP INDEX")
    print("="*80)

    print("\n📊 ACCURACY PAR STEP (global)")
    print("-"*60)
    print(f"{'Step':<8} {'Total':<10} {'Correct':<10} {'Errors':<10} {'Accuracy':<10}")
    print("-"*60)

    for step in sorted(stats.keys()):
        data = stats[step]
        total = data['total']
        correct = data['correct']
        errors = data['errors']
        acc = correct / total if total > 0 else 0

        # Indicateur visuel
        if acc >= 0.87:
            indicator = "🟢"
        elif acc >= 0.83:
            indicator = "🟡"
        else:
            indicator = "🔴"

        print(f"Step {step:<3} {total:<10} {correct:<10} {errors:<10} {acc:.1%} {indicator}")

    # Analyse par indicateur et step
    print("\n" + "="*80)
    print("ACCURACY PAR INDICATEUR ET STEP")
    print("="*80)

    for name in ['RSI', 'CCI', 'MACD']:
        print(f"\n📈 {name}")
        print("-"*50)
        print(f"{'Step':<8} {'Accuracy':<12} {'Errors':<10}")
        print("-"*50)

        for step in sorted(stats.keys()):
            ind_data = stats[step]['by_indicator'][name]
            total = ind_data['total']
            correct = ind_data['correct']
            errors = ind_data['errors']
            acc = correct / total if total > 0 else 0

            bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
            print(f"Step {step:<3} {acc:.1%} {bar} ({errors} err)")

    # Résumé
    print("\n" + "="*80)
    print("RÉSUMÉ")
    print("="*80)

    # Comparer Steps 1-2 vs Steps 5-6
    early_steps = [1, 2]
    late_steps = [5, 6]

    early_correct = sum(stats[s]['correct'] for s in early_steps if s in stats)
    early_total = sum(stats[s]['total'] for s in early_steps if s in stats)
    early_acc = early_correct / early_total if early_total > 0 else 0

    late_correct = sum(stats[s]['correct'] for s in late_steps if s in stats)
    late_total = sum(stats[s]['total'] for s in late_steps if s in stats)
    late_acc = late_correct / late_total if late_total > 0 else 0

    print(f"\n  Steps 1-2 (début bougie 30min): {early_acc:.1%}")
    print(f"  Steps 5-6 (fin bougie 30min):   {late_acc:.1%}")
    print(f"  Delta: {(late_acc - early_acc)*100:+.1f}%")

    if late_acc > early_acc + 0.02:
        print("\n  ⚠️ Le modèle est moins confiant en début de cycle 30min")
        print("  → Considérer le Pivot Filtering pour améliorer Steps 1-2")
    elif early_acc > late_acc + 0.02:
        print("\n  ⚠️ Comportement inattendu: meilleur en début qu'en fin de cycle")
    else:
        print("\n  ✅ Performance stable sur tout le cycle 30min")


def main():
    parser = argparse.ArgumentParser(
        description="Analyse des erreurs par Step Index"
    )
    parser.add_argument('--data', '-d', type=str, required=True,
                        help='Chemin vers les données préparées (.npz)')
    parser.add_argument('--step-col', type=int, default=6,
                        help='Index de la colonne Step Index (défaut: 6)')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Charger données
    print(f"\n📂 Chargement: {args.data}")
    prepared = load_prepared_data(args.data)
    X_test, Y_test = prepared['test']

    num_features = X_test.shape[2]
    print(f"   Features: {num_features}")
    print(f"   Test samples: {len(X_test)}")

    # Vérifier que Step Index existe
    if args.step_col >= num_features:
        print(f"\n❌ Erreur: step_col={args.step_col} mais seulement {num_features} features")
        print("   Pour 7 features (Clock-Injected), Step Index est en colonne 6")
        return

    # DataLoader
    test_dataset = IndicatorDataset(X_test, Y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Charger modèle
    print(f"\n📂 Chargement modèle: {BEST_MODEL_PATH}")
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)

    model, _ = create_model(device=device, num_indicators=num_features)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"   Époque: {checkpoint['epoch']}")

    # Analyser
    print("\n🔍 Analyse des erreurs...")
    stats = analyze_errors_by_step(model, test_loader, device, step_index_col=args.step_col)

    # Afficher résultats
    print_analysis(stats)


if __name__ == '__main__':
    main()
