#!/usr/bin/env python
"""
Test rapide des arguments CLI sans importer PyTorch.
"""
import argparse
import sys
sys.path.insert(0, 'src')

from constants import BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, EARLY_STOPPING_PATIENCE

def parse_args():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(
        description='Entraînement du modèle CNN-LSTM Multi-Output',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Hyperparamètres d'entraînement
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Taille des batches')
    parser.add_argument('--lr', '--learning-rate', type=float, default=LEARNING_RATE,
                        dest='learning_rate', help='Learning rate')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help='Nombre maximum d\'époques')
    parser.add_argument('--patience', type=int, default=EARLY_STOPPING_PATIENCE,
                        help='Patience pour early stopping')

    # Chemins
    parser.add_argument('--save-path', type=str, default='models/best_model.pth',
                        help='Chemin pour sauvegarder le meilleur modèle')

    # Génération de labels
    parser.add_argument('--filter', type=str, default='decycler',
                        choices=['decycler', 'kalman'],
                        help='Type de filtre pour générer les labels (decycler ou kalman)')

    # Autres
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed pour reproductibilité')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device à utiliser (auto détecte automatiquement)')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    print("="*80)
    print("TEST DES ARGUMENTS CLI")
    print("="*80)
    print(f"\n⚙️ Hyperparamètres reçus:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Max epochs: {args.epochs}")
    print(f"  Early stopping patience: {args.patience}")
    print(f"  Filter type: {args.filter}")
    print(f"  Random seed: {args.seed}")
    print(f"  Device: {args.device}")
    print(f"  Save path: {args.save_path}")
    print("\n✅ Arguments CLI fonctionnels!")
