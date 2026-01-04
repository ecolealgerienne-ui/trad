"""
Apprentissage de règles CART pour raffiner la State Machine.

Architecture:
    CNN-LSTM → Prédictions (MACD, RSI, CCI)
                    ↓
            Agreement == TOTAL ?
                    ↓
                CART léger
                /    |    \
            ENTER  HOLD  EXIT

CART n'agit QUE dans la zone TOTAL où on a déjà un edge.
Il apprend les nuances : quand entrer/sortir/attendre.

Paramètres recommandés (expert):
    - max_depth = 3-4 (règles simples)
    - min_samples_leaf = élevé (éviter sur-apprentissage)
    - criterion = gini (stable avec signaux corrélés)
    - class_weight asymétrique (EXIT > ENTER)

Usage:
    python src/learn_cart_policy.py \
        --rsi-octave data/prepared/dataset_..._rsi_octave20.npz \
        --cci-octave data/prepared/dataset_..._cci_octave20.npz \
        --macd-octave data/prepared/dataset_..._macd_octave20.npz \
        --rsi-kalman data/prepared/dataset_..._rsi_kalman.npz \
        --cci-kalman data/prepared/dataset_..._cci_kalman.npz \
        --macd-kalman data/prepared/dataset_..._macd_kalman.npz \
        --split train --max-depth 4
"""

import numpy as np
import argparse
from pathlib import Path
from typing import Tuple, List, Dict
from enum import IntEnum
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class Action(IntEnum):
    """Actions possibles."""
    EXIT = -1   # Sortir de position
    HOLD = 0    # Ne rien faire
    ENTER = 1   # Entrer en position


class Agreement(IntEnum):
    """Niveaux d'accord."""
    FORT = 0      # Désaccord fort
    PARTIEL = 1   # Désaccord partiel
    TOTAL = 2     # Accord total


def load_dataset(path: str, split: str = 'train') -> dict:
    """Charge un dataset."""
    data = np.load(path, allow_pickle=True)

    result = {
        'X': data[f'X_{split}'],
        'Y': data[f'Y_{split}'],
        'Y_pred': data.get(f'Y_{split}_pred', None),
        'assets': None
    }

    # Charger les métadonnées
    if 'metadata' in data:
        try:
            import json
            metadata = json.loads(str(data['metadata']))
            result['assets'] = metadata.get('assets', None)
        except:
            pass

    if 'assets' in data:
        result['assets'] = list(data['assets'])

    return result


def get_agreement_level(macd: int, rsi: int, cci: int,
                        octave: int, kalman: int) -> Agreement:
    """Calcule le niveau d'accord des signaux."""
    indicators_agree = (macd == rsi == cci)
    filters_agree = (octave == kalman)

    if indicators_agree and filters_agree:
        return Agreement.TOTAL
    elif not indicators_agree and not filters_agree:
        return Agreement.FORT
    else:
        return Agreement.PARTIEL


def calculate_oracle_actions(returns: np.ndarray,
                            lookahead: int = 6) -> np.ndarray:
    """
    Calcule les actions Oracle (rétrospectives).

    Oracle sait ce qui va se passer dans les N prochaines périodes.

    Args:
        returns: Rendements (c_ret) pour chaque période
        lookahead: Nombre de périodes à regarder en avant

    Returns:
        actions: Array d'actions (ENTER=1, HOLD=0, EXIT=-1)
    """
    n = len(returns)
    actions = np.zeros(n, dtype=int)

    for i in range(n - lookahead):
        # Rendement cumulé sur les N prochaines périodes
        future_return = np.sum(returns[i+1:i+1+lookahead])

        # Seuil pour considérer un move significatif (> frais)
        threshold = 0.002  # 0.2% = couvre les frais

        if future_return > threshold:
            actions[i] = Action.ENTER  # Devrait être LONG
        elif future_return < -threshold:
            actions[i] = Action.EXIT   # Devrait sortir/shorter
        else:
            actions[i] = Action.HOLD   # Pas assez de mouvement

    return actions


def calculate_oracle_actions_v2(returns: np.ndarray,
                                positions: np.ndarray = None,
                                min_profit: float = 0.002) -> np.ndarray:
    """
    Calcule les actions Oracle basées sur l'état actuel.

    Pour chaque sample:
    - Si FLAT: ENTER si le prochain move > min_profit
    - Si en position: EXIT si le move va contre nous, HOLD sinon

    Args:
        returns: Rendements futurs
        positions: Positions actuelles (1=LONG, -1=SHORT, 0=FLAT)
        min_profit: Seuil minimum pour justifier une entrée

    Returns:
        actions: ENTER, HOLD, EXIT
    """
    n = len(returns)
    actions = np.zeros(n, dtype=int)

    # Lookahead pour décision
    lookahead = 6

    for i in range(n - lookahead):
        future_return = np.sum(returns[i+1:i+1+lookahead])

        # Simplification: action basée sur direction future
        if future_return > min_profit:
            actions[i] = Action.ENTER  # GO LONG
        elif future_return < -min_profit:
            actions[i] = Action.EXIT   # GO SHORT ou EXIT LONG
        else:
            actions[i] = Action.HOLD

    return actions


def build_features(datasets: dict, split: str,
                   threshold: float = 0.5) -> Tuple[np.ndarray, List[str]]:
    """
    Construit les features pour CART.

    Features:
        - macd_prob, rsi_prob, cci_prob: Probabilités ML [0,1]
        - macd_conf, rsi_conf, cci_conf: Confiance |p-0.5|
        - min_conf: Min des 3 confiances
        - volatility: range_ret (ATR proxy)
        - octave_dir, kalman_dir: Direction des filtres
        - filters_agree: 1 si octave == kalman

    Returns:
        X: Features array
        feature_names: Noms des features
    """
    # Extraire les prédictions (probabilités)
    macd_prob = datasets['macd_octave']['Y_pred'].flatten()
    rsi_prob = datasets['rsi_octave']['Y_pred'].flatten()
    cci_prob = datasets['cci_octave']['Y_pred'].flatten()

    # Confiances
    macd_conf = np.abs(macd_prob - 0.5)
    rsi_conf = np.abs(rsi_prob - 0.5)
    cci_conf = np.abs(cci_prob - 0.5)
    min_conf = np.minimum(np.minimum(macd_conf, rsi_conf), cci_conf)

    # Directions des filtres (labels = pente filtrée)
    macd_octave = datasets['macd_octave']['Y'].flatten()
    rsi_octave = datasets['rsi_octave']['Y'].flatten()
    cci_octave = datasets['cci_octave']['Y'].flatten()

    macd_kalman = datasets['macd_kalman']['Y'].flatten()
    rsi_kalman = datasets['rsi_kalman']['Y'].flatten()
    cci_kalman = datasets['cci_kalman']['Y'].flatten()

    # Filtres d'accord
    octave_dir = macd_octave  # MACD comme référence
    kalman_dir = macd_kalman
    filters_agree = (octave_dir == kalman_dir).astype(float)

    # Volatilité (range_ret = index 4)
    X_data = datasets['macd_octave']['X']
    volatility = X_data[:, -1, 4]  # Dernier timestep, range_ret

    # Binariser pour agreement
    macd_bin = (macd_prob >= threshold).astype(int)
    rsi_bin = (rsi_prob >= threshold).astype(int)
    cci_bin = (cci_prob >= threshold).astype(int)

    # Indicateurs d'accord
    indicators_agree = ((macd_bin == rsi_bin) & (rsi_bin == cci_bin)).astype(float)

    # Construire le feature array
    X = np.column_stack([
        macd_prob,        # 0: Probabilité MACD
        rsi_prob,         # 1: Probabilité RSI
        cci_prob,         # 2: Probabilité CCI
        min_conf,         # 3: Confiance minimale
        volatility,       # 4: Volatilité (ATR proxy)
        filters_agree,    # 5: Filtres d'accord
        indicators_agree, # 6: Indicateurs d'accord
        macd_conf,        # 7: Confiance MACD
    ])

    feature_names = [
        'macd_prob',
        'rsi_prob',
        'cci_prob',
        'min_conf',
        'volatility',
        'filters_agree',
        'indicators_agree',
        'macd_conf'
    ]

    return X, feature_names


def get_agreement_mask(datasets: dict, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcule le masque d'accord TOTAL et les niveaux d'accord.

    Returns:
        total_mask: Booléen, True si accord TOTAL
        agreements: Niveau d'accord pour chaque sample
    """
    macd_prob = datasets['macd_octave']['Y_pred'].flatten()
    rsi_prob = datasets['rsi_octave']['Y_pred'].flatten()
    cci_prob = datasets['cci_octave']['Y_pred'].flatten()

    macd_octave = datasets['macd_octave']['Y'].flatten()
    macd_kalman = datasets['macd_kalman']['Y'].flatten()

    # Binariser
    macd_bin = (macd_prob >= threshold).astype(int)
    rsi_bin = (rsi_prob >= threshold).astype(int)
    cci_bin = (cci_prob >= threshold).astype(int)

    n_samples = len(macd_prob)
    agreements = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        agreement = get_agreement_level(
            macd_bin[i], rsi_bin[i], cci_bin[i],
            int(macd_octave[i]), int(macd_kalman[i])
        )
        agreements[i] = agreement

    total_mask = (agreements == Agreement.TOTAL)

    return total_mask, agreements


def train_cart(X: np.ndarray, y: np.ndarray,
               max_depth: int = 4,
               min_samples_leaf: int = 2000,
               class_weight: dict = None) -> DecisionTreeClassifier:
    """
    Entraîne un arbre CART avec les paramètres recommandés.

    Args:
        X: Features
        y: Labels (actions Oracle)
        max_depth: Profondeur max (3-4 recommandé)
        min_samples_leaf: Min samples par feuille (élevé = stable)
        class_weight: Poids par classe (asymétrique recommandé)

    Returns:
        Arbre CART entraîné
    """
    if class_weight is None:
        # Asymétrie CORRIGÉE:
        # - EXIT poids ÉLEVÉ = modèle réactif aux sorties (ne pas les rater)
        # - ENTER poids FAIBLE = modèle prudent aux entrées
        # sklearn: poids = pénalité pour erreur sur cette classe
        class_weight = {
            Action.EXIT: 2.0,   # IMPORTANT: ne pas rater les sorties
            Action.HOLD: 1.0,   # Neutre
            Action.ENTER: 0.5   # Prudent: mieux vaut rater une entrée
        }

    cart = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        criterion='gini',
        class_weight=class_weight,
        random_state=42
    )

    cart.fit(X, y)

    return cart


def extract_rules(cart: DecisionTreeClassifier,
                  feature_names: List[str]) -> str:
    """Extrait les règles lisibles de l'arbre."""
    return export_text(cart, feature_names=feature_names, decimals=4)


def evaluate_cart(cart: DecisionTreeClassifier,
                  X: np.ndarray, y: np.ndarray,
                  split_name: str = "Test") -> dict:
    """Évalue l'arbre CART."""
    y_pred = cart.predict(X)

    # Métriques
    action_names = ['EXIT', 'HOLD', 'ENTER']
    report = classification_report(y, y_pred,
                                   target_names=action_names,
                                   output_dict=True,
                                   zero_division=0)

    cm = confusion_matrix(y, y_pred, labels=[-1, 0, 1])

    accuracy = (y_pred == y).mean()

    return {
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm,
        'predictions': y_pred
    }


def main():
    parser = argparse.ArgumentParser(
        description="Apprendre des règles CART pour la State Machine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Datasets Octave20
    parser.add_argument('--rsi-octave', type=str, required=True)
    parser.add_argument('--cci-octave', type=str, required=True)
    parser.add_argument('--macd-octave', type=str, required=True)

    # Datasets Kalman
    parser.add_argument('--rsi-kalman', type=str, required=True)
    parser.add_argument('--cci-kalman', type=str, required=True)
    parser.add_argument('--macd-kalman', type=str, required=True)

    # Options
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--max-depth', type=int, default=4,
                        help='Profondeur max de l\'arbre')
    parser.add_argument('--min-samples-leaf', type=int, default=2000,
                        help='Min samples par feuille')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Seuil de binarisation des probabilités')
    parser.add_argument('--lookahead', type=int, default=6,
                        help='Périodes à regarder pour Oracle')
    parser.add_argument('--min-profit', type=float, default=0.002,
                        help='Profit min pour justifier une action (0.2%%)')
    parser.add_argument('--total-only', action='store_true',
                        help='Entraîner uniquement sur accord TOTAL')
    parser.add_argument('--eval-split', type=str, default=None,
                        help='Split pour évaluation (défaut: même que --split)')
    parser.add_argument('--balanced', action='store_true',
                        help='Utiliser des poids égaux (1.0, 1.0, 1.0) au lieu d\'asymétriques')
    parser.add_argument('--two-class', action='store_true',
                        help='Reformuler en 2 classes: AGIR vs HOLD (direction via macd_prob)')

    args = parser.parse_args()

    print("="*80)
    print("CART POLICY LEARNING")
    print("="*80)

    # Charger les datasets
    print(f"\n📂 Chargement des datasets ({args.split})...")

    datasets = {}

    print("   Loading RSI Octave20...")
    datasets['rsi_octave'] = load_dataset(args.rsi_octave, args.split)
    print("   Loading CCI Octave20...")
    datasets['cci_octave'] = load_dataset(args.cci_octave, args.split)
    print("   Loading MACD Octave20...")
    datasets['macd_octave'] = load_dataset(args.macd_octave, args.split)

    print("   Loading RSI Kalman...")
    datasets['rsi_kalman'] = load_dataset(args.rsi_kalman, args.split)
    print("   Loading CCI Kalman...")
    datasets['cci_kalman'] = load_dataset(args.cci_kalman, args.split)
    print("   Loading MACD Kalman...")
    datasets['macd_kalman'] = load_dataset(args.macd_kalman, args.split)

    # Vérifier les prédictions
    for name, data in datasets.items():
        if 'octave' in name and data['Y_pred'] is None:
            print(f"\n❌ ERREUR: {name} n'a pas de prédictions!")
            return

    # Construire les features
    print("\n🔧 Construction des features...")
    X, feature_names = build_features(datasets, args.split, args.threshold)
    n_samples = len(X)
    print(f"   Samples: {n_samples:,}")
    print(f"   Features: {len(feature_names)} ({', '.join(feature_names)})")

    # Calculer les actions Oracle
    print("\n🎯 Calcul des actions Oracle...")
    X_data = datasets['macd_octave']['X']
    returns = X_data[:, -1, 3]  # c_ret

    y_oracle = calculate_oracle_actions_v2(
        returns,
        min_profit=args.min_profit
    )

    # Distribution des actions Oracle
    n_enter = (y_oracle == Action.ENTER).sum()
    n_hold = (y_oracle == Action.HOLD).sum()
    n_exit = (y_oracle == Action.EXIT).sum()
    print(f"   ENTER: {n_enter:,} ({n_enter/n_samples*100:.1f}%)")
    print(f"   HOLD: {n_hold:,} ({n_hold/n_samples*100:.1f}%)")
    print(f"   EXIT: {n_exit:,} ({n_exit/n_samples*100:.1f}%)")

    # Filtrer sur accord TOTAL si demandé
    if args.total_only:
        print("\n🎯 Filtrage sur accord TOTAL uniquement...")
        total_mask, agreements = get_agreement_mask(datasets, args.threshold)
        n_total = total_mask.sum()
        print(f"   Samples TOTAL: {n_total:,} ({n_total/n_samples*100:.1f}%)")

        X_train = X[total_mask]
        y_train = y_oracle[total_mask]

        # Re-afficher distribution dans TOTAL
        n_enter = (y_train == Action.ENTER).sum()
        n_hold = (y_train == Action.HOLD).sum()
        n_exit = (y_train == Action.EXIT).sum()
        print(f"   ENTER (dans TOTAL): {n_enter:,} ({n_enter/len(y_train)*100:.1f}%)")
        print(f"   HOLD (dans TOTAL): {n_hold:,} ({n_hold/len(y_train)*100:.1f}%)")
        print(f"   EXIT (dans TOTAL): {n_exit:,} ({n_exit/len(y_train)*100:.1f}%)")
    else:
        X_train = X
        y_train = y_oracle

    # Mode 2 classes: AGIR vs HOLD
    if args.two_class:
        print("\n🔄 Reformulation en 2 classes: AGIR vs HOLD...")
        # AGIR = abs(action) == 1 (ENTER ou EXIT)
        # HOLD = action == 0
        y_train_binary = (np.abs(y_train) > 0).astype(int)  # 1=AGIR, 0=HOLD
        n_agir = y_train_binary.sum()
        n_hold = len(y_train_binary) - n_agir
        print(f"   AGIR: {n_agir:,} ({n_agir/len(y_train_binary)*100:.1f}%)")
        print(f"   HOLD: {n_hold:,} ({n_hold/len(y_train_binary)*100:.1f}%)")
        y_train = y_train_binary

    # Entraîner CART
    print(f"\n🌳 Entraînement CART...")
    print(f"   max_depth: {args.max_depth}")
    print(f"   min_samples_leaf: {args.min_samples_leaf}")
    print(f"   criterion: gini")

    # Déterminer les poids
    if args.balanced:
        if args.two_class:
            class_weight = {0: 1.0, 1: 1.0}
            print(f"   class_weight: HOLD=1.0, AGIR=1.0 (équilibré)")
        else:
            class_weight = {Action.EXIT: 1.0, Action.HOLD: 1.0, Action.ENTER: 1.0}
            print(f"   class_weight: EXIT=1.0, HOLD=1.0, ENTER=1.0 (équilibré)")
    else:
        if args.two_class:
            class_weight = {0: 1.0, 1: 1.5}  # Favoriser AGIR légèrement
            print(f"   class_weight: HOLD=1.0, AGIR=1.5")
        else:
            class_weight = None  # Utiliser les défauts asymétriques
            print(f"   class_weight: EXIT=2.0, HOLD=1.0, ENTER=0.5 (asymétrique)")

    cart = train_cart(
        X_train, y_train,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight=class_weight
    )

    print(f"\n   Nodes: {cart.tree_.node_count}")
    print(f"   Leaves: {cart.get_n_leaves()}")
    print(f"   Depth: {cart.get_depth()}")

    # Extraire les règles
    print("\n" + "="*80)
    print("RÈGLES APPRISES (CART)")
    print("="*80 + "\n")

    rules = extract_rules(cart, feature_names)
    print(rules)

    # Évaluer sur le split d'entraînement
    print("\n" + "="*80)
    print(f"ÉVALUATION ({args.split.upper()})")
    print("="*80)

    # Noms des classes selon le mode
    if args.two_class:
        action_names = ['HOLD', 'AGIR']
        labels = [0, 1]
    else:
        action_names = ['EXIT', 'HOLD', 'ENTER']
        labels = [-1, 0, 1]

    y_pred = cart.predict(X_train)
    accuracy = (y_pred == y_train).mean()

    print(f"\n📊 Accuracy: {accuracy*100:.1f}%")

    print(f"\n📊 Rapport par classe:")
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(
        y_train, y_pred, labels=labels, zero_division=0
    )
    for i, name in enumerate(action_names):
        print(f"   {name}: precision={precision[i]:.2f}, recall={recall[i]:.2f}, f1={f1[i]:.2f}, support={support[i]:,}")

    print(f"\n📊 Matrice de confusion:")
    cm = confusion_matrix(y_train, y_pred, labels=labels)
    if args.two_class:
        print("            Prédit")
        print("           HOLD  AGIR")
        print(f"Réel HOLD  {cm[0,0]:6d} {cm[0,1]:6d}")
        print(f"     AGIR  {cm[1,0]:6d} {cm[1,1]:6d}")
    else:
        print("            Prédit")
        print("           EXIT HOLD ENTER")
        print(f"Réel EXIT  {cm[0,0]:5d} {cm[0,1]:5d} {cm[0,2]:5d}")
        print(f"     HOLD  {cm[1,0]:5d} {cm[1,1]:5d} {cm[1,2]:5d}")
        print(f"     ENTER {cm[2,0]:5d} {cm[2,1]:5d} {cm[2,2]:5d}")

    # Évaluer sur un autre split si demandé
    if args.eval_split and args.eval_split != args.split:
        print(f"\n" + "="*80)
        print(f"ÉVALUATION ({args.eval_split.upper()})")
        print("="*80)

        # Charger le split d'évaluation
        datasets_eval = {}
        for key in datasets:
            base_path = args.__dict__.get(key.replace('_', '-').replace('octave', 'octave').replace('kalman', 'kalman'), None)
            if base_path is None:
                if 'rsi_octave' in key:
                    base_path = args.rsi_octave
                elif 'cci_octave' in key:
                    base_path = args.cci_octave
                elif 'macd_octave' in key:
                    base_path = args.macd_octave
                elif 'rsi_kalman' in key:
                    base_path = args.rsi_kalman
                elif 'cci_kalman' in key:
                    base_path = args.cci_kalman
                elif 'macd_kalman' in key:
                    base_path = args.macd_kalman
            datasets_eval[key] = load_dataset(base_path, args.eval_split)

        X_eval, _ = build_features(datasets_eval, args.eval_split, args.threshold)
        X_data_eval = datasets_eval['macd_octave']['X']
        returns_eval = X_data_eval[:, -1, 3]
        y_eval = calculate_oracle_actions_v2(returns_eval, min_profit=args.min_profit)

        if args.total_only:
            total_mask_eval, _ = get_agreement_mask(datasets_eval, args.threshold)
            X_eval = X_eval[total_mask_eval]
            y_eval = y_eval[total_mask_eval]

        eval_result_test = evaluate_cart(cart, X_eval, y_eval, args.eval_split)

        print(f"\n📊 Accuracy: {eval_result_test['accuracy']*100:.1f}%")
        print(f"\n📊 Rapport par classe:")
        for action_name in ['EXIT', 'HOLD', 'ENTER']:
            if action_name in eval_result_test['report']:
                r = eval_result_test['report'][action_name]
                print(f"   {action_name}: precision={r['precision']:.2f}, recall={r['recall']:.2f}, f1={r['f1-score']:.2f}")

    # Feature importance
    print(f"\n" + "="*80)
    print("IMPORTANCE DES FEATURES")
    print("="*80 + "\n")

    importances = cart.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    for idx in sorted_idx:
        if importances[idx] > 0.01:
            print(f"   {feature_names[idx]:20s}: {importances[idx]*100:.1f}%")

    # Résumé des règles principales
    print(f"\n" + "="*80)
    print("RÈGLES PRINCIPALES À INTÉGRER")
    print("="*80 + "\n")

    print("Basé sur l'arbre CART, les règles à intégrer dans la state machine sont:")
    print("(Interpréter manuellement l'arbre ci-dessus pour extraire les conditions)")
    print()
    print("Exemple de lecture:")
    print("  |--- volatility <= 0.0021 → classe majoritaire")
    print("  |--- volatility >  0.0021")
    print("  |    |--- min_conf <= 0.35 → HOLD")
    print("  |    |--- min_conf >  0.35 → ENTER")
    print()

    print("="*80)
    print("FIN")
    print("="*80)


if __name__ == '__main__':
    main()
