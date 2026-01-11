# Phase 2.18 - Modifications Aligned Meta-Labels

**Date**: 2026-01-11
**Objectif**: Support aligned meta-labels (signal reversal) vs Triple Barrier

---

## 📋 Vue d'Ensemble

Les fichiers aligned `.npz` existent déjà:
```
data/prepared/meta_labels_macd_kalman_{train,val,test}_aligned.npz
```

**Modifications nécessaires**: 2 scripts seulement
1. `src/train_meta_model_phase217.py` - Ajout flag `--aligned`
2. `tests/test_meta_model_backtest.py` - Ajout flag `--aligned`

---

## 📄 Modification 1: `src/train_meta_model_phase217.py`

### Change 1.1 - Fonction `load_meta_dataset` (ligne 46)

**Ajouter paramètre `aligned`:**

```python
def load_meta_dataset(split: str, indicator: str = 'macd', filter_type: str = 'kalman', aligned: bool = False) -> Dict:
    """
    Charge le dataset meta-labels avec prédictions.

    Args:
        split: 'train', 'val', ou 'test'
        indicator: Indicateur utilisé pour meta-labels (default: 'macd')
        filter_type: Type de filtre (default: 'kalman')
        aligned: Si True, charge labels aligned (signal reversal) au lieu de Triple Barrier

    Returns:
        Dict avec predictions, meta_labels, ohlcv, etc.
    """
    suffix = '_aligned' if aligned else ''
    npz_path = Path(f'data/prepared/meta_labels_{indicator}_{filter_type}_{split}{suffix}.npz')
    # ... reste inchangé
```

**Ligne à modifier**: Ajouter `suffix` dans le path (ligne ~58)

---

### Change 1.2 - Argument Parser (ligne ~318)

**Ajouter l'argument `--aligned`:**

```python
def main():
    parser = argparse.ArgumentParser(description='Train meta-model Phase 2.17/2.18')
    parser.add_argument('--filter', type=str, default='kalman', choices=['kalman', 'octave20'],
                        help='Filter type (default: kalman)')
    parser.add_argument('--aligned', action='store_true',
                        help='Use aligned labels (signal reversal) instead of Triple Barrier')  # <-- NOUVEAU
    parser.add_argument('--output-dir', type=Path, default=Path('models/meta_model'),
                        help='Output directory for meta-model')
    args = parser.parse_args()
```

---

### Change 1.3 - Appel `load_meta_dataset` (ligne ~342)

**Passer le paramètre `aligned`:**

```python
    datasets = {}
    for split in ['train', 'val', 'test']:
        print(f"\n{split.upper()}:")
        datasets[split] = load_meta_dataset(split, indicator='macd', filter_type=args.filter, aligned=args.aligned)
        # Ajouter: aligned=args.aligned ──────────────────────────────────────────────────────────^
```

---

### Change 1.4 - Nom du Modèle Sauvegardé (ligne ~380-390)

**Ajouter suffix au nom du fichier:**

Chercher la ligne:
```python
model_path = args.output_dir / f'meta_model_baseline_{args.filter}.pkl'
```

Remplacer par:
```python
suffix = '_aligned' if args.aligned else ''
model_path = args.output_dir / f'meta_model_baseline_{args.filter}{suffix}.pkl'
```

Faire pareil pour le fichier results JSON:
```python
results_path = args.output_dir / f'meta_model_results_{args.filter}{suffix}.json'
```

---

## 📄 Modification 2: `tests/test_meta_model_backtest.py`

### Change 2.1 - Fonction `load_meta_labels_data` (ligne 95)

**Ajouter paramètre `aligned`:**

```python
def load_meta_labels_data(indicator: str, filter_type: str = 'kalman', split: str = 'test', aligned: bool = False) -> Dict:
    """
    Charge TOUTES les données depuis meta_labels_*.npz.

    Args:
        indicator: 'macd', 'rsi', ou 'cci'
        filter_type: 'kalman' ou 'octave'
        split: 'train', 'val', ou 'test'
        aligned: Si True, charge labels aligned (signal reversal)  # <-- NOUVEAU

    Returns:
        Dict avec predictions, OHLCV, meta_labels
    """
    suffix = '_aligned' if aligned else ''
    path = Path(f'data/prepared/meta_labels_{indicator}_{filter_type}_{split}{suffix}.npz')
    # ... reste inchangé
```

**Ligne à modifier**: Ajouter `suffix` dans le path (ligne ~112)

---

### Change 2.2 - Argument Parser (fonction main)

**Chercher la section du parser et ajouter:**

```python
parser.add_argument('--aligned', action='store_true',
                    help='Use aligned meta-model (signal reversal labels)')
```

---

### Change 2.3 - Chargement du Modèle

**Chercher la ligne qui charge le modèle (probablement ligne ~300-400):**

```python
model_path = Path(f'models/meta_model/meta_model_baseline_{args.filter}.pkl')
```

**Remplacer par:**

```python
suffix = '_aligned' if args.aligned else ''
model_path = Path(f'models/meta_model/meta_model_baseline_{args.filter}{suffix}.pkl')
```

---

### Change 2.4 - Chargement des Données

**Chercher l'appel à `load_meta_labels_data`:**

```python
data = load_meta_labels_data(args.indicator, args.filter, args.split)
```

**Remplacer par:**

```python
data = load_meta_labels_data(args.indicator, args.filter, args.split, aligned=args.aligned)
# Ajouter: aligned=args.aligned ────────────────────────────────────────^
```

---

## ✅ Validation Structurelle

**Après modifications, vérifier:**

1. ✅ Les deux scripts ont l'argument `--aligned`
2. ✅ Les fonctions de chargement ont le paramètre `aligned`
3. ✅ Les chemins incluent `suffix = '_aligned' if aligned else ''`
4. ✅ Le paramètre est passé dans tous les appels

---

## 🚀 Commandes d'Utilisation

### Entraînement Aligned Meta-Model

```bash
python src/train_meta_model_phase217.py --filter kalman --aligned
```

**Output:**
```
models/meta_model/meta_model_baseline_kalman_aligned.pkl
models/meta_model/meta_model_results_kalman_aligned.json
```

---

### Backtest avec Aligned Meta-Model

```bash
# Comparer toutes stratégies (baseline, 0.5, 0.6, 0.7)
python tests/test_meta_model_backtest.py \
    --indicator macd \
    --split test \
    --aligned \
    --compare-thresholds

# Tester un seul threshold
python tests/test_meta_model_backtest.py \
    --indicator macd \
    --split test \
    --aligned \
    --threshold 0.6
```

---

## 📊 Critères de Succès

### Résultats Attendus

| Stratégie | Trades | Win Rate | PnL Net | Verdict |
|-----------|--------|----------|---------|---------|
| **Baseline (no filter)** | 108,702 | 22.49% | -21,382% | Référence |
| **Aligned (0.5)** | ~76,000 | ≥25% | Meilleur | Win Rate **augmente** ✅ |
| **Aligned (0.6)** | ~40,000 | **≥35%** ✅ | **Positif?** ✅ | Win Rate **augmente** ✅ |
| **Aligned (0.7)** | ~16,000 | ≥40% | Positif | Win Rate **augmente** ✅ |

### Comparaison Triple Barrier (Ancien)

| Stratégie | Trades | Win Rate | PnL Net | Problème |
|-----------|--------|----------|---------|----------|
| **Triple Barrier (0.6)** | 40,315 | **20.34%** ❌ | -7,790% | Win Rate **BAISSE** |
| **Triple Barrier (0.7)** | 16,277 | **19.22%** ❌ | -3,034% | Win Rate **BAISSE** encore |

**Différence Clé**: Le Win Rate doit **AUGMENTER** avec aligned, pas diminuer!

---

## 🎯 Objectif Final

**Si Win Rate augmente avec filtrage aligned:**
- ✅ Mismatch résolu (labels alignés avec backtest)
- ✅ Meta-model filtre correctement les mauvais trades
- ✅ Phase 2.18 validée → Production-ready

**Si Win Rate continue de baisser:**
- ❌ Problème plus profond (modèles primaires)
- ❌ Retour aux fondamentaux (améliorer accuracy primaire)
- ❌ Autres approches (timeframe, features, etc.)

---

## 📝 Notes Importantes

1. **Les fichiers aligned existent déjà** - Pas besoin de régénérer
2. **Structure identique aux Triple Barrier** - Seuls les labels changent
3. **Modifications minimales** - 2 scripts, 4 changements par script
4. **Rétrocompatible** - Sans `--aligned`, comportement ancien préservé

---

**Créé**: 2026-01-11
**Auteur**: Claude (session hASdA)
**Référence**: CLAUDE.md Phase 2.18
