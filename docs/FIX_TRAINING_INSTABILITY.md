# Fix Training Instability - Guide Complet

**Date**: 2026-01-12
**Problème identifié**: Le modèle oscille entre prédire tout 0 et tout 1

## Symptômes d'Instabilité d'Entraînement

```
Époque 1: Prédit 100% classe 1
Époque 2: Prédit 100% classe 0
Époque 3: Prédit 100% classe 1

Loss bloquée à ~0.693 (= ln(2) pour classification binaire aléatoire)
Accuracy ~50% (aléatoire)
F1 → 0 (modèle prédit une seule classe)
```

## Diagnostic

Le diagnostic a révélé:
```
[DEBUG] Prédictions: 0=0 (0.0%), 1=608460 (100.0%)
[DEBUG] Targets:     0=304573 (50.1%), 1=303887 (49.9%)
```

✅ **Les targets sont équilibrés** (50.1% / 49.9%) → **Les données sont OK**
❌ **Le modèle oscille** entre extrêmes → **Problème d'entraînement**

## Causes Possibles

1. **Learning rate trop élevé** → Gradients explosent
2. **Features non normalisées** → Valeurs explosives
3. **Gradient explosion** → Poids divergent

## Solutions Implémentées

### 1. Gradient Clipping ✅ (Commit: [hash])

**Ajouté dans `train.py`:**
```python
# Après loss.backward()
if grad_clip is not None:
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
```

**Utilisation:**
```bash
# Avec gradient clipping (valeur par défaut: 1.0)
python src/train.py --data <dataset.npz> --grad-clip 1.0

# Plus agressif (si oscillations persistent)
python src/train.py --data <dataset.npz> --grad-clip 0.5

# Désactiver (si modèle stable)
python src/train.py --data <dataset.npz> --grad-clip 0
```

### 2. Learning Rate Réduit ✅ (Commit: [hash])

**Modifié dans `constants.py`:**
```python
# AVANT
LEARNING_RATE = 0.001

# APRÈS
LEARNING_RATE = 0.0001  # Divisé par 10
```

**Utilisation:**
```bash
# Utiliser le nouveau défaut (0.0001)
python src/train.py --data <dataset.npz>

# Encore plus bas (si oscillations persistent)
python src/train.py --data <dataset.npz> --lr 0.00001

# Plus agressif (si convergence trop lente)
python src/train.py --data <dataset.npz> --lr 0.0005
```

## Vérifier la Normalisation des Features

**Script de diagnostic créé:** `tests/check_features_normalization.py`

```bash
# Vérifier si les features sont normalisées
python tests/check_features_normalization.py \
    --data data/prepared/dataset_btc_eth_bnb_ada_ltc_universal_kalman.npz
```

**Ce que le script vérifie:**
- Min/Max/Mean/Std par feature
- Détection de valeurs > 10 ou < -10 (non normalisées)
- NaN/Inf dans les données

**Si features NON normalisées:**
→ Régénérer le dataset avec `prepare_data_universal.py`
→ S'assurer que la normalisation est appliquée

## Configuration Recommandée (Pour MACD Direction)

### Configuration Conservatrice (Recommandée pour Debugging)

```bash
python src/train.py \
    --data data/prepared/dataset_btc_eth_bnb_ada_ltc_universal_kalman.npz \
    --indicator macd \
    --epochs 10 \
    --lr 0.00005 \
    --grad-clip 0.5 \
    --batch-size 64 \
    --patience 5
```

**Pourquoi cette config:**
- `--lr 0.00005`: Learning rate très bas pour stabilité maximale
- `--grad-clip 0.5`: Clipping agressif pour éviter explosion
- `--batch-size 64`: Plus petit batch = gradients plus stables
- `--epochs 10`: Court pour debug rapide
- `--patience 5`: Early stopping rapide si convergence

### Configuration Standard (Après Stabilisation)

```bash
python src/train.py \
    --data data/prepared/dataset_btc_eth_bnb_ada_ltc_universal_kalman.npz \
    --indicator macd \
    --epochs 50 \
    --lr 0.0001 \
    --grad-clip 1.0 \
    --batch-size 128
```

### Configuration Agressive (Si Stable)

```bash
python src/train.py \
    --data data/prepared/dataset_btc_eth_bnb_ada_ltc_universal_kalman.npz \
    --indicator macd \
    --epochs 50 \
    --lr 0.0005 \
    --grad-clip 2.0 \
    --batch-size 256
```

## Monitoring Pendant l'Entraînement

Le diagnostic inline affiche maintenant:

```
[DEBUG] Prédictions: 0=X (Y%), 1=Z (W%)
[DEBUG] Targets:     0=A (B%), 1=C (D%)
```

**Signes de bonne convergence:**
- Prédictions se rapprochent de 50/50 (équilibre)
- Loss descend progressivement
- Accuracy monte progressivement

**Signes d'instabilité:**
- Prédictions oscillent entre 100%/0%
- Loss reste à ~0.693
- Accuracy bloquée à ~50%

## Prochaines Actions si Instabilité Persiste

### Action 1: Vérifier Features
```bash
python tests/check_features_normalization.py --data <dataset.npz>
```

**Si features > 10 ou < -10:**
→ Régénérer dataset avec normalisation

### Action 2: Réduire Encore Learning Rate
```bash
python src/train.py --lr 0.00001 --grad-clip 0.5
```

### Action 3: Ajouter LayerNorm (Si LR très bas ne suffit pas)

Modifier `model.py` pour ajouter LayerNorm après CNN:
```python
self.layer_norm = nn.LayerNorm(cnn_output_size)

# Dans forward():
x = self.layer_norm(x)
```

### Action 4: Utiliser un Scheduler (Optionnel)

Ajouter dans `train.py` après optimizer:
```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3,
    verbose=True
)

# Dans la boucle d'entraînement:
scheduler.step(val_metrics['loss'])
```

## Résumé des Modifications Appliquées

| Fichier | Modification | Commit |
|---------|--------------|--------|
| `src/train.py` | Ajout gradient clipping | [hash] |
| `src/train.py` | Ajout diagnostic inline | 054d6bb |
| `src/train.py` | Fix metadata JSON error | [hash] |
| `src/constants.py` | Learning rate 0.001 → 0.0001 | [hash] |
| `tests/check_features_normalization.py` | Nouveau script diagnostic | [hash] |

## Commandes de Test Rapide

```bash
# 1. Vérifier features
python tests/check_features_normalization.py \
    --data data/prepared/dataset_btc_eth_bnb_ada_ltc_universal_kalman.npz

# 2. Entraînement court avec config conservatrice
python src/train.py \
    --data data/prepared/dataset_btc_eth_bnb_ada_ltc_universal_kalman.npz \
    --indicator macd \
    --epochs 3 \
    --lr 0.00005 \
    --grad-clip 0.5

# 3. Observer les [DEBUG] logs
# → Si prédictions équilibrées: ✅ Problème résolu
# → Si oscillations persistent: Vérifier normalisation features
```

## Références

- **Gradient Clipping**: [PyTorch docs - clip_grad_norm_](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)
- **Training Instability**: [CS231n - Babysitting the Learning Process](http://cs231n.github.io/neural-networks-3/)
- **Learning Rate Tuning**: [FastAI - Finding Good Learning Rates](https://docs.fast.ai/callback.schedule.html#learningratefinder)
