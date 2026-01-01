# 🤖 Modèle CNN-LSTM Multi-Output - Guide Complet

**Date**: 2026-01-01
**Statut**: Pipeline complet implémenté ✅

---

## 📋 Vue d'Ensemble

Ce projet implémente un système de prédiction de tendance crypto utilisant un modèle CNN-LSTM multi-output pour prédire la **pente (direction)** de 4 indicateurs techniques.

### Objectif

Prédire si chaque indicateur technique va **monter** (label=1) ou **descendre** (label=0) au prochain timestep.

### Architecture

```
Input: (batch, 12, 4)  ← 12 timesteps × 4 indicateurs
  ↓
CNN 1D (64 filters)    ← Extraction features
  ↓
LSTM (64 hidden × 2)   ← Patterns temporels
  ↓
Dense partagé (32)     ← Représentation commune
  ↓
4 têtes indépendantes  ← RSI, CCI, BOL, MACD
  ↓
Output: (batch, 4)     ← 4 probabilités binaires
```

---

## 🚀 Quick Start

### 1. Installation

```bash
cd ~/projects/trad
pip install -r requirements.txt
```

### 2. Vérifier les Données

```bash
python src/data_utils.py
```

**Attendu**: 199,600 bougies chargées (BTC 99,800 + ETH 99,800)

### 3. Test Pipeline Indicateurs

```bash
python src/indicators.py
```

**Attendu**: Datasets prêts avec shapes:
- Train: X=(139,708, 12, 4), Y=(139,708, 4)
- Val: X=(29,928, 12, 4), Y=(29,928, 4)
- Test: X=(29,928, 12, 4), Y=(29,928, 4)

### 4. Test Modèle

```bash
python src/model.py
```

**Attendu**: Forward pass OK, métriques calculées

### 5. Entraînement

```bash
python src/train.py
```

**Durée estimée**: 10-30 min (dépend CPU/GPU)

### 6. Évaluation

```bash
python src/evaluate.py
```

**Attendu**: Métriques sur test set + comparaison baseline

---

## 📁 Structure du Projet

```
trad/
├── src/
│   ├── constants.py           ← Toutes les constantes centralisées
│   ├── data_utils.py          ← Chargement données (split temporel)
│   ├── indicators.py          ← Calcul indicateurs + labels
│   ├── model.py               ← Modèle CNN-LSTM + loss
│   ├── train.py               ← Script d'entraînement
│   └── evaluate.py            ← Script d'évaluation
│
├── docs/
│   ├── SPEC_ARCHITECTURE_IA.md       ← Spécification complète
│   ├── APPROCHE_IA_PREDICTION_PENTE.md  ← Approche IA (prédire pente)
│   ├── REGLE_CRITIQUE_DATA_LEAKAGE.md   ← Split temporel obligatoire
│   └── RESULTATS_DECYCLER_INDICATEURS.md ← Tests monde parfait
│
├── models/                    ← Modèles sauvegardés
│   ├── best_model.pth         ← Meilleur modèle
│   └── training_history.json  ← Historique entraînement
│
├── results/                   ← Résultats évaluation
│   └── test_results.json      ← Métriques test set
│
├── GUIDE_TEST_DONNEES.md      ← Guide test chargement données
└── CLAUDE.md                  ← Ce fichier
```

---

## 🎯 Pipeline Complet

### Étape 1: Chargement Données

```python
from data_utils import load_and_split_btc_eth

train_df, val_df, test_df = load_and_split_btc_eth()
```

**Caractéristiques**:
- BTC: 100k bougies (les dernières)
- ETH: 100k bougies (les dernières)
- Trim edges: 100 début + 100 fin (warm-up filtres)
- **Split temporel STRICT**: 70% train / 15% val / 15% test
- **Pas de shuffle global** (évite data leakage)

### Étape 2: Calcul Indicateurs

```python
from indicators import prepare_datasets

datasets = prepare_datasets(train_df, val_df, test_df)
X_train, Y_train = datasets['train']
```

**Indicateurs normalisés (0-100)**:
1. RSI(14) - Déjà 0-100
2. CCI(20) - Normalisé depuis -200/+200
3. Bollinger %B(20, 2σ) - Position dans bandes
4. MACD(12/26/9) - Histogram normalisé dynamiquement

**Labels**:
- Générés avec **Decycler parfait** (forward-backward, non-causal)
- Label = 1 si filtre[t-1] > filtre[t-2] (pente haussière)
- Label = 0 sinon (pente baissière)

**Séquences**:
- Longueur: 12 timesteps
- Format: X=(N, 12, 4), Y=(N, 4)

### Étape 3: Entraînement

```python
from train import main

main()
```

**Hyperparamètres** (voir `constants.py`):
- Batch size: 32
- Learning rate: 0.001
- Epochs: 100 (max)
- Early stopping: 10 patience
- Optimizer: Adam

**Loss**:
- BCE multi-output
- Moyenne pondérée des 4 sorties (poids égaux par défaut)

**Early Stopping**:
- Surveille validation loss
- Arrête si pas d'amélioration pendant 10 époques
- Sauvegarde le meilleur modèle

### Étape 4: Évaluation

```python
from evaluate import main

main()
```

**Métriques calculées**:
- Par indicateur: Accuracy, Precision, Recall, F1
- Moyenne des 4 indicateurs
- **Vote majoritaire**: Moyenne des 4 prédictions

---

## 📊 Résultats Attendus

### Baseline (Hasard)

- Accuracy: ~50%
- F1: ~50%

### Objectif

- **Accuracy moyenne: ≥70%**
- F1 moyen: ≥70%
- Vote majoritaire: ≥70%

### Interprétation

Si accuracy ~50% :
- ⚠️ Le modèle n'apprend pas (équivalent hasard)
- Vérifier: data leakage, labels, architecture

Si accuracy 60-70% :
- ✅ Le modèle apprend des patterns
- Améliorer: hyperparamètres, plus de données

Si accuracy ≥70% :
- 🎯 Objectif atteint !
- Prochaine étape: Backtest réel

---

## ⚠️ Points Critiques

### 1. Data Leakage - ÉVITÉ ✅

**Problème potentiel**: Shuffle avant split
- Séquences t et t+1 dans train ET test
- Accuracy artificielle 90%+ mais 50% en prod

**Solution implémentée**:
- **Split temporel STRICT** dans `data_utils.py`
- Train = 70% premiers
- Val = 15% suivants
- Test = 15% derniers
- Shuffle APRÈS split (uniquement train)

### 2. Labels Non-Causaux - CORRECT ✅

**Approche**:
- Labels générés avec **Decycler parfait** (forward-backward)
- NON-CAUSAL (utilise le futur) mais OK car ce sont des **labels**
- Les **features** (indicateurs) sont CAUSALES

**Règle**:
- Input X: TOUJOURS causal (n'utilise que le passé)
- Labels Y: Peuvent être non-causaux (vérité terrain)

### 3. Normalisation - CORRECT ✅

**Principe**:
- Tous les indicateurs normalisés 0-100
- Facilite apprentissage du réseau
- Évite domination d'un indicateur

**Implémentation**:
- RSI: Déjà 0-100
- CCI: Min-max -200/+200 → 0-100
- Bollinger %B: 0-100
- MACD: Normalisation dynamique (rolling window)

---

## 🔧 Ajuster les Hyperparamètres

Tous dans `src/constants.py` :

### Architecture

```python
# CNN
CNN_FILTERS = 64          # Nombre de filtres (essayer 32, 64, 128)
CNN_KERNEL_SIZE = 3       # Taille kernel (essayer 3, 5)

# LSTM
LSTM_HIDDEN_SIZE = 64     # Taille hidden (essayer 32, 64, 128)
LSTM_NUM_LAYERS = 2       # Nombre de couches (essayer 1, 2, 3)
LSTM_DROPOUT = 0.2        # Dropout LSTM (essayer 0.1, 0.2, 0.3)

# Dense
DENSE_HIDDEN_SIZE = 32    # Taille couche dense (essayer 16, 32, 64)
DENSE_DROPOUT = 0.3       # Dropout dense (essayer 0.2, 0.3, 0.4)
```

### Entraînement

```python
BATCH_SIZE = 32           # Batch size (essayer 16, 32, 64)
LEARNING_RATE = 0.001     # Learning rate (essayer 0.0001, 0.001, 0.01)
NUM_EPOCHS = 100          # Époques max (essayer 50, 100, 200)
EARLY_STOPPING_PATIENCE = 10  # Patience (essayer 5, 10, 20)
```

### Données

```python
SEQUENCE_LENGTH = 12      # Longueur séquences (essayer 6, 12, 24)
BTC_CANDLES = 100000      # Bougies BTC (essayer 50k, 100k, 200k)
ETH_CANDLES = 100000      # Bougies ETH (essayer 50k, 100k, 200k)
```

---

## 📈 Monitoring

### Pendant l'entraînement

Observer dans les logs:
- **Train loss**: Doit descendre progressivement
- **Val loss**: Doit descendre aussi (si monte → overfitting)
- **Train accuracy**: Doit monter
- **Val accuracy**: Doit monter et rester proche de train

**Signes de bon entraînement**:
- Val loss suit train loss
- Gap train/val ≤ 5%
- Accuracy > 50% (sinon = hasard)

**Signes de problème**:
- Val loss monte pendant que train loss descend → Overfitting
- Accuracy stagne à ~50% → Modèle n'apprend pas
- Loss explose → Learning rate trop élevé

### Après entraînement

Fichiers générés:
- `models/best_model.pth` - Meilleur modèle
- `models/training_history.json` - Historique complet
- `results/test_results.json` - Métriques test

Visualiser:
```python
import json
import matplotlib.pyplot as plt

with open('models/training_history.json') as f:
    history = json.load(f)

plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Val')
plt.legend()
plt.show()
```

---

## 🎯 Prochaines Étapes

### Si accuracy ≥70% atteinte :

1. **Backtest réel** sur données de production
2. **Trading strategy** basée sur prédictions
3. **Monitoring live** en conditions réelles

### Si accuracy <70% :

1. Augmenter `NUM_EPOCHS` (essayer 200)
2. Ajuster architecture (plus de CNN_FILTERS/LSTM_HIDDEN_SIZE)
3. Augmenter données (plus de BTC_CANDLES/ETH_CANDLES)
4. Vérifier qualité des labels (distribution ~50/50)

---

## 📚 Documentation Technique

### Fichiers de documentation

- `docs/SPEC_ARCHITECTURE_IA.md` - Spécification complète du modèle
- `docs/APPROCHE_IA_PREDICTION_PENTE.md` - Pourquoi prédire la pente
- `docs/REGLE_CRITIQUE_DATA_LEAKAGE.md` - Data leakage et split temporel
- `docs/RESULTATS_DECYCLER_INDICATEURS.md` - Validation théorique

### Concepts clés

**Decycler Parfait**:
- Filtre de Ehlers appliqué forward puis backward
- Résultat: Signal lissé SANS lag temporel
- Utilisation: Génération labels uniquement (non-causal OK)

**Split Temporel**:
- Train sur passé, valide sur futur
- Simule conditions réelles de trading
- Évite data leakage massif

**Multi-Output**:
- 4 sorties indépendantes (une par indicateur)
- Chaque sortie prédit pente de son indicateur
- Vote majoritaire pour décision finale

---

## ✅ Checklist Avant Production

- [ ] Accuracy ≥70% sur test set
- [ ] Gap train/val ≤5%
- [ ] Vote majoritaire ≥70%
- [ ] Pas de data leakage (validation timestamps OK)
- [ ] Backtest sur données non vues
- [ ] Trading strategy définie
- [ ] Risk management implémenté

---

**Créé par**: Claude Code
**Dernière MAJ**: 2026-01-01
**Version**: 1.0
