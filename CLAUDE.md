# Modele CNN-LSTM Multi-Output - Guide Complet

**Date**: 2026-01-02
**Statut**: Pipeline complet implemente
**Version**: 2.0

---

## Vue d'Ensemble

Ce projet implemente un systeme de prediction de tendance crypto utilisant un modele CNN-LSTM multi-output pour predire la **pente (direction)** de 4 indicateurs techniques.

### Objectif

Predire si chaque indicateur technique va **monter** (label=1) ou **descendre** (label=0) au prochain timestep.

**Cible de performance**: 85% accuracy

### Architecture

```
Input: (batch, 12, 4)  <- 12 timesteps x 4 indicateurs
  |
CNN 1D (64 filters)    <- Extraction features
  |
LSTM (64 hidden x 2)   <- Patterns temporels
  |
Dense partage (32)     <- Representation commune
  |
4 tetes independantes  <- RSI, CCI, BOL, MACD
  |
Output: (batch, 4)     <- 4 probabilites binaires
```

---

## Quick Start

### 1. Installation

```bash
cd ~/projects/trad
pip install -r requirements.txt
```

### 2. Preparer les Donnees (une seule fois)

```bash
# Option recommandee: combiner 1min + 5min
python src/prepare_data.py --timeframe all --filter kalman

# Ou avec 5min seulement
python src/prepare_data.py --timeframe 5 --filter kalman
```

### 3. Entrainement (rapide avec donnees preparees)

```bash
python src/train.py --data data/prepared/dataset_all_kalman.npz --epochs 50
```

### 4. Evaluation

```bash
python src/evaluate.py
```

---

## Workflow Recommande

### Separation Preparation / Entrainement

Pour gagner du temps lors des tests de differentes configurations:

```bash
# 1. Preparer les donnees UNE FOIS (lent ~2-3 min)
python src/prepare_data.py --timeframe all --filter kalman

# 2. Entrainer PLUSIEURS FOIS (rapide ~10s de chargement)
python src/train.py --data data/prepared/dataset_all_kalman.npz --epochs 50
python src/train.py --data data/prepared/dataset_all_kalman.npz --lr 0.0001
python src/train.py --data data/prepared/dataset_all_kalman.npz --batch-size 64
```

### Options de Timeframe

| Option | Description | Train | Val/Test |
|--------|-------------|-------|----------|
| `--timeframe 5` | 5 minutes seulement | 5min | 5min |
| `--timeframe 1` | 1 minute seulement | 1min | 1min |
| `--timeframe all` | **Recommande** - Combine les deux | 1min + 5min | 5min only |

L'option `all` augmente les donnees d'entrainement (~3x) tout en evaluant sur le timeframe cible (5min).

---

## Configuration des Indicateurs

### Periodes Agressives (IMPORTANT)

Les indicateurs utilisent des periodes **agressives** pour capturer les mouvements rapides et reduire l'overfitting:

```python
# src/constants.py - Periodes optimisees

# RSI - Periode courte pour reactivite
RSI_PERIOD = 5          # (au lieu de 14 standard)

# CCI - Periode courte
CCI_PERIOD = 7          # (au lieu de 20 standard)

# MACD - Periodes agressives
MACD_FAST = 5           # (au lieu de 12 standard)
MACD_SLOW = 13          # (au lieu de 26 standard)
MACD_SIGNAL = 9         # (inchange)

# Bollinger Bands - Inchange
BOL_PERIOD = 20
BOL_NUM_STD = 2
```

**Pourquoi des periodes agressives?**

Les indicateurs lents (RSI 14, CCI 20, MACD 12/26) ne peuvent pas capturer la cible rapide (pente du filtre), causant de l'overfitting massif (84% train vs 65% test).

### Bibliotheque TA

Les indicateurs sont calcules avec la bibliotheque `ta` (Technical Analysis):

```python
# Installation
pip install ta

# Utilisation automatique dans indicators.py
# Plus optimise et fiable que les calculs manuels
```

---

## Structure du Projet

```
trad/
|-- src/
|   |-- constants.py           <- Toutes les constantes centralisees
|   |-- data_utils.py          <- Chargement donnees (split temporel)
|   |-- indicators.py          <- Calcul indicateurs (utilise ta lib)
|   |-- indicators_ta.py       <- Fonctions ta library
|   |-- prepare_data.py        <- Preparation et cache des datasets
|   |-- model.py               <- Modele CNN-LSTM + loss
|   |-- train.py               <- Script d'entrainement
|   |-- evaluate.py            <- Script d'evaluation
|   |-- filters.py             <- Filtres pour labels (Kalman, Decycler)
|   |-- adaptive_filters.py    <- Filtres adaptatifs (KAMA, HMA, etc.)
|   `-- adaptive_features.py   <- Features adaptatives
|
|-- data/
|   `-- prepared/              <- Datasets prepares (.npz)
|       |-- dataset_all_kalman.npz
|       `-- dataset_all_kalman_metadata.json
|
|-- models/
|   |-- best_model.pth         <- Meilleur modele
|   `-- training_history.json  <- Historique entrainement
|
|-- docs/
|   |-- SPEC_ARCHITECTURE_IA.md
|   |-- REGLE_CRITIQUE_DATA_LEAKAGE.md
|   `-- ...
|
|-- CLAUDE.md                  <- Ce fichier
`-- requirements.txt
```

---

## Donnees Disponibles

### Fichiers CSV

```
../data_trad/
|-- BTCUSD_all_1m.csv    # ~16 MB, bougies 1 minute
|-- BTCUSD_all_5m.csv    # ~9 MB, bougies 5 minutes
|-- ETHUSD_all_1m.csv    # ~15 MB, bougies 1 minute
`-- ETHUSD_all_5m.csv    # ~8 MB, bougies 5 minutes
```

### Configuration dans constants.py

```python
BTC_DATA_FILE_1M = '../data_trad/BTCUSD_all_1m.csv'
ETH_DATA_FILE_1M = '../data_trad/ETHUSD_all_1m.csv'
BTC_DATA_FILE_5M = '../data_trad/BTCUSD_all_5m.csv'
ETH_DATA_FILE_5M = '../data_trad/ETHUSD_all_5m.csv'
```

---

## Pipeline de Preparation des Donnees

### Commande

```bash
python src/prepare_data.py --timeframe all --filter kalman
```

### Processus

1. **Chargement**: BTC + ETH (1min et 5min)
2. **Trim edges**: 100 bougies debut + 100 fin
3. **Split temporel**: 70% train / 15% val / 15% test
4. **Calcul indicateurs**: RSI, CCI, BOL, MACD (normalises 0-100)
5. **Generation labels**: Filtre Kalman/Decycler (non-causal)
6. **Creation sequences**: 12 timesteps
7. **Sauvegarde**: `.npz` compresse

### Options CLI

```bash
python src/prepare_data.py --help

Options:
  --timeframe {1,5,all}   Timeframe (defaut: 5)
  --filter {decycler,kalman}  Filtre pour labels (defaut: decycler)
  --output PATH           Chemin de sortie
  --list                  Liste les datasets disponibles
```

---

## Entrainement

### Commande

```bash
# Avec donnees preparees (recommande)
python src/train.py --data data/prepared/dataset_all_kalman.npz --epochs 50

# Preparation a la volee (lent)
python src/train.py --filter kalman --epochs 50
```

### Options CLI

```bash
python src/train.py --help

Options:
  --data PATH             Donnees preparees (.npz)
  --batch-size N          Taille batch (defaut: 128)
  --lr FLOAT              Learning rate (defaut: 0.001)
  --epochs N              Nombre epoques (defaut: 100)
  --patience N            Early stopping (defaut: 10)
  --filter {decycler,kalman}  Filtre (ignore si --data)
  --device {auto,cuda,cpu}
```

---

## Points Critiques

### 1. Split Temporel (Test=fin, Val=echantillonne)

```python
# data_utils.py - Strategie optimisee pour re-entrainement mensuel

# 1. TEST = toujours a la fin (donnees les plus recentes)
test = data[-15%:]

# 2. VAL = echantillonne aleatoirement du reste (meilleure representativite)
val = remaining.sample(15%)

# 3. TRAIN = le reste
train = remaining - val
```

**Avantages:**
- Test = donnees futures (simulation realiste)
- Val echantillonne de partout → pas d'overfit a une periode specifique
- Ideal pour re-entrainement mensuel

**Durees avec donnees 5min (~160k bougies par asset):**

| Split | Ratio | Bougies | Duree | Source |
|-------|-------|---------|-------|--------|
| Train | 70% | ~112,000 | ~13 mois | Echantillonne |
| Val | 15% | ~24,000 | ~2.8 mois | Echantillonne de partout |
| Test | 15% | ~24,000 | ~2.8 mois | FIN du dataset |

### 2. Calcul Indicateurs PAR ASSET

```python
# prepare_data.py - Evite la pollution entre assets!
# CORRECT: Calculer par asset, puis merger
X_btc, Y_btc = prepare_single_asset(btc_data, filter_type)
X_eth, Y_eth = prepare_single_asset(eth_data, filter_type)
X_train = np.concatenate([X_btc, X_eth])

# INCORRECT: Merger puis calculer (pollue les indicateurs!)
# all_data = pd.concat([btc, eth])  # NON!
# indicators = calculate(all_data)   # RSI de fin BTC pollue debut ETH
```

### 3. Periodes Agressives des Indicateurs

```python
# constants.py - Periodes courtes pour eviter overfitting
RSI_PERIOD = 5      # Pas 14!
CCI_PERIOD = 7      # Pas 20!
MACD_FAST = 5       # Pas 12!
MACD_SLOW = 13      # Pas 26!
```

### 3. Labels Non-Causaux (OK)

- Labels generes avec filtre forward-backward (Kalman/Decycler)
- Utilise le futur mais c'est la **cible** a predire
- Les **features** sont toujours causales

### 4. Bibliotheque TA

- Utilise `ta` library pour les indicateurs (pas de calcul manuel)
- Plus fiable, optimise et teste

---

## Hyperparametres

### Dans constants.py

```python
# Architecture
CNN_FILTERS = 64
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.2
DENSE_HIDDEN_SIZE = 32
DENSE_DROPOUT = 0.3

# Entrainement
BATCH_SIZE = 128          # Augmente pour utiliser GPU >80%
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10

# Donnees
SEQUENCE_LENGTH = 12
```

---

## Objectifs de Performance

| Metrique | Baseline | Cible | Actuel (2026-01-02) |
|----------|----------|-------|---------------------|
| Accuracy moyenne | 50% | 85%+ | **76.4%** |
| Gap train/val | - | <10% | 3.6% ✅ |
| Gap val/test | - | <10% | 8% ✅ |

### Resultats par Indicateur (Test Set)

| Indicateur | Accuracy | F1 | Notes |
|------------|----------|-----|-------|
| MACD | 79.5% | 0.795 | Meilleur |
| CCI | 77.9% | 0.782 | |
| BOL | 74.6% | 0.746 | |
| RSI | 73.8% | 0.739 | Plus difficile |
| **MOYENNE** | **76.4%** | **0.765** | +26.4pts vs baseline |

### Configuration Optimale Actuelle

```bash
python src/train.py --data data/prepared/dataset_all_kalman.npz \
    --cnn-filters 128 --lstm-hidden 128 --lstm-layers 3 \
    --dense-hidden 64 --lstm-dropout 0.3 --dense-dropout 0.4
```

### Signes de bon entrainement

- Val loss suit train loss
- Gap train/val <= 10%
- Accuracy > 60% des l'epoque 1

### Signes de probleme

- Val loss monte pendant que train loss descend -> Overfitting
- Accuracy stagne a ~50% -> Modele n'apprend pas
- Gap train/test > 15% -> Indicateurs trop lents

---

## Commandes Utiles

```bash
# Lister les datasets prepares
python src/prepare_data.py --list

# Preparer avec 1min + 5min
python src/prepare_data.py --timeframe all --filter kalman

# Entrainer
python src/train.py --data data/prepared/dataset_all_kalman.npz

# Evaluer
python src/evaluate.py

# Verifier constantes
python src/constants.py
```

---

## Checklist Avant Production

- [ ] Accuracy >= 85% sur test set
- [ ] Gap train/test <= 10%
- [ ] Periodes indicateurs agressives (RSI=5, CCI=7, MACD=5/13)
- [ ] Split temporel strict
- [ ] Bibliotheque ta utilisee
- [ ] Backtest sur donnees non vues
- [ ] Trading strategy definie

---

## Pistes d'Amelioration (Litterature)

### 1. Features Additionnelles (Priorite Haute)

**Volume et Derivees:**
- Volume brut normalise
- Volume relatif (vs moyenne mobile)
- OBV (On-Balance Volume)
- Volume-Price Trend (VPT)

**Volatilite:**
- ATR (Average True Range)
- Volatilite historique (std des returns)
- Largeur des bandes de Bollinger

**Momentum additionnels:**
- ROC (Rate of Change) sur plusieurs periodes
- Williams %R
- Stochastic Oscillator

### 2. Features Multi-Resolution (Litterature: "Multi-Scale Features")

Encoder l'information a plusieurs echelles temporelles:
```
Features actuelles: indicateurs sur 5min
Ajouter: memes indicateurs sur 15min, 1h, 4h
```

Cela capture les tendances court/moyen/long terme simultanement.

### 3. Features de Marche (Cross-Asset)

- Correlation BTC/ETH glissante
- Dominance BTC (si donnees disponibles)
- Spread BTC-ETH

### 4. Embeddings Temporels

- Heure du jour (sin/cos encoding)
- Jour de la semaine (sin/cos encoding)
- Session de trading (Asie/Europe/US)

### 5. Features Derivees des Prix

- Returns logarithmiques
- Returns sur plusieurs horizons (1, 5, 15, 60 periodes)
- High-Low range normalise
- Close position dans la bougie (close-low)/(high-low)

### References

- "Deep Learning for Financial Time Series" - recommande multi-scale features
- "Attention-based Models for Crypto" - importance du volume
- "Technical Analysis with ML" - combinaison indicateurs + prix bruts

### Prochaines Etapes Recommandees

1. **Court terme**: Ajouter Volume + ATR (2 features, impact potentiel eleve)
2. **Moyen terme**: Multi-resolution (indicateurs 15min/1h)
3. **Long terme**: Embeddings temporels + cross-asset

---

**Cree par**: Claude Code
**Derniere MAJ**: 2026-01-02
**Version**: 2.1
