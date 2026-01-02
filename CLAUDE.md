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
  --batch-size N          Taille batch (defaut: 32)
  --lr FLOAT              Learning rate (defaut: 0.001)
  --epochs N              Nombre epoques (defaut: 100)
  --patience N            Early stopping (defaut: 10)
  --filter {decycler,kalman}  Filtre (ignore si --data)
  --device {auto,cuda,cpu}
```

---

## Points Critiques

### 1. Split Temporel STRICT

```python
# data_utils.py - JAMAIS de shuffle avant split!
train = data[0:70%]      # Passe
val   = data[70%:85%]    # Present
test  = data[85%:100%]   # Futur
```

### 2. Periodes Agressives des Indicateurs

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
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10

# Donnees
SEQUENCE_LENGTH = 12
```

---

## Objectifs de Performance

| Metrique | Baseline | Cible |
|----------|----------|-------|
| Accuracy moyenne | 50% | 85%+ |
| Gap train/val | - | <10% |
| Gap train/test | - | <10% |

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

**Cree par**: Claude Code
**Derniere MAJ**: 2026-01-02
**Version**: 2.0
