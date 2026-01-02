# Modele CNN-LSTM Multi-Output - Guide Complet

**Date**: 2026-01-02
**Statut**: Pipeline complet implemente
**Version**: 2.1

---

## IMPORTANT - Regles pour Claude

**NE PAS EXECUTER les scripts d'entrainement/evaluation.**
L'utilisateur possede les donnees reelles et un GPU. Claude doit:
1. Fournir les scripts et commandes a executer
2. Expliquer les modifications du code
3. Laisser l'utilisateur lancer les tests lui-meme

---

## IMPORTANT - Privilegier GPU

**Tous les scripts doivent utiliser le GPU quand c'est possible.**

### Regles de developpement:

1. **PyTorch pour les calculs**: Utiliser `torch.Tensor` sur GPU plutot que `numpy` pour les operations vectorisees
2. **Argument --device**: Ajouter `--device {auto,cuda,cpu}` a tous les scripts
3. **Auto-detection**: Par defaut, utiliser CUDA si disponible
4. **Kalman sur CPU**: Exception - pykalman ne supporte pas GPU, garder sur CPU
5. **Metriques sur GPU**: Concordance, correlation, comparaisons → GPU

### Pattern standard:

```python
import torch

# Global device
DEVICE = torch.device('cpu')

def main():
    global DEVICE
    if args.device == 'auto':
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        DEVICE = torch.device(args.device)

# Conversion numpy → GPU tensor
tensor = torch.tensor(numpy_array, device=DEVICE, dtype=torch.float32)

# Calcul GPU
result = (tensor1 == tensor2).float().mean().item()
```

---

## Vue d'Ensemble

Ce projet implemente un systeme de prediction de tendance crypto utilisant un modele CNN-LSTM multi-output pour predire la **pente (direction)** de 3 indicateurs techniques.

**Note**: BOL (Bollinger Bands) a ete retire car impossible a synchroniser avec les autres indicateurs (toujours lag +1).

### Objectif

Predire si chaque indicateur technique va **monter** (label=1) ou **descendre** (label=0) au prochain timestep.

**Cible de performance**: 85% accuracy

### Architecture

```
Input: (batch, 12, 3)  <- 12 timesteps x 3 indicateurs
  |
CNN 1D (64 filters)    <- Extraction features
  |
LSTM (64 hidden x 2)   <- Patterns temporels
  |
Dense partage (32)     <- Representation commune
  |
3 tetes independantes  <- RSI, CCI, MACD
  |
Output: (batch, 3)     <- 3 probabilites binaires
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

### Periodes Synchronisees (IMPORTANT)

Les indicateurs utilisent des periodes **optimisees pour la synchronisation** avec Kalman(Close):

```python
# src/constants.py - Periodes synchronisees (Lag 0)

# RSI - Synchronise avec Kalman(Close)
RSI_PERIOD = 14         # Lag 0, Concordance 82%

# CCI - Synchronise avec Kalman(Close)
CCI_PERIOD = 20         # Lag 0, Concordance 74%

# MACD - Synchronise avec Kalman(Close)
MACD_FAST = 10          # Lag 0, Concordance 70%
MACD_SLOW = 26
MACD_SIGNAL = 9

# BOL (Bollinger Bands) - RETIRE
# Impossible a synchroniser (toujours lag +1 quelque soit les parametres)
# BOL_PERIOD = 20  # DEPRECATED
```

**Pourquoi la synchronisation?**

Les indicateurs doivent etre alignes (Lag 0) avec la reference Kalman(Close) pour eviter la "pollution des gradients" pendant l'entrainement. Un indicateur desynchronise (lag +1) envoie des signaux contradictoires.

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
4. **Calcul indicateurs**: RSI, CCI, MACD (normalises 0-100) - BOL retire
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

### 3. Periodes Synchronisees des Indicateurs

```python
# constants.py - Periodes optimisees pour Lag 0
RSI_PERIOD = 14     # Synchronise
CCI_PERIOD = 20     # Synchronise
MACD_FAST = 10      # Synchronise
MACD_SLOW = 26
# BOL retire (impossible a synchroniser)
```

### 4. Labels Non-Causaux (OK)

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
| RSI | 82% | 0.82 | Lag 0, Conc 82% |
| CCI | 74% | 0.74 | Lag 0, Conc 74% |
| MACD | 70% | 0.70 | Lag 0, Conc 70% |
| **MOYENNE** | **~75%** | **~0.75** | Tous synchronises |

Note: BOL retire car toujours Lag +1 (non synchronisable).

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
- [ ] Indicateurs synchronises (RSI=14, CCI=20, MACD=10/26, Lag 0)
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
**Version**: 3.0 (3 indicateurs synchronises, BOL retire)
