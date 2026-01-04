# Modele CNN-LSTM Multi-Output - Guide Complet

**Date**: 2026-01-04
**Statut**: Pipeline complet implemente - Objectif 85% ATTEINT + Approche OHLC
**Version**: 4.6

---

## DECOUVERTE IMPORTANTE - Retrait de BOL (Bollinger Bands)

### Probleme identifie

L'indicateur **BOL (Bollinger Bands %B)** a ete **retire** du modele car il est **impossible a synchroniser** avec la reference Kalman(Close).

### Analyse de synchronisation

| Indicateur | Periode testee | Lag optimal | Concordance | Status |
|------------|---------------|-------------|-------------|--------|
| RSI | 14 | **0** | 82% | ✅ Synchronise |
| CCI | 20 | **0** | 74% | ✅ Synchronise |
| MACD | 10/26/9 | **0** | 70% | ✅ Synchronise |
| BOL | 5-50 (toutes) | **+1** | ~65% | ❌ Non synchronisable |

### Pourquoi BOL ne peut pas etre synchronise?

1. **Nature de l'indicateur**: BOL %B mesure la position du prix par rapport aux bandes
2. **Calcul des bandes**: Utilise une moyenne mobile + ecart-type (retard inherent)
3. **Toutes les periodes testees** (5, 10, 15, 20, 25, 30, 40, 50) donnent Lag +1
4. **Pollution des gradients**: Un indicateur avec Lag +1 envoie des signaux contradictoires

### Impact sur le modele

- **Avant**: 4 indicateurs (RSI, CCI, BOL, MACD) → 4 sorties
- **Apres**: 3 indicateurs (RSI, CCI, MACD) → 3 sorties
- **Benefice**: Gradients plus propres, meilleure convergence

### Conclusion

BOL est structurellement incompatible avec notre approche de synchronisation. Les 3 indicateurs restants (RSI, CCI, MACD) sont tous synchronises (Lag 0) et offrent une base solide pour la prediction.

---

## RESULTAT MAJEUR - Architecture Clock-Injected (85.1%)

### Comparaison des Approches (2026-01-03)

| Approche | RSI | CCI | MACD | **MOYENNE** | Delta |
|----------|-----|-----|------|-------------|-------|
| Baseline 5min (3 feat) | 79.4% | 83.7% | 86.9% | **83.3%** | - |
| Position Index (4 feat) | 79.4% | 83.7% | 87.0% | **83.4%** | +0.1% |
| **Clock-Injected (7 feat)** | **83.0%** | **85.6%** | **86.8%** | **85.1%** | **+1.8%** |

### Analyse des Gains

**RSI = Grand Gagnant (+3.6%)**
- En tant qu'oscillateur de vitesse pure, le RSI 5min est tres nerveux
- L'injection des indicateurs 30min sert de "Laisse de Securite"
- Le modele a appris a ignorer les surachats/surventes 5min si le RSI 30min ne confirme pas encore le pivot

**MACD (Stable a 86.8%)**
- Deja un indicateur de tendance "lourd"
- L'ajout de sa version 30min n'apporte pas d'information radicalement nouvelle
- Reste le pilier de stabilite du modele

**Position Index vs Step Index**
- Position Index (constant): +0.1% → **ECHEC** (LSTM encode deja l'ordre)
- Step Index (variable selon timestamp): +1.8% → **SUCCES** (information nouvelle)

### Commandes Clock-Injected

```bash
# Preparer (7 features)
python src/prepare_data_30min.py --filter kalman --assets BTC ETH BNB ADA LTC --include-30min-features

# Entrainer
python src/train.py --data data/prepared/dataset_btc_eth_bnb_ada_ltc_5min_30min_labels30min_kalman.npz --epochs 50

# Evaluer
python src/evaluate.py --data data/prepared/dataset_btc_eth_bnb_ada_ltc_5min_30min_labels30min_kalman.npz
```

### Structure des 7 Features

```
| RSI_5min | CCI_5min | MACD_5min | RSI_30min | CCI_30min | MACD_30min | StepIdx |
|  0-100   |  0-100   |   0-100   |   0-100   |   0-100   |   0-100    |   0-1   |
| reactif  | reactif  |  reactif  |  stable   |  stable   |   stable   | horloge |
```

Le **Step Index** (0.0 → 1.0) indique la position dans la fenetre 30min:
- Step 1 (0.0): Debut de bougie 30min → plus de poids sur 5min
- Step 6 (1.0): Fin de bougie 30min → confirmation fiable

---

## NOUVELLE APPROCHE - Features OHLC (2026-01-04)

### Contexte

Approche alternative utilisant les donnees OHLC brutes normalisees au lieu des indicateurs techniques (RSI, CCI, MACD).

### Pipeline prepare_data_ohlc_v2.py

```
ETAPE 1: Chargement avec DatetimeIndex
ETAPE 2: Calcul indicateurs (si besoin pour target)
ETAPE 3: Calcul features OHLC normalisees
ETAPE 4: Calcul filtre + labels
ETAPE 5: TRIM edges (100 debut + 100 fin)
ETAPE 6: Creation sequences avec verification index
```

### Features OHLC (5 canaux)

| Feature | Formule | Role |
|---------|---------|------|
| **O_ret** | (Open[t] - Close[t-1]) / Close[t-1] | Gap d'ouverture (micro-structure) |
| **H_ret** | (High[t] - Close[t-1]) / Close[t-1] | Extension haussiere intra-bougie |
| **L_ret** | (Low[t] - Close[t-1]) / Close[t-1] | Extension baissiere intra-bougie |
| **C_ret** | (Close[t] - Close[t-1]) / Close[t-1] | Rendement net (patterns principaux) |
| **Range_ret** | (High[t] - Low[t]) / Close[t-1] | Volatilite intra-bougie |

### Notes de l'Expert (IMPORTANT)

**1. C_ret vs Micro-structure**
- **C_ret** encode les patterns **cloture-a-cloture** → le "gros" du signal appris par CNN
- **O_ret, H_ret, L_ret** capturent la **micro-structure intra-bougie**
- **Range_ret** capture l'**activite/volatilite** du marche

**2. Definition du Label (MISE A JOUR 2026-01-04)**
```
label[i] = 1 si filtered[i-2] > filtered[i-3] (pente PASSEE, decalee)
```
- **Decalage d'un pas** par rapport a la formule initiale `f[i-1] > f[i-2]`
- Raison: Reduire la correlation avec filtfilt (filtre non-causal)
- Le modele **re-estime l'etat PASSE** du marche, pas le futur
- La valeur vient de la **DYNAMIQUE des predictions** (changements d'avis)

**3. Convention Timestamp OHLC**
```
Timestamp = Open time (debut de la bougie)

Exemple bougie 5min timestampee "10:05":
- Open  = premier prix a 10:05:00
- High  = prix max entre 10:05:00 et 10:09:59
- Low   = prix min entre 10:05:00 et 10:09:59
- Close = dernier prix a ~10:09:59

→ Close[10:05] est disponible APRES 10:10:00
→ Donc causal si utilise a partir de l'index suivant
```

**4. Alignement Features/Labels**
```python
# Pour chaque sequence i:
X[i] = features[i-12:i]  # indices i-12 a i-1 (12 elements)
Y[i] = labels[i]          # label a l'index i

# Relation temporelle:
# - Derniere feature: index i-1 (Close[i-1] disponible)
# - Label: filtered[i-2] > filtered[i-3] (pente passee, decalee)
# → Pas de data leakage (decalage supplementaire vs filtfilt)
```

### Commandes OHLC

```bash
# Preparer (5 features OHLC)
python src/prepare_data_ohlc_v2.py --target close --assets BTC ETH BNB ADA LTC

# Entrainer
python src/train.py --data data/prepared/dataset_btc_eth_bnb_ada_ltc_ohlcv2_close_octave20.npz --indicator close

# Evaluer
python src/evaluate.py --data data/prepared/dataset_btc_eth_bnb_ada_ltc_ohlcv2_close_octave20.npz --indicator close
```

### Resultats OHLC (2026-01-04)

#### Impact du decalage de label (filtfilt correlation fix)

| Formule Label | Accuracy RSI | Notes |
|---------------|--------------|-------|
| `f[i-1] > f[i-2]` (ancienne) | 76.6% | Modele "trichait" via filtfilt |
| `f[i-1] > f[i-3]` (delta=1) | 79.7% | Amelioration partielle |
| **`f[i-2] > f[i-3]`** (nouvelle) | **83.3%** | Formule finale, honnete |

**Conclusion**: Le decalage d'un pas supplementaire (de i-1 a i-2) elimine la correlation residuelle avec le filtre non-causal.

#### Resultats par target

| Target | Features | Accuracy | Notes |
|--------|----------|----------|-------|
| **RSI** | OHLC 5ch | **83.3%** | Avec formule corrigee |
| MACD | OHLC 5ch | 84.3% | Indicateur de tendance lourde |
| CLOSE | OHLC 5ch | 78.1% | Plus volatil, plus difficile |

### Backtest Oracle (Labels Parfaits)

Resultats sur 20000 samples (~69 jours) en mode Oracle:

| Metrique | Valeur |
|----------|--------|
| **Rendement strategie** | **+1628%** |
| Rendement Buy & Hold | +45% |
| **Surperformance** | **+1584%** |
| Win Rate | 78.4% |
| Total trades | 2543 |
| Rendement moyen/trade | +0.640% |
| Duree moyenne trade | 8 periodes (~40 min) |
| Max Drawdown | -2.78% |
| LONG (1272 trades) | +837% |
| SHORT (1271 trades) | +792% |

**Note**: Calcul en rendement simple (somme), pas compose.

### Objectif Realiste

Meme a **5% du gain Oracle**, on obtient:
- Rendement: **+81%** sur 69 jours
- Surperformance vs B&H: **+36%**

### Interpretation Strategique

Le modele ne "predit pas le futur" mais **re-estime le passe** de maniere robuste:
- A chaque instant, il estime si la pente filtree entre t-3 et t-2 etait positive
- L'interet n'est pas l'accuracy brute, mais les **changements d'avis**
- Un changement d'avis indique que les features recentes contredisent la tendance passee → signal de retournement

---

## BACKTEST REEL - Resultats et Diagnostic (2026-01-04)

### Bug Corrige: Double Sigmoid

**Probleme identifie**: Le modele applique sigmoid dans `forward()` (model.py:201), mais les scripts de backtest et train appliquaient sigmoid une deuxieme fois.

**Impact**: Toutes les predictions etaient ecrasees vers 0.5 → 100% LONG apres seuil.

**Fichiers corriges**:
- `tests/test_trading_strategy_ohlc.py` - fonction `load_model_predictions()`
- `src/train.py` - fonction `generate_predictions()`

```python
# AVANT (bug)
preds = (torch.sigmoid(outputs) > 0.5)  # Double sigmoid!

# APRES (corrige)
preds = (outputs > 0.5)  # outputs deja en [0,1]
```

### Resultats Backtest Reels

| Mode | Split | Inversé | Rendement | Win Rate | Trades |
|------|-------|---------|-----------|----------|--------|
| Oracle | Train | Non | **+1042%** | 67.9% | ~800 |
| Model | Train | Non | -754% | 27.7% | ~2500 |
| Model | Train | Oui | +739% | 70.0% | ~2500 |
| Model | Test | Oui | **-1.57%** | 61.7% | ~500 |

**Note**: L'inversion des signaux sur train (+739%) etait de l'overfitting pur - ne generalise pas sur test.

### Diagnostic: Probleme de Micro-Sorties

Le modele predit bien les tendances (accuracy 83%), mais :

1. **Trop de trades**: ~2500 sur train vs ~800 pour Oracle (3x plus)
2. **Micro-sorties**: Le modele change d'avis en pleine tendance
3. **Duree moyenne**: ~1h par trade (vs ~40min Oracle, mais trop de trades)

**Cause racine**: Le modele "flicke" entre 0 et 1 meme quand la tendance globale est correcte. Ces micro-sorties generent des entrees/sorties inutiles qui mangent les profits.

### Solutions a Implementer

| # | Solution | Description | Statut |
|---|----------|-------------|--------|
| 1 | **Hysteresis** | Seuil asymetrique: entrer si P > 0.6, sortir si P < 0.4 | A tester |
| 2 | **Confirmation N periodes** | Attendre signal stable 2-3 periodes avant changement | A tester |
| 3 | **Lissage probabilites** | Moyenne mobile sur outputs avant seuillage | A tester |
| 4 | **Filtre anti-flicker** | Ignorer changements < 5 periodes apres dernier trade | A tester |

### Prochaine Etape

Implementer un filtre de stabilite sur les signaux dans `test_trading_strategy_ohlc.py` pour reduire les micro-sorties et evaluer l'impact sur le rendement.

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

### 2. Preparer les Donnees (5min)

```bash
# COMMANDE PRINCIPALE: 5 assets, donnees 5min
python src/prepare_data.py --filter kalman --assets BTC ETH BNB ADA LTC
```

**Architecture:**
- **Features**: 3 indicateurs (RSI, CCI, MACD) normalises 0-100
- **Labels**: Pente des indicateurs (filtre Kalman)
- **Sequences**: 12 timesteps

### 3. Entrainement

```bash
python src/train.py --data data/prepared/dataset_btc_eth_bnb_ada_ltc_5min_kalman.npz --epochs 50
```

### 4. Evaluation

```bash
python src/evaluate.py
```

---

## Workflow Recommande

### Workflow 5min

```bash
# 1. Preparer les donnees UNE FOIS avec tous les assets
python src/prepare_data.py --filter kalman --assets BTC ETH BNB ADA LTC

# 2. Entrainer PLUSIEURS FOIS (rapide ~10s de chargement)
python src/train.py --data data/prepared/dataset_btc_eth_bnb_ada_ltc_5min_kalman.npz --epochs 50
python src/train.py --data data/prepared/dataset_btc_eth_bnb_ada_ltc_5min_kalman.npz --lr 0.0001
```

### Options de prepare_data.py

| Option | Description |
|--------|-------------|
| `--filter kalman` | Filtre Kalman pour labels (recommande) |
| `--assets BTC ETH ...` | Liste des assets a inclure |
| `--list` | Liste les datasets disponibles |

---

## Configuration des Indicateurs

### Periodes Synchronisees (IMPORTANT)

Les indicateurs utilisent des periodes **optimisees pour la synchronisation** avec Kalman(Close):

```python
# src/constants.py - Periodes synchronisees (Lag 0)
# Score = Concordance (Lag=0 requis)

# RSI - Synchronise avec Kalman(Close)
RSI_PERIOD = 22         # Lag 0, Concordance 85.3%

# CCI - Synchronise avec Kalman(Close)
CCI_PERIOD = 32         # Lag 0, Concordance 77.9%

# MACD - Synchronise avec Kalman(Close)
MACD_FAST = 8           # Lag 0, Concordance 71.8%
MACD_SLOW = 42
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

### Fichiers CSV (5 assets)

```
data_trad/
|-- BTCUSD_all_5m.csv    # Bitcoin
|-- ETHUSD_all_5m.csv    # Ethereum
|-- BNBUSD_all_5m.csv    # Binance Coin
|-- ADAUSD_all_5m.csv    # Cardano
`-- LTCUSD_all_5m.csv    # Litecoin
```

### Configuration dans constants.py

```python
# Assets disponibles pour le workflow 5min/30min
AVAILABLE_ASSETS_5M = {
    'BTC': 'data_trad/BTCUSD_all_5m.csv',
    'ETH': 'data_trad/ETHUSD_all_5m.csv',
    'BNB': 'data_trad/BNBUSD_all_5m.csv',
    'ADA': 'data_trad/ADAUSD_all_5m.csv',
    'LTC': 'data_trad/LTCUSD_all_5m.csv',
}

# Assets par defaut (peut etre etendu)
DEFAULT_ASSETS = ['BTC', 'ETH']
```

**Note**: Pour utiliser tous les assets, specifier explicitement: `--assets BTC ETH BNB ADA LTC`

---

## Pipeline de Preparation des Donnees (5min)

### Commande principale

```bash
python src/prepare_data.py --filter kalman --assets BTC ETH BNB ADA LTC
```

### Processus

1. **Chargement**: Donnees 5min pour chaque asset
2. **Trim edges**: 100 bougies debut + 100 fin
3. **Calcul indicateurs**: RSI, CCI, MACD (normalises 0-100)
4. **Generation labels**: Pente des indicateurs (filtre Kalman)
5. **Split temporel**: 70% train / 15% val / 15% test (avec GAP)
6. **Creation sequences**: 12 timesteps
7. **Sauvegarde**: `.npz` compresse

### Options CLI

```bash
python src/prepare_data.py --help

Options:
  --assets BTC ETH ...    Assets a inclure (defaut: BTC ETH)
  --filter {decycler,kalman}  Filtre pour labels (defaut: decycler)
  --output PATH           Chemin de sortie (defaut: auto)
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
RSI_PERIOD = 22     # Concordance 85.3%
CCI_PERIOD = 32     # Concordance 77.9%
MACD_FAST = 8       # Concordance 71.8%
MACD_SLOW = 42
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

| Metrique | Baseline | Cible | Actuel (2026-01-03) |
|----------|----------|-------|---------------------|
| Accuracy moyenne | 50% | 85%+ | **85.1%** ✅ ATTEINT |
| Gap train/val | - | <10% | 3.6% ✅ |
| Gap val/test | - | <10% | 0.9% ✅ |
| Prochain objectif | - | **90%** | En cours |

### Resultats par Indicateur (Test Set) - Clock-Injected 7 Features

| Indicateur | Accuracy | F1 | Precision | Recall |
|------------|----------|-----|-----------|--------|
| RSI | 83.0% | 0.827 | 0.856 | 0.800 |
| CCI | 85.6% | 0.858 | 0.846 | 0.869 |
| MACD | **86.8%** | 0.871 | 0.849 | 0.894 |
| **MOYENNE** | **85.1%** | **0.852** | **0.851** | **0.854** |

### Configuration Optimale Actuelle (Clock-Injected)

```bash
# Preparation
python src/prepare_data_30min.py --filter kalman --assets BTC ETH BNB ADA LTC --include-30min-features

# Entrainement
python src/train.py --data data/prepared/dataset_btc_eth_bnb_ada_ltc_5min_30min_labels30min_kalman.npz --epochs 50
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

## Roadmap: Le Saut vers 90%

### Situation Actuelle (2026-01-03)

| Metrique | Valeur |
|----------|--------|
| Test Accuracy | **85.1%** ✅ |
| Gap Val/Test | 0.9% (excellent) |
| Objectif | **90%** |

L'architecture Clock-Injected a franchi le cap des 85%. Le gap Val/Test ultra-faible indique une excellente generalisation.

### Leviers Identifies (Analyse Expert)

#### Levier 1: Optimisation Fine des Hyperparametres

Le modele converge en seulement 5 epoques → il "apprend vite" mais peut-etre de maniere trop superficielle.

**Actions recommandees:**
- **Learning Rate Decay**: Commencer avec LR=0.001, diviser par 10 toutes les 3 epoques
- **Patience Early Stopping**: Augmenter a 15-20 pour laisser le modele affiner ses poids
- **Plus d'epoques**: Permettre jusqu'a 50-100 epoques avec LR decay

```bash
# Exemple avec LR plus bas et plus de patience
python src/train.py --data <dataset> --epochs 100 --lr 0.0005 --patience 20
```

#### Levier 2: Architecture "Fusion de Canaux"

Pour franchir les 90%, creer deux branches LSTM separees:

```
                    Input (12, 7)
                         |
          ┌──────────────┴──────────────┐
          ▼                              ▼
    ┌─────────────┐              ┌─────────────┐
    │ Branche     │              │ Branche     │
    │ Signaux     │              │ Contexte    │
    │ Rapides     │              │ Lourd       │
    │ (5min)      │              │ (30min+Step)│
    └──────┬──────┘              └──────┬──────┘
           │                            │
           └──────────┬─────────────────┘
                      ▼
               ┌─────────────┐
               │ Concatenate │
               │ + Dense     │
               └──────┬──────┘
                      ▼
                 3 Outputs
```

Cela force le reseau a traiter le contexte 30min comme une "verite de controle".

#### Levier 3: Pivot Filtering (Synchronisation RSI)

Regarder les erreurs de prediction (Faux Positifs):
- Si elles surviennent souvent sur Steps 1-2 → manque de confiance en debut de cycle
- Action: Augmenter le poids de Pivot Accuracy a 0.5 pour le RSI dans `optimize_sync.py`

### Volume et ATR

**Note**: Le Volume et l'ATR seront utilises **apres le modele**, dans la strategie de trading, pas comme features du modele.

### Checklist Avant Production

- [x] Accuracy >= 85% sur test set ✅ (85.1%)
- [x] Gap train/test <= 10% ✅ (0.9%)
- [x] Indicateurs synchronises (RSI=14, CCI=20, MACD=10/26, Lag 0) ✅
- [x] Split temporel strict ✅
- [x] Bibliotheque ta utilisee ✅
- [ ] Accuracy >= 90% sur test set (en cours)
- [ ] Backtest sur donnees non vues
- [ ] Trading strategy definie avec Volume filtering

**Voir spec complete**: [docs/SPEC_CLOCK_INJECTED.md](docs/SPEC_CLOCK_INJECTED.md)

---

## Strategie de Trading

### Principe Fondamental

Le modele predit la pente **passee** (t-2 → t-1) avec haute accuracy (~85%).
L'interet n'est pas la prediction elle-meme, mais la **stabilite** des predictions sur les 6 steps.

### Comment ca marche

A chaque periode 30min, le modele fait 6 predictions (Steps 1-6) sur la MEME pente passee:

| Step | Timestamp | Predit | Interpretation |
|------|-----------|--------|----------------|
| 1 | 10:00 | pente(9:00→9:30) | Premiere lecture |
| 2 | 10:05 | pente(9:00→9:30) | Confirmation ? |
| 3 | 10:10 | pente(9:00→9:30) | Stable ? |
| 4 | 10:15 | pente(9:00→9:30) | Stable ? |
| 5 | 10:20 | pente(9:00→9:30) | Stable ? |
| 6 | 10:25 | pente(9:00→9:30) | Derniere lecture |

**Signal de trading** = Quand le modele **change d'avis** sur la meme pente passee.
Cela indique que les features recentes (prix actuel) contredisent la tendance passee → retournement probable.

### Regles de Trading

| # | Regle | Raison |
|---|-------|--------|
| 1 | **Ne jamais agir a Step 1** (xx:00 ou xx:30) | Premiere lecture, pas de confirmation |
| 2 | Attendre Step 2+ pour confirmer | Evite les faux signaux |
| 3 | Changement d'avis = Signal d'action | Le modele voit le retournement dans les features |
| 4 | Stabilite sur 3+ steps = Confiance haute | Tendance confirmee |

### Exemple Concret

```
Pente reelle: 9:00→9:30 = UP, puis retournement a 10:15

10:00  Modele: UP   → Attendre (Step 1)
10:05  Modele: UP   → Confirme, entrer LONG
10:10  Modele: UP   → Stable, rester
10:15  Modele: DOWN → ⚠️ Changement! Le modele voit le retournement
10:20  Modele: DOWN → Confirme, sortir/inverser
```

Le modele se "trompe" sur la pente passee car ses features actuelles voient deja le retournement.
C'est un **signal avance** du changement de tendance.

---

## Methodologie d'Optimisation des Indicateurs

### Principe: Concordance Pure (Prediction Focus)

L'optimisation des parametres d'indicateurs est basee sur la **concordance** avec la reference, pas sur les pivots ou l'anticipation.

**Pourquoi?**
- L'objectif du modele ML est de **PREDIRE** (maximiser accuracy train/val)
- Les pivots et l'anticipation sont pour le **TRADING** (apres le modele)
- Des features concordantes = signal coherent pour le modele

### Scoring

```python
Score = Concordance   # si Lag == 0 (synchronise)
Score = 0             # si Lag != 0 (desynchronise, disqualifie)
```

Un indicateur desynchronise (Lag != 0) envoie des signaux contradictoires au modele → il est elimine.

### Grilles de Parametres

Chaque indicateur est teste avec **±60% (3 pas de 20%)** autour de sa valeur par defaut:

| Indicateur | Defaut | Grille testee |
|------------|--------|---------------|
| RSI period | 22 | [35, 26, 22, 18, 9] |
| CCI period | 32 | [51, 38, 32, 26, 13] |
| MACD fast | 8 | [13, 10, 8, 6, 3] |
| MACD slow | 42 | [67, 50, 42, 34, 17] |

Plage de lag testee: **-3 a +2** (suffisant pour detecter la synchronisation)

### Pipeline en 2 Etapes

**Etape 1: Optimisation sur Close**

Trouver les parametres optimaux pour synchroniser chaque indicateur avec Kalman(Close):

```bash
python src/optimize_sync.py --assets BTC ETH BNB --val-assets ADA LTC
```

Resultat: Nouveaux parametres par defaut pour `constants.py`

**Etape 2: Multi-View Learning - ABANDONNE**

L'approche Multi-View a ete testee et abandonnee. Voir section "Resultats des Experiences" pour details.

### Multi-View Learning: Analyse Post-Mortem

**Hypothese initiale:**
Synchroniser les features (CCI, MACD) avec la cible (ex: RSI) devrait reduire les signaux contradictoires et ameliorer la prediction.

**Parametres testes (2026-01-03):**

| Cible | RSI | CCI | MACD |
|-------|-----|-----|------|
| RSI | 22 (defaut) | 51 | 13/67 |
| CCI | 18 | 32 (defaut) | 10/67 |
| MACD | 18 | 26 | 8/42 (defaut) |

**Resultats:**

| Indicateur | Baseline 5min | Multi-View 5min | Delta |
|------------|---------------|-----------------|-------|
| MACD | 86.9% | 86.2% | **-0.7%** |

**Conclusion: Multi-View n'ameliore pas la prediction.**

**Pourquoi ca n'a pas fonctionne:**

1. **Synchronisation ≠ Predictibilite**: Des features synchronisees avec la cible sont plus **correlees** avec elle, donc apportent **moins d'information nouvelle**. Pour predire, on veut des features **complementaires**, pas des features qui "copient" la cible.

2. **Redondance vs Diversite**: Le modele ML beneficie de features qui capturent des aspects **differents** du marche. En synchronisant RSI et CCI avec MACD, on perd cette diversite.

3. **Optimisation sur le mauvais critere**: L'optimisation maximisait la **concordance de direction**, mais le modele a besoin de features qui apportent de l'**information predictive**, pas juste de la coherence.

**Decision: Revenir aux parametres par defaut (optimises pour Close)**

```python
# constants.py - Parametres FINAUX
RSI_PERIOD = 22    # Optimise pour Kalman(Close)
CCI_PERIOD = 32    # Optimise pour Kalman(Close)
MACD_FAST = 8      # Optimise pour Kalman(Close)
MACD_SLOW = 42     # Optimise pour Kalman(Close)
```

Ces parametres restent les meilleurs car ils sont optimises pour suivre la tendance du prix (Close), ce qui est l'objectif final du trading.

---

## Backlog: Experiences a Tester

Liste organisee des experiences et optimisations a tester pour atteindre 90%+.

### Priorite 1: Architecture et Training

| # | Experience | Hypothese | Commande/Implementation | Statut |
|---|------------|-----------|-------------------------|--------|
| 1.1 | **Training par indicateur** | Un modele specialise par indicateur (RSI, CCI, MACD) pourrait mieux apprendre les patterns specifiques | `python src/train.py --indicator rsi` | **Teste** - Gain negligeable |
| 1.2 | **Fusion de canaux** | Separer branche 5min et branche 30min dans le LSTM | Modifier `model.py` (voir Roadmap Levier 2) | A tester |
| 1.3 | **Learning Rate Decay** | LR=0.001 → 0.0001 progressif pour affiner les poids | `--lr-decay step --lr-step 10` | A tester |
| 1.4 | **Plus de patience** | Early stopping a 20 epoques au lieu de 10 | `--patience 20 --epochs 100` | A tester |
| 1.5 | **Multi-View Learning** | Optimiser les features (CCI, MACD) pour synchroniser avec la cible (RSI) | `python src/optimize_sync_per_target.py --target rsi` | **Teste** - MACD -0.7%, Abandonne |

### Priorite 2: Features et Donnees

| # | Experience | Hypothese | Commande/Implementation | Statut |
|---|------------|-----------|-------------------------|--------|
| 2.1 | **Multi-resolution 1h** | Ajouter indicateurs 1h comme contexte macro | `--include-1h-features` | A tester |
| 2.2 | **Embeddings temporels** | Heure/jour en sin/cos pour capturer cycles | Ajouter 4 features (sin/cos hour, sin/cos day) | A tester |
| 2.3 | **Sequence length 24** | Plus de contexte temporel (2h au lieu de 1h) | `--seq-length 24` | A tester |

### Priorite 3: Regularisation et Robustesse

| # | Experience | Hypothese | Commande/Implementation | Statut |
|---|------------|-----------|-------------------------|--------|
| 3.1 | **Dropout augmente** | LSTM dropout 0.3 au lieu de 0.2 | Modifier `constants.py` | A tester |
| 3.2 | **Label smoothing** | Adoucir labels (0.1/0.9 au lieu de 0/1) | Modifier `train.py` loss | A tester |
| 3.3 | **Data augmentation** | Ajouter bruit gaussien sur features | Modifier `prepare_data_30min.py` | A tester |

### Priorite 4: Analyse et Debug

| # | Experience | Hypothese | Commande/Implementation | Statut |
|---|------------|-----------|-------------------------|--------|
| 4.1 | **Verification alignement** | S'assurer que Step 1-6 ont meme accuracy | `python src/analyze_errors.py` | En cours |
| 4.2 | **Confusion par asset** | Certains assets plus faciles que d'autres? | Ajouter `--by-asset` a evaluate.py | A tester |
| 4.3 | **Erreurs temporelles** | Les erreurs sont-elles clustered dans le temps? | Ajouter analyse temporelle des erreurs | A tester |

### Comment utiliser ce backlog

1. **Choisir** une experience par priorite
2. **Implementer** la modification
3. **Tester** avec le dataset standard
4. **Documenter** le resultat dans la colonne Statut
5. **Garder** si gain > 0.5%, sinon revenir en arriere

### Resultats des Experiences

| Date | Experience | Resultat | Delta | Decision |
|------|------------|----------|-------|----------|
| 2026-01-03 | Position Index | 83.4% | +0.1% | Abandonne |
| 2026-01-03 | Clock-Injected 7 feat | 85.1% | +1.8% | **Adopte** |
| 2026-01-03 | Single-output RSI | 83.6% | +0.6% vs multi | Pas de gain significatif |
| 2026-01-03 | Single-output CCI | 85.6% | = vs multi | Pas de gain significatif |
| 2026-01-03 | Single-output MACD | 86.8% | = vs multi | Pas de gain significatif |
| 2026-01-03 | Multi-View MACD 5min | 86.2% | **-0.7%** | **Abandonne** - synchronisation reduit diversite |

### Analyse Single-Output (2026-01-03)

**Resultats detailles:**

| Indicateur | Train Acc | Val Acc | Test Acc | Gap Train/Val | Gap Val/Test |
|------------|-----------|---------|----------|---------------|--------------|
| RSI | ~88% | ~84% | 83.6% | ~4% | ~0% |
| CCI | ~89% | ~86% | 85.6% | ~3% | ~0% |
| MACD | 90.4% | 86.4% | 86.8% | **4%** | -0.4% |

**Conclusion:**
- Le training single-output **n'apporte pas d'amelioration** significative
- Gap train/val de ~4% = leger overfitting acceptable
- Gap val/test proche de 0% = bonne generalisation
- Early stopping efficace (arret epoque 4-14)

**Pistes pour reduire le gap train/val:**
- Data augmentation (bruit gaussien σ=0.01-0.02)
- Dropout augmente (0.3 → 0.4)
- Label smoothing (0.1)

---

## FEATURE FUTURE - Machine a Etat Multi-Filtres (Octave + Kalman)

**Date**: 2026-01-04
**Statut**: A implementer apres stabilisation du modele ML
**Priorite**: Post-production

### Concept

Utiliser **deux filtres** (Octave + Kalman) appliques au meme signal pour obtenir plusieurs estimations de l'etat latent. Ces estimations sont utilisees dans la **machine a etat** (pas dans le modele ML).

### Difference Fondamentale Octave vs Kalman

| Filtre | Nature | Ce qu'il "voit" bien |
|--------|--------|----------------------|
| **Octave** | Frequentiel (Butterworth) | Structure, cycles, tendances |
| **Kalman** | Etat probabiliste | Continuite, incertitude, variance |

Les deux sont **complementaires**, pas redondants.

### Resultats Empiriques - Comparaison Octave20 vs Kalman (2026-01-04)

#### Concordance des labels (Train vs Test)

| Indicateur | Train | Test | Delta | Isoles (Test) |
|------------|-------|------|-------|---------------|
| RSI | 86.8% | 88.5% | +1.7% | 69.0% |
| CCI | 88.6% | 89.2% | +0.6% | 67.0% |
| MACD | 90.2% | 89.9% | -0.3% | 64.6% |

**Observation** : Concordance stable ou meilleure sur test → les filtres generalisent bien.

#### Accuracy ML (OHLC 5 features)

| Indicateur | Octave20 | Kalman | Delta |
|------------|----------|--------|-------|
| RSI | 83.3% | 81.4% | **-1.9%** |
| CCI | ~85% | 79.0% | **~-6%** |
| MACD | 84.3% | 77.5% | **-6.8%** |

**Conclusion** : **Octave20 > Kalman** pour le ML, sans exception.

#### Paradoxe MACD (RESOLU)

| Observation | MACD | RSI |
|-------------|------|-----|
| Concordance filtres | **90%** (meilleure) | 87% |
| Perte accuracy Kalman | **-6.8%** (pire) | -1.9% |

**Ce n'est PAS un paradoxe** (validation expert) :

- MACD est deja un indicateur tres lisse
- Kalman re-lisse encore → **trop peu d'entropie**
- Resultat : peu de retournements, transitions graduelles, frontieres floues
- **Pour un humain** : excellent (signal propre)
- **Pour un classifieur ML** : cauchemar (pas assez de contraste)

> "Haute concordance ≠ bonne predictibilite. Le ML a besoin de contraste, pas de douceur."

#### Observations cles

1. **Plus l'indicateur est "lourd", plus les filtres sont d'accord**
   - RSI (oscillateur vitesse) : 87-89% concordance
   - CCI (oscillateur deviation) : 89% concordance
   - MACD (indicateur tendance) : 90% concordance

2. **~2/3 des desaccords sont isoles** (1 sample) - CHIFFRE CLE
   - = Moments transitoires brefs (micro pullbacks, respirations)
   - Les 35% restants = blocs de desaccord (vraies zones d'incertitude)
   - **Implication** : Sortir sur un desaccord isole est presque toujours une erreur
   - **Justification mathematique** pour la regle de confirmation 2+ periodes

3. **Recommandations finales (validees par expert) :**
   - **Modele ML** : Utiliser **Octave20 exclusivement** (labels nets, meilleure separabilite)
   - **Kalman** : Detecteur d'incertitude, pas predicteur ("Est-ce que je suis confiant ?")
   - **Anti-flicker** : Confirmation 2+ periodes = filtre quasi-optimal (elimine 65% faux signaux)
   - **MACD** : Indicateur pivot (plus stable), RSI/CCI = modulateurs

#### Architecture Finale (convergence)

```
OHLC → Modele ML (Octave20)
           ↓
     Direction probabiliste
           ↓
 Kalman → Incertitude / confiance
           ↓
  Machine a etats :
    - MACD pivot (declencheur principal)
    - RSI/CCI modulateurs (pas declencheurs)
    - Confirmation temporelle (2+ periodes)
    - Ignorer desaccords isoles
    - Prudence en zone Kalman floue
```

> "Tu n'es plus dans l'exploration, mais dans la convergence."
> — Expert

**Commande de comparaison :**
```bash
python src/compare_datasets.py \
    --file1 data/prepared/dataset_btc_eth_bnb_ada_ltc_ohlcv2_<indicator>_octave20.npz \
    --file2 data/prepared/dataset_btc_eth_bnb_ada_ltc_ohlcv2_<indicator>_kalman.npz \
    --split train --sample 20000
```

### Ce que ca apporte

- Mesure de **robustesse** du signal
- Information sur la **vitesse** (Octave) et la **stabilite/confiance** (Kalman)
- Capacite a detecter:
  - Transitions reelles vs bruit transitoire
  - Zones d'incertitude (desaccord entre filtres)

### Ce que ca N'apporte PAS

- Pas de nouvel alpha
- Pas d'amelioration brute de l'accuracy ML
- Ce n'est pas une source d'edge autonome

**C'est un amplificateur de decision, pas une source d'alpha.**

### Ou utiliser ces filtres (CRUCIAL)

**❌ PAS dans le modele ML:**
- Double comptage d'information
- Correlation extreme entre les deux
- Peu de gain ML
- Risque de fuite deguisee

**✅ Dans la machine a etat:**
- Regles de validation
- Modulation de confiance
- Gestion des sorties

### Regles de Combinaison

#### Cas 1: Accord total
```
Octave_dir == UP
Kalman_dir == UP
```
→ Signal fort → tolerance au bruit ↑
→ Trades plus longs

#### Cas 2: Desaccord
```
Octave_dir != Kalman_dir
```
→ Zone de transition
→ Reduire l'agressivite:
  - Confirmation plus longue
  - Sorties plus strictes
  - Pas d'inversion directe

#### Cas 3: Kalman variance elevee
```
Kalman_var > seuil
```
→ Marche instable
→ Interdire nouvelles entrees
→ Laisser courir positions existantes

### Exemple d'Integration dans la State Machine

**Entree LONG:**
```python
if model_pred == UP:
    if octave_dir == UP and kalman_dir != DOWN:
        enter_long()      # Accord = confiance haute
    else:
        wait_confirmation()  # Desaccord = patience
```

**Sortie LONG (early):**
```python
if octave_dir == DOWN and kalman_dir == DOWN:
    exit_long()  # Vrai retournement confirme
```

**Sortie LONG (late):**
```python
if kalman_var > seuil and rsi_faiblit:
    exit_long()  # Marche devient instable
```

### Application au Probleme de Micro-Sorties

Le modele fait ~2500 trades vs ~800 pour Oracle (3x trop).

Avec cette logique:
- **Accord filtres** → permettre le trade
- **Desaccord filtres** → ignorer le changement (probablement du bruit)

Cela devrait reduire les micro-sorties sans toucher au modele ML.

### Implementation Prevue

1. **Calculer les deux filtres** sur le signal cible (ex: MACD)
2. **Extraire la direction** de chaque filtre (pente > 0 ?)
3. **Extraire la variance Kalman** comme mesure d'incertitude
4. **Ajouter ces colonnes** au DataFrame de backtest
5. **Modifier la state machine** pour utiliser ces informations

### Pieges a Eviter

**⚠️ 1. Trop de regles**
```
Octave + Kalman + RSI + CCI + MACD = explosion combinatoire
```
→ Solution: Garder simple
  - Octave = structure
  - Kalman = confiance
  - ML = direction

**⚠️ 2. Seuils trop fins**
→ Sur-optimisation, non robustesse
→ Garder des seuils grossiers

### Avantage Architectural

C'est une strategie d'**architecture evolutive**:
- **Aujourd'hui**: Modele ML stable + state machine simple
- **Demain**: Enrichir la machine sans retrainer le modele

Le modele reste inchange, on ameliore la **qualite decisionnelle** en aval.

### Methodologie - Apprendre la State Machine des Erreurs

**Principe fondamental**: Les accords sont sans interet, les desaccords contiennent toute l'information.

#### Pourquoi analyser les desaccords?

| Situation | Information | Action |
|-----------|-------------|--------|
| Tous d'accord | Aucune (decision evidente) | Rien a apprendre |
| **Desaccord** | Zone de conflit | **Deduire des regles** |

La state machine n'ajoute pas de signal, elle ajoute de la **coherence temporelle**.

#### Methode 1: Analyse Conditionnelle des Erreurs (RECOMMANDEE)

**Etape 1 - Logger tout** (script `analyze_errors_state_machine.py`):
```
Pour chaque timestep:
- Predictions: RSI_pred, CCI_pred, MACD_pred
- Filtres: Octave_dir, Kalman_dir, Kalman_var
- Contexte: StepIdx, trade_duration
- Reference: Oracle action
- Resultat: action modele, P&L
```

**Etape 2 - Isoler les cas problematiques**:
```python
# Erreurs a analyser
Model = LONG, Oracle = HOLD ou SHORT
Model = SHORT, Oracle = HOLD ou LONG
```

**Etape 3 - Chercher les patterns**:
```
❌ Erreurs frequentes quand:
   - RSI = DOWN, MACD = UP (conflit)
   - Kalman variance elevee
   - StepIdx < 3 (debut de cycle)

❌ Sorties prematurees quand:
   - Octave encore UP
   - trade_duration < 3 periodes
```

**Etape 4 - Transformer en regles**:
```python
if position == LONG and model_pred == DOWN:
    if octave_dir == kalman_dir == UP:
        if trade_duration < 3:
            action = HOLD  # Ignorer le flip
```

#### Methode 2: Decision Tree (Regles Explicites)

Entrainer un arbre de decision peu profond:
```python
Inputs = [RSI_pred, CCI_pred, MACD_pred, Octave_dir, Kalman_dir, StepIdx]
Target = Oracle_action
max_depth = 4  # Limiter pour eviter overfit
```

Extraire les regles:
```
SI MACD == UP
ET StepIdx < 3
ET Kalman_var > seuil
ALORS HOLD (pas encore confirme)
```

#### Methode 3: Clustering des Desaccords

1. Filtrer les timesteps ou indicateurs/filtres divergent
2. Clustering (K-means, DBSCAN) sur les features
3. Chaque cluster = un "type de conflit"

| Cluster | Caracteristiques | Interpretation |
|---------|------------------|----------------|
| A | RSI flip, MACD stable | Faux retournement |
| B | Tous changent, StepIdx > 4 | Vrai retournement |
| C | Kalman_var haute | Zone d'incertitude |

#### Priorite d'Implementation

| # | Methode | Complexite | Risque overfit |
|---|---------|------------|----------------|
| **1** | Analyse erreurs | Faible | Faible |
| 2 | Decision Tree | Moyenne | Moyen |
| 3 | Clustering | Elevee | Eleve |

#### Script analyze_errors_state_machine.py

```bash
# Analyser les erreurs sur le split test
python src/analyze_errors_state_machine.py \
    --data data/prepared/dataset_..._octave20.npz \
    --data-kalman data/prepared/dataset_..._kalman.npz \
    --split test \
    --output results/error_analysis.csv
```

Colonnes generees:
- `timestamp`, `asset`
- `rsi_pred`, `cci_pred`, `macd_pred`
- `octave_dir`, `kalman_dir`, `filters_agree`
- `oracle_action`, `model_action`, `is_error`
- `trade_duration`, `step_idx`

#### Resultats Analyse Erreurs (Test Set - 640k samples)

| Metrique | RSI | CCI | MACD |
|----------|-----|-----|------|
| **Accuracy** | 83.4% | 82.5% | **84.2%** |
| Erreurs totales | 106k | 112k | **101k** |
| False Positive | 8.9% | 10.1% | 8.0% |
| False Negative | 7.7% | 7.4% | 7.8% |
| Accord filtres | 88.4% | 89.1% | 90.2% |
| **Erreur si accord** | 13.8% | 15.8% | 15.6% |
| **Erreur si desaccord** | 38.3% | 31.5% | 18.3% |
| **Ratio desaccord/accord** | **2.8x** | 2.0x | 1.2x |
| Erreurs isolees | **70%** | 62% | 63% |
| Erreur apres transition | **5.4x** | 3.1x | 2.6x |

**Observations cles :**

1. **MACD = Indicateur le plus stable**
   - Meilleure accuracy (84.2%), moins d'erreurs
   - Ratio desaccord/accord = 1.2x seulement → insensible aux conflits de filtres
   - Regle 1 (prudence si desaccord) NON necessaire pour MACD

2. **RSI = Le plus sensible aux conflits**
   - 2.8x plus d'erreurs quand filtres en desaccord
   - 70% d'erreurs isolees (le plus eleve)
   - 5.4x plus d'erreurs apres transition → tres reactif

3. **Regles validees empiriquement :**
   - Confirmation 2+ periodes : elimine 60-70% des erreurs (toutes isolees)
   - Delai post-transition : critique pour RSI (5.4x), modere pour MACD (2.6x)
   - Prudence si desaccord filtres : critique RSI (2.8x), inutile MACD (1.2x)

**Implications State Machine :**

| Regle | RSI | CCI | MACD |
|-------|-----|-----|------|
| Prudence si desaccord filtres | ✅ Critique | ✅ Important | ❌ Pas necessaire |
| Confirmation 2+ periodes | ✅ | ✅ | ✅ |
| Delai post-transition | ✅ Critique | ✅ Important | ✅ Modere |

→ **MACD confirme comme pivot** : plus stable, moins sensible aux conflits
→ **RSI/CCI = modulateurs** necessitant plus de filtrage

#### Ce qu'il ne faut PAS faire

| ⚠️ Piege | Pourquoi |
|----------|----------|
| Chercher des regles ou tout va bien | Aucun signal |
| Laisser un NN decider seul | Perte de stabilite |
| Apprendre sur le P&L directement | Trop bruite |
| Trop de regles | Explosion combinatoire |
| Seuils trop fins | Sur-optimisation |

---

**Cree par**: Claude Code
**Derniere MAJ**: 2026-01-04
**Version**: 4.6 (+ Validation Expert + Architecture Finale)
