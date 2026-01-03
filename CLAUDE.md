# Modele CNN-LSTM Multi-Output - Guide Complet

**Date**: 2026-01-03
**Statut**: Pipeline complet implemente - Objectif 85% ATTEINT
**Version**: 4.0

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

## Backlog: Experiences a Tester

Liste organisee des experiences et optimisations a tester pour atteindre 90%+.

### Priorite 1: Architecture et Training

| # | Experience | Hypothese | Commande/Implementation | Statut |
|---|------------|-----------|-------------------------|--------|
| 1.1 | **Training par indicateur** | Un modele specialise par indicateur (RSI, CCI, MACD) pourrait mieux apprendre les patterns specifiques | Modifier `train.py` pour `--indicator rsi` | A tester |
| 1.2 | **Fusion de canaux** | Separer branche 5min et branche 30min dans le LSTM | Modifier `model.py` (voir Roadmap Levier 2) | A tester |
| 1.3 | **Learning Rate Decay** | LR=0.001 → 0.0001 progressif pour affiner les poids | `--lr-decay step --lr-step 10` | A tester |
| 1.4 | **Plus de patience** | Early stopping a 20 epoques au lieu de 10 | `--patience 20 --epochs 100` | A tester |

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
| - | - | - | - | - |

---

**Cree par**: Claude Code
**Derniere MAJ**: 2026-01-03
**Version**: 4.0 (Clock-Injected 7 features, 85.1% accuracy)
