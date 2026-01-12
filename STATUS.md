# PROJET TRAD - STATUS

**Dernière mise à jour**: 2026-01-12
**État actuel**: Aucun dataset généré (data/prepared/ est vide)

---

## 📊 PIPELINE DATASET - Vue d'ensemble

```
[1] Base Dataset          [2] Train Model A + Enrichment      [3] Train Direction Models + Enrichment
prepare_data_regime.py -> train_regime_classifier.py    -> train.py (macd/rsi/cci)
     (8 colonnes Y)       Entraîne XGBoost + enrichit           Entraîne CNN-LSTM + enrichit
                          (+5 colonnes → Y=13)                  (+6 colonnes → Y=19)
```

---

## 📁 ÉTAPE 1: Base Dataset (RÉGIME)

**Script**: `src/prepare_data_regime.py`
**Commande**:
```bash
python src/prepare_data_regime.py --assets BTC ETH BNB ADA LTC
```

**Fichier généré**: `data/prepared/dataset_btc_eth_bnb_ada_ltc_regime.npz`

**Status actuel**: ✅ **EXÉCUTÉ** (dataset base: 547M, backup: 461M)

### Structure NPZ (Base)

| Array | Shape | Description |
|-------|-------|-------------|
| `X_train` | (n_train, 25, ~22) | Séquences features train (25 timesteps × ~22 features) |
| `Y_train` | (n_train, 8) | Labels + metadata train |
| `OHLCV_train` | (n_train, 7) | Prix OHLCV + metadata train |
| `X_val` | (n_val, 25, ~22) | Séquences features val |
| `Y_val` | (n_val, 8) | Labels + metadata val |
| `OHLCV_val` | (n_val, 7) | Prix OHLCV + metadata val |
| `X_test` | (n_test, 25, ~22) | Séquences features test |
| `Y_test` | (n_test, 8) | Labels + metadata test |
| `OHLCV_test` | (n_test, 7) | Prix OHLCV + metadata test |

**Note**: X contient [timestamp, asset_id, ...features] donc shape (n, 25, 2 + n_features_regime).

### Features X (~22 canaux × 25 timesteps)

**Structure**: X[:, :, 0-1] = metadata, X[:, :, 2:] = features de régime

| Index | Feature | Type | Description |
|-------|---------|------|-------------|
| 0 | `timestamp` | int64 | Unix timestamp (metadata) |
| 1 | `asset_id` | int | ID asset 0-4 (metadata) |
| **2-8** | **Trend features (7)** | float | MA slopes, regression, ADX, Hurst, MACD histogram |
| **9-17** | **Volatility features (9)** | float | ATR, Bollinger Bands, realized vol, compression, range/ATR ratio |
| **18-21** | **Volume & microstructure (4)** | float | Volume ratio/spike, VWAP deviation, OBV derivative |

**Total features régime**: ~20 (7 trend + 9 vol + 4 volume) - Voir `regime_features.py`

### Labels Y (8 colonnes) - BASE DATASET

| Index | Colonne | Type | Valeurs | Description |
|-------|---------|------|---------|-------------|
| **0** | `timestamp` | int64 | Unix timestamp | Timestamp de la bougie (Open time) |
| **1** | `asset_id` | int | 0-4 | ID de l'asset (0=BTC, 1=ETH, 2=BNB, 3=ADA, 4=LTC) |
| **2** | `regime` | int | 0-3 | **Label principal Model A** (4 classes régime) |
| **3** | `trend_strength` | float | 0-1 | Score de force de tendance (ground truth) |
| **4** | `volatility_cluster` | float | 0-1 | Score de cluster de volatilité (ground truth) |
| **5** | `macd_direction` | int | 0/1 | Direction MACD Kalman (0=DOWN, 1=UP) |
| **6** | `rsi_direction` | int | 0/1 | Direction RSI Kalman (0=DOWN, 1=UP) |
| **7** | `cci_direction` | int | 0/1 | Direction CCI Kalman (0=DOWN, 1=UP) |

**Note**: Les colonnes 5-7 (directions) sont des **labels de référence** pour entraîner les modèles de direction.

### OHLCV (7 colonnes)

| Index | Colonne | Type | Description |
|-------|---------|------|-------------|
| 0 | `timestamp` | int64 | Unix timestamp (Open time) |
| 1 | `asset_id` | int | ID de l'asset (0-4) |
| 2 | `open` | float | Prix d'ouverture |
| 3 | `high` | float | Prix haut |
| 4 | `low` | float | Prix bas |
| 5 | `close` | float | Prix de clôture |
| 6 | `volume` | float | Volume brut (non normalisé) |

---

## 🎯 ÉTAPE 2: Entraînement Model A + Enrichissement

**Script**: `src/train_regime_classifier.py`
**Commande**:
```bash
python src/train_regime_classifier.py \
    --data data/prepared/dataset_btc_eth_bnb_ada_ltc_regime.npz \
    --output-dir models
```

**Modèle généré**: `models/regime_classifier_xgboost.pkl`

**Dataset enrichi**: `data/prepared/dataset_btc_eth_bnb_ada_ltc_regime.npz` (REMPLACÉ IN-PLACE)

**Backup créé**: `data/prepared/dataset_btc_eth_bnb_ada_ltc_regime_original.npz`

**Status actuel**: ✅ **EXÉCUTÉ** (dataset enrichi: 547M, backup: 461M)

**Objectif**:
1. Entraîner XGBoost multiclass pour prédire `Y[:, 2]` (regime 0-3)
2. Enrichir automatiquement le dataset avec les prédictions

**Architecture**:
- Input: Features régime extraites de X (trend, vol, volume)
- Modèle: XGBoost multiclass (200 arbres, depth=6)
- Output: Classe prédite + 4 probabilités (une par régime)

### Modification Y: 8 colonnes → 13 colonnes (+5)

**L'enrichissement est fait AUTOMATIQUEMENT par train_regime_classifier.py**

**Colonnes ajoutées** (indices 8-12):

| Index | Colonne | Type | Valeurs | Description |
|-------|---------|------|---------|-------------|
| **8** | `regime_pred` | int | 0-3 | ✨ **Prédiction Model A** (classe prédite) |
| **9** | `regime_prob_0` | float | 0-1 | ✨ Probabilité régime 0 (Model A) |
| **10** | `regime_prob_1` | float | 0-1 | ✨ Probabilité régime 1 (Model A) |
| **11** | `regime_prob_2` | float | 0-1 | ✨ Probabilité régime 2 (Model A) |
| **12** | `regime_prob_3` | float | 0-1 | ✨ Probabilité régime 3 (Model A) |

**Colonnes 0-7**: Inchangées (structure base)

**Note importante**: Le fichier dataset est remplacé in-place, un backup est créé automatiquement.

---

## 🎯 ÉTAPE 3: Entraînement Modèles Direction + Enrichissement

**Script**: `src/train.py` (3 exécutions séparées)

### 3.1 - MACD Direction Model

**Commande**:
```bash
python src/train.py \
    --data data/prepared/dataset_btc_eth_bnb_ada_ltc_regime.npz \
    --indicator macd \
    --epochs 50 \
    --grad-clip 1.0 \
    --lr 0.0001
```

**Modèle généré**: `models/best_model_macd_direction.pth`

**Target**: `Y[:, 5]` (macd_direction)

**Dataset enrichi**: `data/prepared/dataset_btc_eth_bnb_ada_ltc_regime.npz` (CLÉS NPZ ajoutées)

**Clés ajoutées** : `Y_train_pred`, `Y_val_pred`, `Y_test_pred` (arrays séparés, shape (n, 1) pour MACD)

**Objectif**:
1. Entraîner CNN-LSTM pour prédire direction MACD (colonne 5)
2. Enrichir automatiquement le dataset avec clé NPZ `Y_*_pred`

**Note**: train.py ajoute des **clés NPZ séparées**, ne modifie PAS la structure de Y

### 3.2 - RSI Direction Model

**Commande**:
```bash
python src/train.py \
    --data data/prepared/dataset_btc_eth_bnb_ada_ltc_regime.npz \
    --indicator rsi \
    --epochs 50 \
    --grad-clip 1.0 \
    --lr 0.0001
```

**Modèle généré**: `models/best_model_rsi_direction.pth`

**Target**: `Y[:, 6]` (rsi_direction)

**Objectif**:
1. Entraîner CNN-LSTM pour prédire direction RSI
2. Enrichir automatiquement le dataset avec prédictions RSI

### 3.3 - CCI Direction Model

**Commande**:
```bash
python src/train.py \
    --data data/prepared/dataset_btc_eth_bnb_ada_ltc_regime.npz \
    --indicator cci \
    --epochs 50 \
    --grad-clip 1.0 \
    --lr 0.0001
```

**Modèle généré**: `models/best_model_cci_direction.pth`

**Target**: `Y[:, 7]` (cci_direction)

**Objectif**:
1. Entraîner CNN-LSTM pour prédire direction CCI
2. Enrichir automatiquement le dataset avec prédictions CCI

**Status actuel**: ❌ **PAS ENTRAÎNÉS** (dataset base n'existe pas encore)

**Note importante**: Chaque modèle de direction enrichit le dataset en ajoutant ses prédictions comme **clés NPZ séparées**, pas comme colonnes dans Y

---

## 📊 STRUCTURE FINALE Y - 13 COLONNES (Après train_regime_classifier.py)

| Index | Colonne | Source | Type | Description |
|-------|---------|--------|------|-------------|
| **0** | timestamp | Base | int64 | Unix timestamp |
| **1** | asset_id | Base | int | ID asset (0-4) |
| **2** | regime | Base | int | Ground truth régime (0-3) |
| **3** | trend_strength | Base | float | Score force tendance (0-1) |
| **4** | volatility_cluster | Base | float | Score cluster volatilité (0-1) |
| **5** | macd_direction | Base | int | Ground truth direction MACD |
| **6** | rsi_direction | Base | int | Ground truth direction RSI |
| **7** | cci_direction | Base | int | Ground truth direction CCI |
| **8** | regime_pred | ÉTAPE 2 | int | ✨ Prédiction Model A (régime) |
| **9** | regime_prob_0 | ÉTAPE 2 | float | ✨ Prob régime 0 |
| **10** | regime_prob_1 | ÉTAPE 2 | float | ✨ Prob régime 1 |
| **11** | regime_prob_2 | ÉTAPE 2 | float | ✨ Prob régime 2 |
| **12** | regime_prob_3 | ÉTAPE 2 | float | ✨ Prob régime 3 |

---

## 📦 CLÉS NPZ SUPPLÉMENTAIRES (Prédictions Direction - ÉTAPE 3)

**Après train.py (3 exécutions pour MACD, RSI, CCI)**, le fichier NPZ contient aussi :

| Clé NPZ | Shape | Type | Description |
|---------|-------|------|-------------|
| `Y_train_pred` | (n_train, 1) | float | Probabilités prédites (0-1) sur train pour l'indicateur entraîné |
| `Y_val_pred` | (n_val, 1) | float | Probabilités prédites (0-1) sur val pour l'indicateur entraîné |
| `Y_test_pred` | (n_test, 1) | float | Probabilités prédites (0-1) sur test pour l'indicateur entraîné |

**Note**: Ces clés sont **écrasées** à chaque exécution de train.py. Pour conserver les 3 prédictions (MACD, RSI, CCI), il faut soit :
- Les sauvegarder séparément après chaque entraînement
- Ou utiliser un script qui combine les 3 modèles

---

## 📋 PROCHAINES ACTIONS IMMÉDIATES

### 1. Générer le dataset base
```bash
python src/prepare_data_regime.py --assets BTC ETH BNB ADA LTC
```

### 2. Vérifier la normalisation
```bash
python tests/check_features_normalization.py \
    --data data/prepared/dataset_btc_eth_bnb_ada_ltc_regime.npz
```

### 3. Entraîner Model A avec stabilité
```bash
python src/train.py \
    --data data/prepared/dataset_btc_eth_bnb_ada_ltc_regime.npz \
    --epochs 50 \
    --grad-clip 1.0 \
    --lr 0.0001
```

---

## 🚨 PROBLÈME RÉSOLU: Training Instability

**Symptômes identifiés**:
- Modèle oscille entre prédire tout 0 et tout 1
- Loss bloquée à ~0.693 (= ln(2) pour classification binaire aléatoire)
- Accuracy ~50%

**Corrections apportées (commit 9b69971)**:
- ✅ Gradient clipping ajouté (torch.nn.utils.clip_grad_norm_)
- ✅ Learning rate réduit (0.001 → 0.0001)
- ✅ Documentation: docs/FIX_TRAINING_INSTABILITY.md

**Status**: Solutions codées mais **NON TESTÉES** (dataset n'existe pas encore)
