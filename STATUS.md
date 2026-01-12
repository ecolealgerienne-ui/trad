# PROJET TRAD - STATUS

**Dernière mise à jour**: 2026-01-12
**État actuel**: Aucun dataset généré (data/prepared/ est vide)

---

## 📊 PIPELINE DATASET - Vue d'ensemble

```
[1] Base Dataset          [2] Model A Training      [3] Enrichment Stage 1       [4] Direction Training    [5] Enrichment Stage 2
prepare_data_regime.py -> train.py (regime)    -> enrich_dataset_complete.py -> train.py (direction) -> enrich_dataset_complete.py
     (8 colonnes Y)           Model A trained           (+5 colonnes, Y=13)          Direction models        (+6 colonnes, Y=19)
```

---

## 📁 ÉTAPE 1: Base Dataset (RÉGIME)

**Script**: `src/prepare_data_regime.py`
**Commande**:
```bash
python src/prepare_data_regime.py --assets BTC ETH BNB ADA LTC
```

**Fichier généré**: `data/prepared/dataset_btc_eth_bnb_ada_ltc_regime.npz`

**Status actuel**: ❌ **PAS GÉNÉRÉ** (data/prepared/ est vide)

### Structure NPZ (Base)

| Array | Shape | Description |
|-------|-------|-------------|
| `X_train` | (n_train, 25, 6) | Séquences features train |
| `Y_train` | (n_train, 8) | Labels + metadata train |
| `OHLCV_train` | (n_train, 7) | Prix OHLCV + metadata train |
| `X_val` | (n_val, 25, 6) | Séquences features val |
| `Y_val` | (n_val, 8) | Labels + metadata val |
| `OHLCV_val` | (n_val, 7) | Prix OHLCV + metadata val |
| `X_test` | (n_test, 25, 6) | Séquences features test |
| `Y_test` | (n_test, 8) | Labels + metadata test |
| `OHLCV_test` | (n_test, 7) | Prix OHLCV + metadata test |

### Features X (6 canaux × 25 timesteps)

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `c_ret` | Close return (rendement) |
| 1 | `h_ret` | High return |
| 2 | `l_ret` | Low return |
| 3 | `volume_norm` | Volume normalisé |
| 4 | `atr_norm` | ATR normalisé (volatilité) |
| 5 | `rsi` | RSI(14) normalisé 0-1 |

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

## 🎯 ÉTAPE 2: Entraînement Model A (Régime Classifier)

**Script**: `src/train.py`
**Commande**:
```bash
python src/train.py \
    --data data/prepared/dataset_btc_eth_bnb_ada_ltc_regime.npz \
    --epochs 50 \
    --grad-clip 1.0
```

**Modèle généré**: `models/best_model_regime.pth`

**Status actuel**: ❌ **PAS ENTRAÎNÉ** (dataset base n'existe pas)

**Objectif**: Prédire `Y[:, 2]` (regime 0-3) à partir de `X` (6 features × 25 timesteps)

**Architecture**:
- Input: (batch, 25, 6)
- CNN 1D: 64 filters
- LSTM: 64 hidden × 2 layers
- Output: (batch, 4) - probabilités 4 classes régime

---

## 🔄 ÉTAPE 3: Enrichissement Stage 1 (Prédictions Model A)

**Script**: `src/enrich_dataset_complete.py`
**Commande**:
```bash
python src/enrich_dataset_complete.py \
    --base-dataset data/prepared/dataset_btc_eth_bnb_ada_ltc_regime.npz \
    --model-path models/best_model_regime.pth \
    --output data/prepared/dataset_btc_eth_bnb_ada_ltc_regime_enriched_stage1.npz
```

**Status actuel**: ❌ **PAS EXÉCUTÉ**

### Modification Y: 8 colonnes → 13 colonnes (+5)

**Colonnes ajoutées** (indices 8-12):

| Index | Colonne | Type | Valeurs | Description |
|-------|---------|------|---------|-------------|
| **8** | `regime_pred` | int | 0-3 | ✨ **Prédiction Model A** (classe prédite) |
| **9** | `regime_prob_0` | float | 0-1 | ✨ Probabilité régime 0 (Model A) |
| **10** | `regime_prob_1` | float | 0-1 | ✨ Probabilité régime 1 (Model A) |
| **11** | `regime_prob_2` | float | 0-1 | ✨ Probabilité régime 2 (Model A) |
| **12** | `regime_prob_3` | float | 0-1 | ✨ Probabilité régime 3 (Model A) |

**Colonnes 0-7**: Inchangées (structure base)

---

## 🎯 ÉTAPE 4: Entraînement Modèles Direction (MACD, RSI, CCI)

**Script**: `src/train.py` (3 exécutions séparées)

### 4.1 - MACD Direction Model

**Commande**:
```bash
python src/train.py \
    --data data/prepared/dataset_btc_eth_bnb_ada_ltc_regime_enriched_stage1.npz \
    --indicator macd \
    --epochs 50 \
    --grad-clip 1.0 \
    --lr 0.0001
```

**Modèle généré**: `models/best_model_macd_direction.pth`

**Target**: `Y[:, 5]` (macd_direction)

### 4.2 - RSI Direction Model

**Commande**:
```bash
python src/train.py \
    --data data/prepared/dataset_btc_eth_bnb_ada_ltc_regime_enriched_stage1.npz \
    --indicator rsi \
    --epochs 50 \
    --grad-clip 1.0 \
    --lr 0.0001
```

**Modèle généré**: `models/best_model_rsi_direction.pth`

**Target**: `Y[:, 6]` (rsi_direction)

### 4.3 - CCI Direction Model

**Commande**:
```bash
python src/train.py \
    --data data/prepared/dataset_btc_eth_bnb_ada_ltc_regime_enriched_stage1.npz \
    --indicator cci \
    --epochs 50 \
    --grad-clip 1.0 \
    --lr 0.0001
```

**Modèle généré**: `models/best_model_cci_direction.pth`

**Target**: `Y[:, 7]` (cci_direction)

**Status actuel**: ❌ **PAS ENTRAÎNÉS**

---

## 🔄 ÉTAPE 5: Enrichissement Stage 2 (Prédictions Direction)

**Script**: `src/enrich_dataset_complete.py` (mode extended)
**Commande**:
```bash
python src/enrich_dataset_complete.py \
    --base-dataset data/prepared/dataset_btc_eth_bnb_ada_ltc_regime_enriched_stage1.npz \
    --macd-model models/best_model_macd_direction.pth \
    --rsi-model models/best_model_rsi_direction.pth \
    --cci-model models/best_model_cci_direction.pth \
    --output data/prepared/dataset_btc_eth_bnb_ada_ltc_regime_enriched_stage2.npz
```

**Status actuel**: ❌ **PAS EXÉCUTÉ**

### Modification Y: 13 colonnes → 19 colonnes (+6)

**Colonnes ajoutées** (indices 13-18):

| Index | Colonne | Type | Valeurs | Description |
|-------|---------|------|---------|-------------|
| **13** | `macd_pred` | int | 0/1 | ✨ Prédiction MACD Model (0=DOWN, 1=UP) |
| **14** | `macd_prob` | float | 0-1 | ✨ Probabilité MACD UP (confiance) |
| **15** | `rsi_pred` | int | 0/1 | ✨ Prédiction RSI Model (0=DOWN, 1=UP) |
| **16** | `rsi_prob` | float | 0-1 | ✨ Probabilité RSI UP (confiance) |
| **17** | `cci_pred` | int | 0/1 | ✨ Prédiction CCI Model (0=DOWN, 1=UP) |
| **18** | `cci_prob` | float | 0-1 | ✨ Probabilité CCI UP (confiance) |

**Colonnes 0-12**: Inchangées (stage 1)

---

## 📊 STRUCTURE FINALE Y - 19 COLONNES (Dataset Complet Enrichi)

| Index | Colonne | Source | Type | Description |
|-------|---------|--------|------|-------------|
| **0** | timestamp | Base | int64 | Unix timestamp |
| **1** | asset_id | Base | int | ID asset (0-4) |
| **2** | regime | Base | int | Ground truth régime (0-3) |
| **3** | trend_strength | Base | float | Ground truth force tendance |
| **4** | volatility_cluster | Base | float | Ground truth cluster volatilité |
| **5** | macd_direction | Base | int | Ground truth direction MACD |
| **6** | rsi_direction | Base | int | Ground truth direction RSI |
| **7** | cci_direction | Base | int | Ground truth direction CCI |
| **8** | regime_pred | Stage 1 | int | ✨ Prédiction Model A (régime) |
| **9** | regime_prob_0 | Stage 1 | float | ✨ Prob régime 0 |
| **10** | regime_prob_1 | Stage 1 | float | ✨ Prob régime 1 |
| **11** | regime_prob_2 | Stage 1 | float | ✨ Prob régime 2 |
| **12** | regime_prob_3 | Stage 1 | float | ✨ Prob régime 3 |
| **13** | macd_pred | Stage 2 | int | ✨ Prédiction MACD Model |
| **14** | macd_prob | Stage 2 | float | ✨ Confiance MACD |
| **15** | rsi_pred | Stage 2 | int | ✨ Prédiction RSI Model |
| **16** | rsi_prob | Stage 2 | float | ✨ Confiance RSI |
| **17** | cci_pred | Stage 2 | int | ✨ Prédiction CCI Model |
| **18** | cci_prob | Stage 2 | float | ✨ Confiance CCI |

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
