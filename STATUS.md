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
| `X_train` | (n_train, 25, 25) | Séquences features train (25 timesteps × 25 canaux) |
| `Y_train` | (n_train, 8) | Labels + metadata train |
| `OHLCV_train` | (n_train, 7) | Prix OHLCV + metadata train |
| `X_val` | (n_val, 25, 25) | Séquences features val |
| `Y_val` | (n_val, 8) | Labels + metadata val |
| `OHLCV_val` | (n_val, 7) | Prix OHLCV + metadata val |
| `X_test` | (n_test, 25, 25) | Séquences features test |
| `Y_test` | (n_test, 8) | Labels + metadata test |
| `OHLCV_test` | (n_test, 7) | Prix OHLCV + metadata test |

**Note**: X contient [timestamp, asset_id, ...features] donc shape (n, 25, 2 + 23 features = 25 canaux).

### Features X (25 canaux × 25 timesteps)

**Structure**: X[:, :, 0-1] = metadata, X[:, :, 2:24] = features (23 total)

| Index | Feature | Type | Description |
|-------|---------|------|-------------|
| **0** | `timestamp` | int64 | Unix timestamp (metadata) |
| **1** | `asset_id` | int | ID asset 0-4 (metadata) |
| | | | |
| | **PURE SIGNAL (3)** | | **Rendements normalisés pour combiner avec régime** |
| **2** | `h_ret` | float | (High - Close_prev) / Close_prev, clippé ±10% |
| **3** | `l_ret` | float | (Low - Close_prev) / Close_prev, clippé ±10% |
| **4** | `c_ret` | float | (Close - Close_prev) / Close_prev, clippé ±10% |
| | | | |
| | **TREND (7)** | | **Indicateurs de tendance** |
| **5** | `ma20_slope` | float | Pente MA20 normalisée |
| **6** | `ma50_slope` | float | Pente MA50 normalisée |
| **7** | `regression_slope` | float | Pente régression linéaire (20 périodes) |
| **8** | `regression_r2` | float | R² de la régression (0-1) |
| **9** | `adx` | float | ADX normalisé (force de tendance) |
| **10** | `macd_histogram_norm` | float | MACD histogram normalisé |
| **11** | `hurst_exponent` | float | Exposant de Hurst (0-1, >0.5=trending) |
| | | | |
| | **VOLATILITY (9)** | | **Indicateurs de volatilité** |
| **12** | `atr_normalized` | float | ATR / Close (volatilité relative) |
| **13** | `bb_upper` | float | Bande Bollinger supérieure normalisée |
| **14** | `bb_middle` | float | Bande Bollinger médiane normalisée |
| **15** | `bb_lower` | float | Bande Bollinger inférieure normalisée |
| **16** | `bb_width` | float | Largeur Bollinger Bands normalisée |
| **17** | `percent_b` | float | Position du prix dans les Bollinger (0-1) |
| **18** | `realized_volatility` | float | Volatilité réalisée (std returns) |
| **19** | `volatility_compression` | float | Compression/expansion volatilité |
| **20** | `range_atr_ratio` | float | Range / ATR (mesure de range) |
| | | | |
| | **VOLUME & MICROSTRUCTURE (4)** | | **Indicateurs de volume** |
| **21** | `volume_ratio` | float | Volume / MA(Volume) |
| **22** | `volume_spike` | float | Détection spike volume (Z-score) |
| **23** | `vwap_deviation` | float | (Close - VWAP) / VWAP |
| **24** | `obv_derivative` | float | Dérivée OBV normalisée |

**Total**: 2 metadata + 23 features = **25 canaux**
- Pure Signal: 3 (h_ret, l_ret, c_ret)
- Trend: 7 (MA slopes, regression, ADX, Hurst, MACD histogram)
- Volatility: 9 (ATR, Bollinger Bands, realized vol, compression, range/ATR)
- Volume: 4 (ratio, spike, VWAP deviation, OBV derivative)

**Source**: `regime_features.py` - fonction `get_regime_feature_names()`

---

## 🔬 ANALYSE DES FEATURES POUR CLASSIFICATION RÉGIME

### Rappel: Les 4 régimes à prédire

| Régime | Code | Caractéristiques |
|--------|------|------------------|
| RANGE LOW VOL | 0 | Pas de tendance + Volatilité faible |
| RANGE HIGH VOL | 1 | Pas de tendance + Volatilité haute |
| TREND LOW VOL | 2 | Tendance claire + Volatilité faible |
| TREND HIGH VOL | 3 | Tendance claire + Volatilité haute |

**Structure**: 2 dimensions = TREND vs RANGE × HIGH VOL vs LOW VOL

### Features ESSENTIELLES vs REDONDANTES

#### Pour distinguer TREND vs RANGE :

| Feature | Index | Utilité | Essentiel ? |
|---------|-------|---------|-------------|
| `adx` | 9 | Mesure directe de la force de tendance | ✅ **OUI** |
| `regression_slope` | 7 | Direction de tendance | ✅ **OUI** |
| `regression_r2` | 8 | Qualité de la tendance (linéarité) | ✅ **OUI** |
| `hurst_exponent` | 11 | Mean-reversion vs trending | ✅ **OUI** |
| `ma20_slope` | 5 | Pente MA courte | ⚠️ Redondant avec regression |
| `ma50_slope` | 6 | Pente MA longue | ⚠️ Redondant avec regression |
| `macd_histogram_norm` | 10 | Momentum | ⚠️ Redondant |

#### Pour distinguer HIGH VOL vs LOW VOL :

| Feature | Index | Utilité | Essentiel ? |
|---------|-------|---------|-------------|
| `atr_normalized` | 12 | Mesure directe de volatilité | ✅ **OUI** |
| `realized_volatility` | 18 | Volatilité historique | ✅ **OUI** |
| `bb_width` | 16 | Largeur des bandes = volatilité | ⚠️ Redondant avec ATR |
| `volatility_compression` | 19 | Ratio court/long terme | ⚠️ Utile mais secondaire |
| `bb_upper/middle/lower` | 13-15 | Niveaux absolus | ❌ Peu utile pour régime |
| `percent_b` | 17 | Position dans les bandes | ❌ Peu utile pour régime |
| `range_atr_ratio` | 20 | Range vs ATR | ⚠️ Redondant |

#### VOLUME & MICROSTRUCTURE (4 features) :

| Feature | Index | Utilité pour régime | Essentiel ? |
|---------|-------|---------------------|-------------|
| `volume_ratio` | 21 | Volume relatif | ❌ Indirect |
| `volume_spike` | 22 | Pics de volume | ❌ Indirect |
| `vwap_deviation` | 23 | Écart au VWAP | ❌ Indirect |
| `obv_derivative` | 24 | Flux de volume | ❌ Indirect |

#### PURE SIGNAL (3 features) :

| Feature | Index | Utilité pour régime | Essentiel ? |
|---------|-------|---------------------|-------------|
| `h_ret` | 2 | Rendement High | ❌ Pour direction, pas régime |
| `l_ret` | 3 | Rendement Low | ❌ Pour direction, pas régime |
| `c_ret` | 4 | Rendement Close | ❌ Pour direction, pas régime |

### Conclusion: Features MINIMALES pour régime

**Set minimal (~6 features) qui suffirait pour classifier les 4 régimes :**

```python
minimal_regime_features = [
    'adx',                    # TREND vs RANGE (force)
    'regression_r2',          # TREND vs RANGE (qualité)
    'hurst_exponent',         # TREND vs RANGE (persistance)
    'atr_normalized',         # HIGH vs LOW VOL
    'realized_volatility',    # HIGH vs LOW VOL (confirmation)
    'volatility_compression', # Transition volatilité
]
```

**Les 20 features actuelles (hors Pure Signal) sont REDONDANTES** pour la classification de régime, mais :
- ✅ Peuvent améliorer la robustesse (plusieurs perspectives)
- ✅ XGBoost peut apprendre à ignorer les features inutiles (feature importance)
- ⚠️ Risque de surapprentissage sur seulement 4 classes
- ⚠️ Volume features peu utiles pour déterminer le régime

### Usage par script

| Script | Features utilisées | Notes |
|--------|-------------------|-------|
| `train_regime_classifier.py` | 20 features (TREND + VOL + VOLUME) | Agrégées en [mean, std, min, max] → 80 features XGBoost |
| `train.py` (direction) | 23 features (inclut Pure Signal) | Séquences complètes pour CNN-LSTM |

---

## ⚠️ EXTRACTION DES FEATURES - Indices vs Noms

### Comment `train_regime_classifier.py` extrait les features

Le script utilise des **indices numériques** (pas des noms de colonnes) car les données sont en format NumPy :

```python
# Ligne 235 de train_regime_classifier.py
features = X[:, :, 2:]  # Skip timestamp (index 0) et asset_id (index 1)

# Ligne 174 - Extraction du régime
regimes_train = Y_train[:, 2].astype(int)  # Colonne 2 = regime
```

### Garantie de cohérence

**L'ordre des colonnes est garanti par la chaîne de fonctions :**

1. `regime_features.py` → `get_regime_feature_names()` définit l'ordre (lignes 703-733)
2. `prepare_data_regime.py` → utilise `get_regime_feature_names()` (ligne 606) pour construire X
3. `train_regime_classifier.py` → extrait `X[:, :, 2:]` (cohérent avec l'ordre défini)

**Code source de l'ordre des features** (`regime_features.py` lignes 703-733) :

```python
def get_regime_feature_names() -> list:
    return [
        # Pure signal features (3)
        'h_ret', 'l_ret', 'c_ret',
        # Trend features (7)
        'ma20_slope', 'ma50_slope', 'regression_slope', 'regression_r2',
        'adx', 'macd_histogram_norm', 'hurst_exponent',
        # Volatility features (9)
        'atr_normalized', 'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'percent_b',
        'realized_volatility', 'volatility_compression', 'range_atr_ratio',
        # Volume & microstructure features (4)
        'volume_ratio', 'volume_spike', 'vwap_deviation', 'obv_derivative'
    ]
```

### Pourquoi pas de sélection par nom ?

- **NumPy arrays** n'ont pas de noms de colonnes (contrairement à Pandas DataFrames)
- La cohérence est maintenue par la **fonction unique** `get_regime_feature_names()`
- Toute modification de l'ordre doit être faite dans cette fonction uniquement

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

## ⚠️ PROBLÈME CRITIQUE: Déséquilibre des Régimes

**Date d'analyse**: 2026-01-12
**Status**: 🔴 **À CORRIGER AVANT ENTRAÎNEMENT**

### Distribution Observée (Train Set)

```
Regime 0 (RANGE LOW VOL):  1,842,381 samples (65.0%)
Regime 1 (RANGE HIGH VOL):   850,983 samples (30.0%)
Regime 2 (TREND LOW VOL):      3,325 samples (0.1%)  ← PROBLÈME
Regime 3 (TREND HIGH VOL):   135,995 samples (4.8%)
```

**Problème**: Le Régime 2 représente seulement **0.1%** des données - quasi impossible à apprendre pour XGBoost.

### Cause Racine

**Seuils actuels** (dans `regime_labeler.py`) :

```python
# Trend Strength (TS)
TS_TREND_THRESHOLD = 0.6    # TS > 0.6 = TREND
TS_RANGE_THRESHOLD = 0.4    # TS < 0.4 = RANGE

# Volatility Cluster (VC)
VC_HIGH_PERCENTILE = 70     # VC > P70 = HIGH VOL
```

**Le Régime 2 (TREND LOW VOL) requiert** :
- TS > 0.6 (forte tendance)
- VC ≤ P70 (faible volatilité)

**Mais en crypto** : **quand il y a une forte tendance, il y a TOUJOURS de la volatilité élevée**.

C'est la nature du marché - les grandes tendances crypto sont accompagnées de mouvements de prix importants.

### Corrélation TS ↔ VC

| Condition | Résultat Typique |
|-----------|------------------|
| RANGE (TS < 0.4) | Généralement LOW VOL |
| TREND (TS > 0.6) | **Presque toujours HIGH VOL** |

→ TREND + LOW VOL est **structurellement rare** (0.1%)

### Impact sur l'Entraînement

| Problème | Conséquence |
|----------|-------------|
| 0.1% de Régime 2 | XGBoost ne peut pas apprendre ce régime |
| Classe sous-représentée | Modèle prédit jamais Régime 2 |
| Metrics faussées | Accuracy artificielle haute (ignorer 2) |
| Généralisation | Échec sur vrais cas TREND LOW VOL |

### Solutions Proposées

#### 🅰️ Solution A: Ajuster les seuils

**Objectif** : Obtenir au moins 5% par régime

```python
# Nouveaux seuils (à tester)
TS_TREND_THRESHOLD = 0.50  # Moins strict (était 0.6)
TS_RANGE_THRESHOLD = 0.40  # Inchangé
VC_HIGH_PERCENTILE = 80    # Plus strict (était 70)
```

**Avantage** : Conserve les 4 régimes
**Risque** : Peut ne pas suffire (corrélation naturelle TS-VC)

#### 🅱️ Solution B: Merger en 3 classes

**Nouveaux régimes** :
- Régime 0: RANGE LOW VOL
- Régime 1: RANGE HIGH VOL
- Régime 2: **TREND** (fusionne LOW et HIGH VOL)

**Avantage** : Distribution plus équilibrée (~65% / 30% / 5%)
**Inconvénient** : Perd la distinction volatilité en TREND

#### 🅲️ Solution C: Class Weights dans XGBoost

```python
# Dans train_regime_classifier.py
class_weights = {0: 1.0, 1: 2.0, 2: 500.0, 3: 13.0}  # Inverse de la fréquence
```

**Avantage** : Pas de modification des labels
**Risque** : Peut créer de l'instabilité (poids ×500 pour Régime 2)

### Recommandation

**🅰️ Solution A (ajuster seuils)** est la meilleure approche car :
1. Préserve les 4 régimes distincts (meilleure granularité)
2. Modification simple dans `regime_labeler.py`
3. Seuils plus permissifs reflètent mieux la réalité crypto

**Fichier à modifier** : `src/regime_labeler.py` (lignes 77-80)

```python
# AVANT (seuils actuels)
TS_TREND_THRESHOLD = 0.6    # TS > 0.6 = TREND
TS_RANGE_THRESHOLD = 0.4    # TS < 0.4 = RANGE
VC_HIGH_PERCENTILE = 70     # VC > P70 = HIGH VOL

# APRÈS (seuils ajustés)
TS_TREND_THRESHOLD = 0.50   # Plus permissif pour TREND
TS_RANGE_THRESHOLD = 0.40   # Inchangé
VC_HIGH_PERCENTILE = 80     # Plus strict pour HIGH VOL
```

### Action Requise

⏳ **En attente** : Tester différentes combinaisons de seuils et relancer `prepare_data_regime.py` pour obtenir une distribution plus équilibrée (minimum 5% par régime).

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
