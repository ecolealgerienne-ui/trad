# Status du Dataset de Régime - Structure X_train

**Date de mise à jour**: 2026-01-12
**Version**: 1.0 avec Pure Signal Features

## Structure Globale du Dataset

Le dataset de régime (`dataset_*_regime.npz`) contient les données suivantes:

```python
{
    'X_train': (n_train, 25, 25),    # Séquences features train
    'Y_train': (n_train, 13),        # Labels train
    'OHLCV_train': (n_train, 7),     # Prix OHLCV train
    'X_val': (n_val, 25, 25),        # Séquences features validation
    'Y_val': (n_val, 13),            # Labels validation
    'OHLCV_val': (n_val, 7),         # Prix OHLCV validation
    'X_test': (n_test, 25, 25),      # Séquences features test
    'Y_test': (n_test, 13),          # Labels test
    'OHLCV_test': (n_test, 7),       # Prix OHLCV test
    'metadata': {...}                # Métadonnées
}
```

## Structure Détaillée de X_train

### Dimensions
```
X_train: (n_train, 25, 25)
         |        |   |
         |        |   └─ 25 features par timestep
         |        └───── 25 timesteps (séquence de ~2h sur données 5min)
         └──────────── n_train samples
```

### Liste Complète des 25 Features (par timestep)

#### Colonnes 0-1: Métadonnées Temporelles
| Index | Feature | Type | Description |
|-------|---------|------|-------------|
| 0 | `timestamp` | int64 | Timestamp Unix (ms) |
| 1 | `asset_id` | int | ID de l'asset (0=BTC, 1=ETH, etc.) |

#### Colonnes 2-4: Pure Signal Features ✨ NOUVELLES
| Index | Feature | Type | Description |
|-------|---------|------|-------------|
| 2 | `h_ret` | float32 | Extension haussière: (High - Close[t-1]) / Close[t-1] |
| 3 | `l_ret` | float32 | Extension baissière: (Low - Close[t-1]) / Close[t-1] |
| 4 | `c_ret` | float32 | Rendement Close-to-Close: (Close - Close[t-1]) / Close[t-1] |

**Notes sur Pure Signal**:
- Clippés à ±10% pour éviter les outliers extrêmes
- Stationnaires (returns vs prix bruts)
- Utilisés directement pour la prédiction de direction (MACD/RSI/CCI)
- Capturent la microstructure du marché (mouvement intra-candle)

#### Colonnes 5-11: Trend Features (7 features)
| Index | Feature | Type | Description |
|-------|---------|------|-------------|
| 5 | `ma20_slope` | float32 | Pente MA20 normalisée |
| 6 | `ma50_slope` | float32 | Pente MA50 normalisée |
| 7 | `regression_slope` | float32 | Pente régression linéaire (20 périodes) |
| 8 | `regression_r2` | float32 | R² de la régression (qualité de fit) |
| 9 | `adx` | float32 | Average Directional Index (force tendance) |
| 10 | `macd_histogram_norm` | float32 | Histogramme MACD normalisé |
| 11 | `hurst_exponent` | float32 | Exposant de Hurst (persistance/mean-reversion) |

#### Colonnes 12-20: Volatility Features (9 features)
| Index | Feature | Type | Description |
|-------|---------|------|-------------|
| 12 | `atr_normalized` | float32 | Average True Range normalisé par prix |
| 13 | `bb_upper` | float32 | Bande de Bollinger supérieure |
| 14 | `bb_middle` | float32 | Bande de Bollinger moyenne (MA) |
| 15 | `bb_lower` | float32 | Bande de Bollinger inférieure |
| 16 | `bb_width` | float32 | Largeur des bandes (upper - lower) |
| 17 | `percent_b` | float32 | Position dans les bandes (0-1) |
| 18 | `realized_volatility` | float32 | Volatilité réalisée (std returns) |
| 19 | `volatility_compression` | float32 | Ratio vol courte/longue |
| 20 | `range_atr_ratio` | float32 | Ratio (High-Low)/ATR |

#### Colonnes 21-24: Volume & Microstructure Features (4 features)
| Index | Feature | Type | Description |
|-------|---------|------|-------------|
| 21 | `volume_ratio` | float32 | Volume/MA20(Volume) |
| 22 | `volume_spike` | float32 | Z-score volume (20 périodes) |
| 23 | `vwap_deviation` | float32 | Écart au VWAP normalisé |
| 24 | `obv_derivative` | float32 | Dérivée OBV normalisée |

## Structure de Y_train (Labels)

### Dimensions
```
Y_train: (n_train, 13)
         |         |
         |         └─ 13 colonnes de labels
         └─────────── n_train samples
```

### Liste des 13 Colonnes Y
| Index | Label | Type | Description |
|-------|-------|------|-------------|
| 0 | `timestamp` | int64 | Timestamp Unix (ms) |
| 1 | `asset_id` | int | ID de l'asset |
| 2 | `regime` | int | Classe de régime (0-3) |
| 3 | `trend_strength` | float32 | Force de la tendance |
| 4 | `volatility_cluster` | int | Cluster de volatilité |
| 5 | `macd_direction` | int | Direction MACD (0=DOWN, 1=UP) |
| 6 | `rsi_direction` | int | Direction RSI (0=DOWN, 1=UP) |
| 7 | `cci_direction` | int | Direction CCI (0=DOWN, 1=UP) |
| 8 | `regime_prob_0` | float32 | Probabilité régime 0 (Model A) |
| 9 | `regime_prob_1` | float32 | Probabilité régime 1 (Model A) |
| 10 | `regime_prob_2` | float32 | Probabilité régime 2 (Model A) |
| 11 | `regime_prob_3` | float32 | Probabilité régime 3 (Model A) |
| 12 | `regime_pred` | int | Prédiction Model A (classe) |

## Structure OHLCV

### Dimensions
```
OHLCV_train: (n_train, 7)
             |         |
             |         └─ 7 colonnes prix/volume
             └─────────── n_train samples
```

### Colonnes OHLCV
| Index | Colonne | Type | Description |
|-------|---------|------|-------------|
| 0 | `timestamp` | int64 | Timestamp Unix (ms) |
| 1 | `asset_id` | int | ID de l'asset |
| 2 | `open` | float32 | Prix d'ouverture |
| 3 | `high` | float32 | Prix le plus haut |
| 4 | `low` | float32 | Prix le plus bas |
| 5 | `close` | float32 | Prix de clôture |
| 6 | `volume` | float32 | Volume échangé |

## Modifications Récentes

### 2026-01-12: Ajout Pure Signal Features ✨

**Changement**: Ajout de 3 nouvelles features au début de X (après timestamp/asset_id)
- **Avant**: X = (n, 25, ~22) = [timestamp, asset_id, 20 regime features]
- **Après**: X = (n, 25, 25) = [timestamp, asset_id, h_ret, l_ret, c_ret, 20 regime features]

**Justification**:
- Combine les features complexes de régime avec les signaux purs de prix
- Permet au modèle d'accéder directement aux mouvements de prix normalisés
- Architecture hybride: régime détection + direction prédiction
- Les 3 pure signal features sont déjà utilisées avec succès dans les modèles direction-only (MACD: 92.4%, CCI: 89.3%, RSI: 87.4% accuracy)

**Fichiers modifiés**:
1. `/home/user/trad/src/regime_features.py`:
   - `get_regime_feature_names()`: Ajout de h_ret, l_ret, c_ret en tête de liste
   - `calculate_all_regime_features()`: Calcul de h_ret, l_ret, c_ret avec clipping ±10%

**Impact**:
- Augmentation de 3 colonnes dans X: 22 → 25 features
- Aucun changement dans Y (labels) ou OHLCV
- Compatible avec l'architecture CNN-LSTM existante (adaptation automatique du nombre d'input features)

## Usage pour l'Entraînement

### Chargement du Dataset
```python
import numpy as np

# Charger le dataset
data = np.load('data/prepared/dataset_*_regime.npz')

X_train = data['X_train']  # (n_train, 25, 25)
Y_train = data['Y_train']  # (n_train, 13)

# Extraire les features spécifiques
timestamps = X_train[:, :, 0]          # (n_train, 25) timestamps
asset_ids = X_train[:, :, 1]           # (n_train, 25) asset IDs
pure_signal = X_train[:, :, 2:5]       # (n_train, 25, 3) h_ret, l_ret, c_ret
regime_features = X_train[:, :, 5:]    # (n_train, 25, 20) features complexes

# Extraire les labels direction
macd_dir = Y_train[:, 5]               # (n_train,) direction MACD
rsi_dir = Y_train[:, 6]                # (n_train,) direction RSI
cci_dir = Y_train[:, 7]                # (n_train,) direction CCI
```

### Entraînement Model A (Regime Classifier)
```bash
# Préparer le dataset avec les nouvelles features
python src/prepare_data_regime.py --assets BTC ETH BNB ADA LTC

# Entraîner le modèle
python src/train_regime_classifier.py --data data/prepared/dataset_*_regime.npz
```

### Prédiction Direction (utilisant pure signal features)
Les colonnes 2-4 (h_ret, l_ret, c_ret) peuvent être utilisées pour prédire les colonnes 5-7 de Y (macd_direction, rsi_direction, cci_direction).

## Prochaines Étapes

1. **Valider le dataset régénéré** avec les nouvelles features:
   ```bash
   python tests/minimal_verify.py --data data/prepared/dataset_*_regime.npz
   ```

2. **Entraîner les modèles direction** (MACD, RSI, CCI) avec accès aux pure signal features

3. **Enrichir Y avec probabilités** des modèles direction

4. **Entraîner Model B** (Meta-Labeling) pour filtrer les trades profitables
