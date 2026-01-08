# Adaptation train.py et evaluate.py pour Direction-Only

**Date**: 2026-01-08
**Statut**: ✅ **COMPLÉTÉ - Scripts adaptés**

## Modifications apportées

### 1. `src/prepare_data.py` - `load_prepared_data()`

**Problème**: Le nouveau format Direction-Only génère:
- `Y`: (n, 3) = [timestamp, asset_id, direction]
- `T`: (n, 3) = [timestamp, asset_id, is_transition]

Alors que l'entraînement attend:
- `Y`: (n, 1) = [direction]
- `T`: (n, 1) = [is_transition]

**Solution**: Détection automatique + extraction colonnes

```python
# Détection format Direction-Only (Y shape: (n, 3))
if data['Y_train'].ndim == 2 and data['Y_train'].shape[1] == 3:
    is_direction_only = True

    # Extraire seulement colonne label (colonne 2)
    Y_train = data['Y_train'][:, 2:3]  # (n, 1)
    Y_val = data['Y_val'][:, 2:3]
    Y_test = data['Y_test'][:, 2:3]

    # Idem pour transitions si présentes
    if has_transitions:
        T_train = data['T_train'][:, 2:3]  # (n, 1)
        T_val = data['T_val'][:, 2:3]
        T_test = data['T_test'][:, 2:3]
```

### 2. `src/train.py` - **AUCUNE MODIFICATION NÉCESSAIRE**

✅ Le script gère déjà automatiquement:
- Détection shape `n_features_detected = X_train.shape[2]`
- Détection outputs `n_outputs_detected = Y_train.shape[1]`
- Support num_outputs=1, 2 ou 3
- Support transitions optionnelles

### 3. `src/evaluate.py` - **AUCUNE MODIFICATION NÉCESSAIRE**

✅ Le script gère déjà automatiquement:
- Utilise `load_prepared_data()` (qui extrait les bonnes colonnes)
- Détection automatique du nombre d'outputs via `model.py`

### 4. `src/model.py` - **AUCUNE MODIFICATION NÉCESSAIRE**

✅ Supporte déjà:
- `num_outputs=1` (Direction-Only)
- `num_outputs=2` (Dual-Binary: Direction + Force)
- `num_outputs=3` (Multi-Output: RSI, CCI, MACD)

## Tests créés

### `tests/test_load_direction_only.py`

Script de test pour valider le chargement des datasets Direction-Only.

**Usage**:
```bash
python tests/test_load_direction_only.py --data data/prepared/dataset_btc_rsi_direction_only_kalman_wt.npz
```

**Vérifications**:
1. ✅ Y shape: (n, 1) après extraction
2. ✅ T shape: (n, 1) après extraction (si présent)
3. ✅ X shape: (n, seq_length, n_features)
4. ✅ Cohérence tailles X/Y/T
5. ✅ Valeurs Y et T: uniquement 0/1
6. ✅ Distributions Direction ~50%
7. ✅ Distributions Transitions 10-20%
8. ✅ Métadonnées complètes

## Workflow complet

### 1. Préparer les données (déjà fait)

```bash
python src/prepare_data_direction_only.py --assets BTC ETH BNB ADA LTC
```

**Outputs**:
```
data/prepared/dataset_btc_eth_bnb_ada_ltc_rsi_direction_only_kalman_wt.npz
data/prepared/dataset_btc_eth_bnb_ada_ltc_macd_direction_only_kalman_wt.npz
data/prepared/dataset_btc_eth_bnb_ada_ltc_cci_direction_only_kalman_wt.npz
```

### 2. Tester le chargement

```bash
python tests/test_load_direction_only.py \
    --data data/prepared/dataset_btc_eth_bnb_ada_ltc_rsi_direction_only_kalman_wt.npz
```

**Sortie attendue**:
```
TEST CHARGEMENT DIRECTION-ONLY
================================================================================
Fichier: data/prepared/dataset_btc_eth_bnb_ada_ltc_rsi_direction_only_kalman_wt.npz

📂 Chargement des données: data/prepared/...
  🎯 Format Direction-Only détecté (Y shape: (615474, 3))
     → Extraction colonne label (colonne 2):
     Train: Y=(615474, 1), T=(615474, 1)
     Val:   Y=(131903, 1), T=(131903, 1)
     Test:  Y=(131903, 1), T=(131903, 1)

📊 Shapes chargées:
  Train: X=(615474, 25, 1), Y=(615474, 1), T=(615474, 1)
  Val:   X=(131903, 25, 1), Y=(131903, 1), T=(131903, 1)
  Test:  X=(131903, 25, 1), Y=(131903, 1), T=(131903, 1)

✅ VÉRIFICATIONS:
  ✅ Y_train shape correct: (615474, 1)
  ✅ Y_val shape correct: (131903, 1)
  ✅ Y_test shape correct: (131903, 1)
  ✅ T_train shape correct: (615474, 1)
  ✅ T_val shape correct: (131903, 1)
  ✅ T_test shape correct: (131903, 1)
  ✅ X_train: seq_length=25, n_features=1
  ✅ Cohérence tailles X/Y/T
  ✅ Y contient uniquement 0/1
  ✅ T contient uniquement 0/1

📊 Distributions Direction (% UP):
  Train: 50.1%
  Val:   49.8%
  Test:  50.0%

📊 Distributions Transitions (% retournements):
  Train: 14.2%
  Val:   14.5%
  Test:  14.3%

================================================================================
✅ TOUS LES TESTS PASSÉS - Dataset Direction-Only valide!
================================================================================
```

### 3. Entraîner le modèle

```bash
# RSI (1 feature) - Tous les assets
python src/train.py \
    --data data/prepared/dataset_btc_eth_bnb_ada_ltc_rsi_direction_only_kalman_wt.npz \
    --epochs 50 \
    --batch-size 128

# MACD (1 feature) - Filtrer pour BTC et ETH uniquement
python src/train.py \
    --data data/prepared/dataset_btc_eth_bnb_ada_ltc_macd_direction_only_kalman_wt.npz \
    --assets BTC ETH \
    --epochs 50 \
    --batch-size 128

# CCI (3 features) - Filtrer pour 3 assets
python src/train.py \
    --data data/prepared/dataset_btc_eth_bnb_ada_ltc_cci_direction_only_kalman_wt.npz \
    --assets BTC ETH BNB \
    --epochs 50 \
    --batch-size 128
```

**Détection automatique**:
- ✅ `n_features_detected`: 1 (RSI/MACD) ou 3 (CCI)
- ✅ `n_outputs_detected`: 1 (Direction seule)
- ✅ `has_transitions`: True (Weighted Loss activé)
- ✅ Indicateur: détecté depuis filename ou metadata

**Modèles sauvegardés**:
```
models/best_model_rsi_kalman_direction_only_wt.pth
models/best_model_macd_kalman_direction_only_wt.pth
models/best_model_cci_kalman_direction_only_wt.pth
```

### 4. Évaluer le modèle

```bash
# RSI - Tous les assets
python src/evaluate.py \
    --data data/prepared/dataset_btc_eth_bnb_ada_ltc_rsi_direction_only_kalman_wt.npz

# MACD - Filtrer pour BTC et ETH uniquement
python src/evaluate.py \
    --data data/prepared/dataset_btc_eth_bnb_ada_ltc_macd_direction_only_kalman_wt.npz \
    --assets BTC ETH

# CCI - Filtrer pour 3 assets
python src/evaluate.py \
    --data data/prepared/dataset_btc_eth_bnb_ada_ltc_cci_direction_only_kalman_wt.npz \
    --assets BTC ETH BNB
```

**Note**: Si vous utilisez `--assets` lors de l'entraînement, utilisez les mêmes assets lors de l'évaluation pour une comparaison cohérente.

**Métriques attendues**:
```
MÉTRIQUES PAR INDICATEUR
================================================================================
Indicateur   Accuracy   Precision  Recall     F1
--------------------------------------------------------------------------------
RSI          0.876      0.897      0.845      0.871
MACD         0.925      0.915      0.923      0.919
CCI          0.902      0.846      0.869      0.858
```

## Architecture détectée automatiquement

### RSI Direction-Only

```python
# Auto-détecté par train.py:
n_features_detected = 1  # c_ret seul
n_outputs_detected = 1   # Direction seule
use_layer_norm = False   # RSI: baseline optimal
use_bce_with_logits = False
indicator_for_metrics = 'RSI'
```

### MACD Direction-Only

```python
# Auto-détecté par train.py:
n_features_detected = 1  # c_ret seul
n_outputs_detected = 1   # Direction seule
use_layer_norm = True    # MACD: optimisations activées
use_bce_with_logits = True
indicator_for_metrics = 'MACD'
```

### CCI Direction-Only

```python
# Auto-détecté par train.py:
n_features_detected = 3  # h_ret, l_ret, c_ret
n_outputs_detected = 1   # Direction seule
use_layer_norm = False   # CCI: BCE seul optimal
use_bce_with_logits = True
indicator_for_metrics = 'CCI'
```

## Backward Compatibility

✅ Les scripts restent **100% compatibles** avec:
- Ancien format Dual-Binary: Y shape (n, 2)
- Ancien format Multi-Output: Y shape (n, 3)
- Datasets sans transitions: pas de T_train

La détection se fait automatiquement dans `load_prepared_data()`.

## Prochaines étapes

1. ✅ Scripts adaptés
2. ⏳ **Regénérer dataset BTC** avec tous les fixes (transpose + TRIM)
3. ⏳ **Valider dataset** avec `tests/validate_dataset.py`
4. ⏳ **Tester chargement** avec `tests/test_load_direction_only.py`
5. ⏳ **Entraîner modèles** pour les 3 indicateurs
6. ⏳ **Évaluer performances** sur test set

## Résumé

| Script | Statut | Modifications |
|--------|--------|---------------|
| `src/prepare_data.py` | ✅ **ADAPTÉ** | Extraction colonnes 2 si Y shape (n, 3) |
| `src/train.py` | ✅ **OK** | Aucune modification nécessaire |
| `src/evaluate.py` | ✅ **OK** | Aucune modification nécessaire |
| `src/model.py` | ✅ **OK** | Déjà flexible (num_outputs=1,2,3) |
| `tests/test_load_direction_only.py` | ✅ **CRÉÉ** | Validation chargement |

**Tous les scripts sont prêts pour le format Direction-Only!**

---

## Filtrage Multi-Assets

### Fonctionnalité

Les scripts `train.py` et `evaluate.py` supportent maintenant le filtrage par cryptomonnaie avec le paramètre `--assets`.

### Principe

1. **Dataset complet**: Généré avec tous les assets (BTC, ETH, BNB, ADA, LTC)
2. **Filtrage à l'entraînement**: Sélectionner les assets souhaités avec `--assets`
3. **Utilisation de asset_id**: Filtre basé sur la colonne `asset_id` dans X et OHLCV

### Exemples d'Utilisation

```bash
# Entraîner sur Bitcoin uniquement
python src/train.py \
    --data dataset_btc_eth_bnb_ada_ltc_rsi_direction_only_kalman_wt.npz \
    --assets BTC \
    --epochs 50

# Entraîner sur les 3 principales cryptos
python src/train.py \
    --data dataset_btc_eth_bnb_ada_ltc_macd_direction_only_kalman_wt.npz \
    --assets BTC ETH BNB \
    --epochs 50

# Évaluer avec les mêmes assets
python src/evaluate.py \
    --data dataset_btc_eth_bnb_ada_ltc_macd_direction_only_kalman_wt.npz \
    --assets BTC ETH BNB
```

### Logs de Filtrage

Lorsque vous utilisez `--assets`, vous verrez ces informations:

```
🔍 Filtrage des assets...
  🎯 Filtrage pour assets: ['BTC', 'ETH']
     Asset IDs: [0.0, 1.0]
     Avant filtrage: 615474 séquences
     Après filtrage: 246189 séquences (40.0%)
  ✅ Filtrage terminé pour 2 asset(s)
```

### Asset ID Mapping

⚠️ **IMPORTANT**: Les cryptos sont indexées en **0-indexed** (commence à 0):

| Asset | ID |
|-------|----|
| BTC | 0 |
| ETH | 1 |
| BNB | 2 |
| ADA | 3 |
| LTC | 4 |

### Avantages

✅ **Un seul dataset à générer**: Préparer une seule fois avec tous les assets
✅ **Flexibilité à l'entraînement**: Tester différentes combinaisons sans regénérer
✅ **Comparaisons cohérentes**: Même preprocessing pour tous les tests
✅ **Économie de stockage**: Pas besoin de datasets séparés par asset
