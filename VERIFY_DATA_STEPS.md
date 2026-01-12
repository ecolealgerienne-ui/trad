# 🔍 Vérification des Données - Guide Complet

**Date**: 2026-01-12
**Dataset**: `data/prepared/dataset_btc_eth_bnb_ada_ltc_regime.npz` (547M)

## ⚠️ PRÉREQUIS: Installer les dépendances Python

```bash
# 1. Vérifier l'environnement Python actuel
python --version
which python

# 2. Installer les dépendances depuis requirements.txt
pip install -r requirements.txt

# OU installer uniquement les packages critiques pour la vérification:
pip install numpy pandas scipy matplotlib seaborn ta pykalman torch scikit-learn xgboost
```

## 📋 ÉTAPE 1: Inspection Rapide du NPZ (Clés et Shapes)

```bash
# Script le plus simple - affiche les clés et shapes
python tests/inspect_npz.py data/prepared/dataset_btc_eth_bnb_ada_ltc_regime.npz
```

**Output attendu**:
```
Fichier: data/prepared/dataset_btc_eth_bnb_ada_ltc_regime.npz

Clés disponibles (XX total):
  OHLCV_test                    shape=(n_test, 7), dtype=float64
  OHLCV_train                   shape=(n_train, 7), dtype=float64
  OHLCV_val                     shape=(n_val, 7), dtype=float64
  X_test                        shape=(n_test, 25, ~22), dtype=float64
  X_train                       shape=(n_train, 25, ~22), dtype=float64
  X_val                         shape=(n_val, 25, ~22), dtype=float64
  Y_test                        shape=(n_test, 13), dtype=float64
  Y_train                       shape=(n_train, 13), dtype=float64
  Y_val                         shape=(n_val, 13), dtype=float64
  metadata                      type=<class 'numpy.ndarray'>
```

**⚠️ Vérifications critiques**:
- X shape: `(n, 25, ~22)` - 25 timesteps, ~22 features
- Y shape: `(n, 13)` - 13 colonnes après enrichissement
- OHLCV shape: `(n, 7)` - [timestamp, asset_id, O, H, L, C, V]

## 📊 ÉTAPE 2: Diagnostic Détaillé

```bash
# Affiche statistiques, distributions, NaN/Inf
python tests/diagnose_dataset.py \
    --data data/prepared/dataset_btc_eth_bnb_ada_ltc_regime.npz
```

**Vérifications effectuées**:
1. ✅ Shapes de X, Y, OHLCV
2. ✅ Contenu des features (première séquence)
3. ✅ Asset IDs uniques (doit être 0-4 pour 5 assets)
4. ✅ Distribution des labels régime (colonnes 2-7 de Y)
5. ✅ NaN/Inf dans X et Y
6. ✅ Nombre de séquences par asset
7. ✅ Metadata (features, assets, labels)

## 🔬 ÉTAPE 3: Vérification Complète (9 checks)

```bash
# Vérification la plus complète - 9 aspects vérifiés
python tests/verify_regime_dataset.py \
    --data data/prepared/dataset_btc_eth_bnb_ada_ltc_regime.npz \
    --verbose
```

**Vérifications effectuées** (voir `tests/verify_regime_dataset.py` lignes 9-18):
1. ✅ **Shapes des arrays** (X, Y, OHLCV)
2. ✅ **Timestamps** (croissants, pas de doublons, gaps entre splits)
3. ✅ **Asset IDs** (valides 0-4)
4. ✅ **Labels régime** (0-3, distributions, TS/VC scores cohérents)
5. ✅ **Features** (~20 attendues, pas de NaN/Inf, ranges valides)
6. ✅ **OHLCV** (cohérence prix O/H/L/C, volume > 0)
7. ✅ **Primary key** (timestamp, asset_id) synchronisé entre X/Y/OHLCV
8. ✅ **Metadata** (split_indices, features, cohérence)
9. ✅ **Causalité temporelle** (pas de lookahead)

**Output attendu**: Rapport détaillé avec ✅ ou ❌ pour chaque vérification

## 🐛 PROBLÈMES COURANTS À RECHERCHER

### A. Problèmes de Shape/Structure

```python
# Vérifier si Y a bien 13 colonnes (8 base + 5 enrichissement)
# Y devrait avoir: [timestamp, asset_id, regime, trend_strength, volatility,
#                   macd_dir, rsi_dir, cci_dir, regime_prob_0-3, regime_pred]

# Si Y a seulement 8 colonnes → train_regime_classifier.py pas exécuté
# Si Y a 13 colonnes → ✅ OK
```

### B. Problèmes de Valeurs

```python
# NaN/Inf dans X ou Y → problème de calcul features
# Asset IDs hors range 0-4 → problème de préparation
# Régimes hors range 0-3 → problème de labeling
# Timestamps non croissants → problème de tri
# Volume = 0 → données OHLCV corrompues
```

### C. Problèmes de Synchronisation

```python
# X[i], Y[i], OHLCV[i] doivent avoir même timestamp et asset_id
# Si pas synchronisés → causality leak ou data misalignment
```

### D. Problèmes de Distributions

```python
# Régimes équilibrés? (chaque régime 0-3 devrait avoir ~20-30%)
# Si un régime < 5% ou > 50% → problème de thresholds
# trend_strength et volatility entre 0-1?
# Directions (macd/rsi/cci) binaires 0/1?
```

## 🔧 SI PROBLÈMES DÉTECTÉS

### Option 1: Régénérer le dataset base (ÉTAPE 1)

```bash
# Si problèmes dans X ou Y colonnes 0-7
python src/prepare_data_regime.py \
    --assets BTC ETH BNB ADA LTC \
    --output data/prepared/dataset_btc_eth_bnb_ada_ltc_regime.npz
```

### Option 2: Régénérer l'enrichissement (ÉTAPE 2)

```bash
# Si problèmes dans Y colonnes 8-12 (régime probs)
python src/train_regime_classifier.py \
    --data data/prepared/dataset_btc_eth_bnb_ada_ltc_regime.npz \
    --epochs 100
```

### Option 3: Vérifier les features individuellement

```bash
# Script pour vérifier normalisation et ranges des features
python tests/check_features_normalization.py \
    --data data/prepared/dataset_btc_eth_bnb_ada_ltc_regime.npz
```

## 📈 APRÈS VÉRIFICATION: Entraîner les modèles

Une fois les données validées:

```bash
# Entraîner modèle MACD (exemple)
python src/train.py \
    --data data/prepared/dataset_btc_eth_bnb_ada_ltc_regime.npz \
    --indicator macd \
    --epochs 30 \
    --batch-size 64 \
    --lr 0.0001
```

## 🎯 CHECKLIST DE VALIDATION

- [ ] Dependencies Python installées (`pip install -r requirements.txt`)
- [ ] `inspect_npz.py` exécuté → shapes correctes
- [ ] `diagnose_dataset.py` exécuté → pas de NaN/Inf
- [ ] `verify_regime_dataset.py` exécuté → toutes vérifications ✅
- [ ] Y a bien 13 colonnes (enrichissement OK)
- [ ] Distributions régimes équilibrées (~20-30% chacune)
- [ ] Timestamps synchronisés entre X/Y/OHLCV
- [ ] Prêt pour entraînement des modèles direction

## 📝 NOTES IMPORTANTES

1. **Dataset universel** : `dataset_*_regime.npz` contient TOUS les indicateurs (MACD, RSI, CCI directions dans Y colonnes 5-7)

2. **Enrichissement in-place** : train_regime_classifier.py enrichit Y de 8 → 13 colonnes SANS créer nouveau fichier

3. **X features** : ~20 features complexes (trend, volatility, volume) - PAS de simples returns

4. **Backup** : `dataset_*_regime_original.npz` (461M) créé automatiquement avant enrichissement
