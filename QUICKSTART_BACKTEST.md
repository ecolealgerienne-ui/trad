# Quick Start - Backtest Dual-Binary

## 🚀 Test Rapide (Oracle)

Tester la stratégie simple avec labels parfaits (monde idéal) :

```bash
# Test MACD (meilleur indicateur 86.9%)
python tests/test_dual_binary_trading.py --indicator macd --split test --fees 0.1
```

**Attendu** :
- Trades: ~20-30k (vs ~100k sans Force filtering)
- Win Rate: ~50-55%
- PnL Net: **Fortement positif** si edge existe

---

## 📋 Workflow Complet

### Étape 1 : Préparer les Données

```bash
# Générer les 3 datasets dual-binary (RSI, MACD, CCI)
python src/prepare_data_purified_dual_binary.py --assets BTC ETH BNB ADA LTC
```

**Durée** : ~2-5 min
**Output** : 3 fichiers .npz dans `data/prepared/`

### Étape 2 : Test Oracle (Baseline Performance)

```bash
# Test avec labels parfaits (Oracle)
python tests/test_dual_binary_trading.py --indicator macd --split test --fees 0.1
python tests/test_dual_binary_trading.py --indicator cci --split test --fees 0.1
python tests/test_dual_binary_trading.py --indicator rsi --split test --fees 0.1
```

**Objectif** : Vérifier que la stratégie simple est profitable en mode parfait.

### Étape 3 : Entraîner les Modèles (Optionnel)

```bash
# Entraîner les 3 modèles (config auto-détectée)
python src/train.py --data data/prepared/dataset_btc_eth_bnb_ada_ltc_macd_dual_binary_kalman.npz --epochs 50
python src/train.py --data data/prepared/dataset_btc_eth_bnb_ada_ltc_cci_dual_binary_kalman.npz --epochs 50
python src/train.py --data data/prepared/dataset_btc_eth_bnb_ada_ltc_rsi_dual_binary_kalman.npz --epochs 50
```

**Durée** : ~10-30 min par modèle (GPU)
**Output** : `models/best_model_*_kalman_dual_binary.pth`

### Étape 4 : Générer Prédictions

```bash
# Évaluer et sauvegarder prédictions dans .npz
python src/evaluate.py --data data/prepared/dataset_btc_eth_bnb_ada_ltc_macd_dual_binary_kalman.npz
python src/evaluate.py --data data/prepared/dataset_btc_eth_bnb_ada_ltc_cci_dual_binary_kalman.npz
python src/evaluate.py --data data/prepared/dataset_btc_eth_bnb_ada_ltc_rsi_dual_binary_kalman.npz
```

**Durée** : ~30 sec par modèle
**Output** : Y_pred ajouté dans les .npz

### Étape 5 : Test Modèle (Performance Réelle)

```bash
# Test avec prédictions modèle
python tests/test_dual_binary_trading.py --indicator macd --split test --use-predictions --fees 0.1
python tests/test_dual_binary_trading.py --indicator cci --split test --use-predictions --fees 0.1
python tests/test_dual_binary_trading.py --indicator rsi --split test --use-predictions --fees 0.1
```

**Objectif** : Mesurer performance réelle (accuracy 80-87%).

---

## 🎯 Résultats Attendus (MACD)

### Mode Oracle (Labels Parfaits)

```
Total Trades:     ~22,000 (-78% vs sans Force)
Win Rate:         ~55% (+13% vs sans Force)
Profit Factor:    ~1.15
PnL Net:          +1300% sur test set
```

### Mode Modèle (Accuracy 86.9%)

```
Total Trades:     ~22,000
Win Rate:         ~48%
Profit Factor:    ~1.08
PnL Net:          +500-800% sur test set
```

**Gap Oracle/Modèle** : ~500% (marge d'amélioration via optimisations)

---

## ⚙️ Options Avancées

### Tester Différents Splits

```bash
# Train set (in-sample)
python tests/test_dual_binary_trading.py --indicator macd --split train --fees 0.1

# Validation set
python tests/test_dual_binary_trading.py --indicator macd --split val --fees 0.1

# Test set (out-of-sample)
python tests/test_dual_binary_trading.py --indicator macd --split test --fees 0.1
```

### Tester Différents Frais

```bash
# Frais faibles (Maker: 0.02%)
python tests/test_dual_binary_trading.py --indicator macd --split test --fees 0.02

# Frais standards (Taker: 0.1%)
python tests/test_dual_binary_trading.py --indicator macd --split test --fees 0.1

# Frais élevés (Slippage: 0.2%)
python tests/test_dual_binary_trading.py --indicator macd --split test --fees 0.2
```

---

## 📊 Comparaison Indicateurs

Comparer les 3 indicateurs côte à côte :

```bash
# Script batch (Linux/Mac)
for indicator in macd cci rsi; do
    echo "===== Testing $indicator ====="
    python tests/test_dual_binary_trading.py --indicator $indicator --split test --fees 0.1
done
```

**Classement attendu** (Oracle) :
1. 🥇 **MACD** : 86.9% accuracy, meilleur PnL
2. 🥈 **CCI** : 83.3% accuracy, bon équilibre
3. 🥉 **RSI** : 80.7% accuracy, ultra-sélectif

---

## 🐛 Troubleshooting

### Erreur : Dataset introuvable

```
FileNotFoundError: data/prepared/dataset_btc_eth_bnb_ada_ltc_macd_dual_binary_kalman.npz
```

**Solution** :
```bash
python src/prepare_data_purified_dual_binary.py --assets BTC ETH BNB ADA LTC
```

### Erreur : Prédictions non disponibles

```
⚠️ Prédictions non disponibles (utiliser --use-predictions après entraînement)
```

**Solution** :
```bash
# 1. Entraîner
python src/train.py --data data/prepared/dataset_..._macd_dual_binary_kalman.npz --epochs 50

# 2. Évaluer (sauvegarde Y_pred)
python src/evaluate.py --data data/prepared/dataset_..._macd_dual_binary_kalman.npz

# 3. Re-tester
python tests/test_dual_binary_trading.py --indicator macd --split test --use-predictions --fees 0.1
```

### PnL Négatif en Mode Oracle

Si PnL Oracle < 0 :
- ❌ La stratégie simple n'a pas d'edge
- 🔍 Vérifier distribution labels (Direction 50-50, Force 30-33%)
- 🔍 Vérifier frais (peut-être trop élevés pour fréquence de trading)

### Trop de Trades (>50k)

Si trades > 50k :
- 🔍 Vérifier que Force filtering fonctionne
- 🔍 Vérifier distribution Force (devrait être ~30% STRONG)
- 🔍 Ajouter hysteresis (prochaine étape)

---

## 📚 Documentation Complète

- **`docs/BACKTEST_DUAL_BINARY.md`** : Guide détaillé
- **`CLAUDE.md`** : Architecture complète v7.1
- **`src/prepare_data_purified_dual_binary.py`** : Code préparation données
- **`tests/test_dual_binary_trading.py`** : Code backtest

---

## ✅ Checklist

Avant de lancer les tests :

- [ ] Données préparées (`prepare_data_purified_dual_binary.py` exécuté)
- [ ] 3 fichiers .npz présents dans `data/prepared/`
- [ ] Script backtest vérifié (`python -m py_compile tests/test_dual_binary_trading.py`)
- [ ] GPU disponible (si entraînement modèles)

C'est parti ! 🚀
