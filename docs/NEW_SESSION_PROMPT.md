# 🚀 Prompt Nouvelle Session - Trading ML

**Date**: 2026-01-10
**Version**: 10.0 - Phase 2.15: Nouvelle Formule Labels (t vs t-1)
**Branch Git**: `claude/review-context-update-main-844S0`

---

## 📋 Contexte à Charger

Bonjour Claude,

Je continue le projet **CNN-LSTM Direction-Only** pour prédiction de tendance crypto. **Lis d'abord `/home/user/trad/CLAUDE.md`** pour le contexte complet.

---

## 🎯 État Actuel du Projet

### Modèles Entraînés (Test Accuracy)

| Indicateur | Accuracy | Config | Rôle |
|------------|----------|--------|------|
| **MACD** | **92.4%** 🥇 | Kalman, baseline | **Indicateur PIVOT** |
| **CCI** | 88.6% 🥈 | Kalman + Shortcut s=2 | Modulateur |
| **RSI** | 87.6% 🥉 | Kalman, baseline | Modulateur |

### 🎉 Phase 2.15 (VALIDÉE): Nouvelle Formule Labels - SUCCÈS TOTAL

**TRANSFORMATION MAJEURE - Win Rate × Win Rate**

| Aspect | AVANT (t-2 vs t-3) | APRÈS (t vs t-1) | Gain |
|--------|-------------------|------------------|------|
| **Formule** | `filtered[t-2] > filtered[t-3]` | `filtered[t] > filtered[t-1]` | - |
| **Signal** | Pente passée (décalée -2) | **Pente immédiate** | Réactivité ×2 |
| **Win Rate** | ~33% | **53-57%** ✅ | **+20-24%** |
| **PnL Net** | **NÉGATIF** ❌ | **+14k-23k%** ✅ | Transformation |
| **ML Accuracy** | 92.4% (MACD) | 81.1% (MACD) | -11% (sacrifié) |

**Commit**: `b1490e6` - Script modifié: `src/prepare_data_direction_only.py`

**Résultats Oracle (Test Set, 640k samples):**

| Indicateur | PnL Net | Win Rate | Profit Factor | Sharpe |
|------------|---------|----------|---------------|--------|
| **RSI** 🥇 | **+23,039%** | 57.3% | 4.02 | 102.67 |
| **CCI** 🥈 | **+17,335%** | 56.4% | 3.16 | 87.55 |
| **MACD** 🥉 | **+14,359%** | 53.4% | 2.79 | 85.44 |

**Découverte Majeure:**
> **Timing d'entrée > ML Accuracy**
>
> Sacrifice ML accuracy (92%→81%) justifié par Win Rate (+20%) et PnL transformé

### Découverte Majeure - Phase 2.13

**RSI, CCI, MACD capturent le MÊME signal latent!**
- Corrélation Oracle = **1.000** (identiques)
- 80.6% des erreurs ML sont partagées
- Fusion/voting = **INUTILE** (0% gain prouvé)

### Résultats Phase 2.14 (Terminée)

**Test**: Entrée pondérée ML + Sortie Oracle (labels parfaits)
**Script**: `tests/test_entry_oracle_exit.py`

| Oracle Exit | Trades | Win Rate | PnL Gross | PnL Net | Durée |
|-------------|--------|----------|-----------|---------|-------|
| **MACD** 🥇 | 13,444 | 22.1% | +607% | **-2,082%** | 8.4p |
| **CCI** 🥈 | 15,248 | 20.2% | +667% | -2,382% | 6.8p |
| **RSI** 🥉 | 17,026 | 19.3% | +768% | -2,638% | 5.8p |

**Configuration optimale**: ThLong=0.8, ThShort=0.2, w_MACD=0.8

---

## ❌ Problème Fondamental Non Résolu

**Même avec sortie Oracle PARFAITE, PnL Net reste NÉGATIF!**

```
Signal MACD:  +607% brut
Trades:       13,444
Frais:        13,444 × 0.2% × 2 = 5,378%
PnL Net:      -2,082% (frais > signal)
```

**Le problème = TROP DE TRADES**, pas le signal (qui fonctionne).

---

## 🎯 Prochaines Étapes (Après Phase 2.15)

**Contexte**: Nouvelle formule (t vs t-1) transforme PnL Net négatif → +14k-23k% ✅

**NOUVEAU PARADIGME**: Win Rate ≥ 50% (validé: 53-57%) > ML Accuracy

### Option 1: Tester ML Predictions (pas Oracle) ⭐ (PRIORITÉ)
- Oracle: Win Rate 53-57% ✅
- ML à vérifier: Accuracy 81% → Win Rate ?
- Si Win Rate ML ≥ 45%, **SUCCÈS PRODUCTION**

### Option 2: Réentraîner avec Shortcut steps=2
- Nouvelle formule (t vs t-1) aligne Shortcut avec label
- Shortcut accède à [t-2, t-1], label compare t vs t-1
- Gain potentiel: +1-3% Win Rate

### Option 3: Timeframe 15min/30min
- Réduction naturelle trades ÷3 à ÷6
- Signal plus stable, moins de bruit
- Maintenir Win Rate 50%+

### Option 4: Focus Asset ADA
- ADA: Meilleur asset (+6,475% moyen sur 3 indicateurs)
- Test ML predictions sur ADA uniquement
- Si validé, étendre aux autres assets

---

## 🚫 Approches qui ont ÉCHOUÉ (Ne Pas Retester)

| Approche | Résultat | Raison |
|----------|----------|--------|
| Fusion multi-indicateurs | -15% à -43% | Corrélation 100% |
| Vote majoritaire | 0% gain | Mêmes erreurs |
| Force filter STRONG/WEAK | -354% à -800% | Non prédictif |
| ATR filter | Neutre | Flickering bypass |
| Kalman sliding window | -19% à -30% | Lag détruit signal |
| Octave sliding window | -37% à -116% | Pire que Kalman |
| Weighted probability fusion | Tous négatifs | Amplifie bruit |
| Stacking/Ensemble | -3% à -12% | Erreurs corrélées |

---

## 📁 Datasets Disponibles

```
data/prepared/dataset_btc_eth_bnb_ada_ltc_macd_direction_only_kalman.npz
data/prepared/dataset_btc_eth_bnb_ada_ltc_rsi_direction_only_kalman.npz
data/prepared/dataset_btc_eth_bnb_ada_ltc_cci_direction_only_kalman.npz
```

**Structure**:
- **X**: (n, 25, features) - séquences 25 timesteps
- **Y**: (n, 3) - [timestamp, asset_id, direction]
- **OHLCV**: (n, 7) - [timestamp, asset_id, O, H, L, C, V]
- **Y_*_pred**: Prédictions ML (probabilités 0-1)

**Assets**: BTC=0, ETH=1, BNB=2, ADA=3, LTC=4

---

## 🛠️ Scripts Clés

| Script | Usage |
|--------|-------|
| `tests/test_entry_oracle_exit.py` | Test entry/exit avec Oracle (Phase 2.14) |
| `tests/test_oracle_direction_only.py` | Test Oracle pur par indicateur |
| `tests/test_indicator_independence.py` | Preuve corrélation indicateurs |
| `tests/test_holding_strategy.py` | Test holding minimum |
| `src/train.py` | Entraînement modèles |
| `src/prepare_data_direction_only.py` | Préparation datasets |

---

## ⚠️ Règles Critiques

### 1. Ne JAMAIS exécuter de scripts
Claude n'a PAS les données. Fournir commandes, utilisateur exécute.

### 2. Réutiliser l'existant
Chercher logique dans scripts existants avant réécrire.

### 3. MACD = Indicateur Pivot
- Meilleur pour trading réel (moins trades, plus stable)
- RSI/CCI = modulateurs seulement

### 4. Hiérarchie des indicateurs

| Contexte | Classement |
|----------|------------|
| **Trading réel (PnL Net)** | MACD 🥇 > CCI 🥈 > RSI 🥉 |
| Oracle PnL Brut | RSI 🥇 > CCI 🥈 > MACD 🥉 |
| ML Accuracy | MACD 🥇 > CCI 🥈 > RSI 🥉 |

---

## 📊 Historique des Phases

| Phase | Résultat | Conclusion |
|-------|----------|------------|
| 2.6 Holding Min | +110% brut, 30k trades | Signal fonctionne! |
| 2.7 Veto Rules | -3.9% trades | Insuffisant |
| 2.8 Direction-Only | +0.1% à +0.9% | Stable |
| 2.9 ATR Filters | Échec | Flickering bypass |
| 2.10 Transition Sync | 58% sync | Gap accuracy expliqué |
| 2.11 Weighted Loss | -6.5% | Dégradation |
| 2.12 Prob Fusion | -15% à -43% | Échec total |
| 2.13 Indépendance | Corr=1.0 | Même signal prouvé |
| 2.14 Entry/Exit Oracle | MACD -2,082% | MACD meilleur |
| **2.15 Formule Labels** | **t vs t-1, Win Rate 53-57%** | **✅ SUCCÈS TOTAL** |

---

## 🚀 Pour Continuer

### Commandes Utiles

```bash
# Test Oracle par indicateur
python tests/test_oracle_direction_only.py --indicator macd --split test --fees 0.001

# Test Entry/Exit avec Oracle
python tests/test_entry_oracle_exit.py --asset BTC --split test

# Entraînement modèle
python src/train.py --data data/prepared/dataset_*_macd_direction_only_kalman.npz --epochs 50
```

### Ce Que Tu Dois Faire

1. **Lire** `/home/user/trad/CLAUDE.md` pour contexte complet
2. **Proposer** une approche pour réduire trades à ~3,000
3. **Créer** le script ou modifier l'existant
4. **Fournir** la commande à exécuter

---

## 📌 Résumé Exécutif

| Aspect | État |
|--------|------|
| **Modèles ML** | ✅ Fonctionnent (92.4% MACD) |
| **Signal** | ✅ Existe (+607% brut avec Oracle) |
| **Problème** | ❌ Trop de trades (13k × frais) |
| **Solution** | 🎯 Réduire à ~3,000 trades |
| **Indicateur pivot** | MACD (moins trades, plus stable) |
| **Prochaine action** | Timeframe 15/30min ou holding agressif |

---

## 💡 Suggestions Immédiates

1. **🕐 Timeframe 15min** → Créer script préparation données 15min
2. **⏱️ Holding 100p** → Modifier `test_holding_strategy.py` pour tester
3. **📊 Seuils 0.95/0.05** → Modifier grid search dans `test_entry_oracle_exit.py`

**Dis-moi quelle approche tu veux tester et je prépare le code!**
