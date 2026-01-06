# 🚀 Stacking - Guide de Démarrage Rapide

**Date**: 2026-01-06
**Objectif**: Combiner les 3 experts (MACD, RSI, CCI) pour améliorer Direction Accuracy
**Cible**: 92% → 95-96% accuracy → Win Rate 14% → 55-65%

---

## 🎯 Concept - Résoudre le Proxy Learning Failure

### Problème Actuel
- **Accuracy Direction**: 92% (excellent!)
- **Win Rate Trading**: 14% (catastrophique!)
- **Cause**: Proxy Learning Failure - IA ne prédit pas ce que prédit l'Oracle

### Hypothèse Stacking
> "Le Kalman original est rentable (Oracle 65-70% Win Rate). Si le Stacking améliore l'Accuracy de 92% → 95-96%, on devrait **coller mieux au Kalman** et retrouver naturellement la rentabilité."

**Approche**: Ensemble Learning pur - Combiner les 3 experts pour retrouver la Vérité (Kalman)

---

## ⚡ Lancement Ultra-Rapide

### Option 1: Script Automatisé (Recommandé) 🏆

```bash
./run_stacking_workflow.sh
```

**Ce que fait le script**:
1. ✅ Vérifie tous les prérequis
2. 🤔 Propose de générer datasets si manquants
3. 🤔 Propose d'entraîner modèles si manquants
4. 🚀 Génère méta-features automatiquement
5. 🤖 Entraîne meta-modèle (choix interactif)
6. 📊 Affiche résultats et critères de succès

**Avantages**:
- Workflow complet automatisé
- Checks et validations à chaque étape
- Instructions claires en cas d'erreur
- Interactif (demande confirmation)

---

### Option 2: Commandes Manuelles (Contrôle Total)

#### Étape 1: Générer Datasets (~5 min)
```bash
python src/prepare_data_purified_dual_binary.py --assets BTC ETH BNB ADA LTC
```

**Output attendu**:
- `dataset_btc_eth_bnb_ada_ltc_macd_dual_binary_kalman.npz`
- `dataset_btc_eth_bnb_ada_ltc_rsi_dual_binary_kalman.npz`
- `dataset_btc_eth_bnb_ada_ltc_cci_dual_binary_kalman.npz`

#### Étape 2: Entraîner 3 Modèles (~30-90 min total)
```bash
# MACD
python src/train.py \
    --data data/prepared/dataset_btc_eth_bnb_ada_ltc_macd_dual_binary_kalman.npz \
    --epochs 50

# RSI
python src/train.py \
    --data data/prepared/dataset_btc_eth_bnb_ada_ltc_rsi_dual_binary_kalman.npz \
    --epochs 50

# CCI
python src/train.py \
    --data data/prepared/dataset_btc_eth_bnb_ada_ltc_cci_dual_binary_kalman.npz \
    --epochs 50
```

**Output attendu**:
- `models/best_model_macd_kalman_dual_binary.pth`
- `models/best_model_rsi_kalman_dual_binary.pth`
- `models/best_model_cci_kalman_dual_binary.pth`

#### Étape 3: Générer Méta-Features (~2-3 min)
```bash
python src/generate_meta_features.py --assets BTC ETH BNB ADA LTC
```

**Output attendu**:
- `data/meta/meta_features_train.npz` (X: n×6, Y: n×1)
- `data/meta/meta_features_val.npz`
- `data/meta/meta_features_test.npz`

#### Étape 4: Entraîner Meta-Modèle

**Baseline (Logistic Regression) - COMMENCER PAR CELUI-CI**:
```bash
python src/train_stacking.py --model logistic
```
⏱️ Durée: ~10 secondes
🎯 Attendu: 94-95% si combinaison linéaire suffit
✅ Interprétable: Affiche les poids des features

**Non-Linéaire (Random Forest)**:
```bash
python src/train_stacking.py --model rf
```
⏱️ Durée: ~30 secondes
🎯 Attendu: 95-96% si interactions non-linéaires
✅ Feature Importance: Montre importance relative

**Deep Learning (MLP)**:
```bash
python src/train_stacking.py --model mlp --device cuda
```
⏱️ Durée: ~2 minutes
🎯 Attendu: 96%+ si patterns très complexes
⚠️ Risque d'overfit plus élevé

---

## 📊 Critères de Succès

| Critère | Objectif | Verdict |
|---------|----------|---------|
| **Test Accuracy** | ≥ 95% | ✅ / ❌ |
| **Gap Train/Test** | < 5% | ✅ / ❌ |
| **Amélioration vs Baseline** | +3-4% | ✅ / ❌ |

**Si 3/3 ✅** → Stacking validé, tester en backtest

---

## 🔍 Interprétation des Résultats

### Si Logistic Regression Suffit (94-95%)
**Interprétation**: La combinaison optimale est **linéaire**.

**Exemple de poids appris**:
```
MACD_dir:   +0.45  (fort signal UP si MACD prédit UP)
RSI_dir:    +0.30  (signal UP modéré)
RSI_force:  -0.20  (si Force faible, ignorer RSI)
```

**Règles apprises**: Pondération simple des 3 experts.

---

### Si Random Forest Meilleur (95-96%)
**Interprétation**: Interactions **non-linéaires** importantes.

**Exemple Feature Importance**:
```
RSI_dir:     0.25  (le plus important pour virages)
MACD_dir:    0.20  (tendance principale)
CCI_force:   0.15  (détection extremes)
```

**Règles apprises**: Décisions en arbre (SI RSI_dir > 0.6 ET MACD_force < 0.3 ALORS...)

---

### Si MLP Nécessaire (96%+)
**Interprétation**: Patterns **très complexes** nécessaires.

**Hypothèse**: Le modèle apprend des interactions d'ordre supérieur (ex: RSI×MACD×CCI).

---

## 🚨 Troubleshooting

### Problème: Test Acc < 94%
**Causes possibles**:
- Les 3 modèles de base sont trop similaires (redondants)
- Pas assez de diversité dans les prédictions

**Solutions**:
- Vérifier que les 3 modèles ont des performances différentes
- Ajouter des features (volatilité, volume)

---

### Problème: Overfit (Train 98%, Test 93%)
**Cause**: Meta-modèle trop complexe (MLP)

**Solutions**:
- Revenir à Logistic ou Random Forest
- Augmenter dropout MLP
- Réduire hidden size MLP

---

### Problème: Amélioration Faible (+1-2%)
**Cause**: Les 3 modèles font les mêmes erreurs

**Solutions**:
- Vérifier la diversité des modèles
- Entraîner les modèles de base avec des architectures différentes

---

## 🏁 Prochaines Étapes (Si Succès)

### Étape 5: Backtest Complet
Comparer Win Rate:
- MACD seul: 14%
- RSI seul: 12%
- CCI seul: 13%
- **Stacking**: 55-65% ? 🎯

### Étape 6: Combiner avec Profitability Relabeling
**Approche hybride**:
1. **Stacking** pour améliorer Direction (92% → 95%)
2. **Profitability Relabeling** pour nettoyer Force

**Gain total attendu**: Win Rate 14% → **65-70%** (Oracle-like) 🏆

---

## 📚 Documentation Complète

- **Guide complet**: `STACKING_GUIDE.md` (368 lignes, tout le détail)
- **Scripts créés**:
  - `src/generate_meta_features.py` - Génère les méta-features
  - `src/train_stacking.py` - Entraîne le meta-modèle
  - `run_stacking_workflow.sh` - Script automatisé complet

---

## 📋 Résumé Architecture

```
                    NIVEAU 1 - Les 3 Experts
                    ┌─────────────────────────┐
                    │  MACD (Direction+Force) │
                    │  RSI  (Direction+Force) │
                    │  CCI  (Direction+Force) │
                    └────────────┬────────────┘
                                 │
                                 ↓
                    ┌─────────────────────────┐
                    │  X_meta = [p1, p2, ...] │
                    │  Shape: (n, 6)          │
                    └────────────┬────────────┘
                                 │
                    NIVEAU 2 - Meta-Modèle
                                 ↓
                    ┌─────────────────────────┐
                    │ Logistic / RF / MLP     │
                    │ Apprend à combiner      │
                    └────────────┬────────────┘
                                 │
                                 ↓
                    ┌─────────────────────────┐
                    │  Y_pred = Direction     │
                    │  Cible: Kalman Original │
                    └─────────────────────────┘
```

**Input Meta-Modèle** (6 features):
- `p_macd_dir`, `p_macd_force`
- `p_rsi_dir`, `p_rsi_force`
- `p_cci_dir`, `p_cci_force`

**Output Meta-Modèle** (1 cible):
- `kalman_dir` (Direction Kalman Original)

**Objectif**: Apprendre à combiner les 6 signaux pour retrouver le Kalman avec 95-96% accuracy.

---

## ⚡ TL;DR - Pour Démarrer en 30 Secondes

```bash
# Workflow complet automatisé (recommandé)
./run_stacking_workflow.sh

# OU manuel si tu préfères contrôler chaque étape:
python src/prepare_data_purified_dual_binary.py --assets BTC ETH BNB ADA LTC
python src/train.py --data data/prepared/dataset_*_macd_dual_binary_kalman.npz --epochs 50
python src/train.py --data data/prepared/dataset_*_rsi_dual_binary_kalman.npz --epochs 50
python src/train.py --data data/prepared/dataset_*_cci_dual_binary_kalman.npz --epochs 50
python src/generate_meta_features.py --assets BTC ETH BNB ADA LTC
python src/train_stacking.py --model logistic
```

**C'est la méthode la plus pure pour vérifier l'hypothèse: Est-ce que l'union fait la force pour retrouver le Kalman ?** 🚀
