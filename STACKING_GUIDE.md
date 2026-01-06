# 🎯 Guide Complet - Stacking / Ensemble Learning

**Date**: 2026-01-06
**Objectif**: Combiner les 3 modèles experts (MACD, RSI, CCI) pour améliorer la prédiction Direction

---

## 💡 CONCEPT - Résoudre le Proxy Learning Failure

### Problème Actuel

| Métrique | Valeur | Problème |
|----------|--------|----------|
| **Accuracy Direction** | 92% | ✅ Excellent |
| **Win Rate Trading** | 14% | ❌ Catastrophique |
| **Cause** | Proxy Learning Failure | IA ne prédit pas ce que prédit l'Oracle |

### Hypothèse Stacking

> "Le Kalman original est rentable (Oracle 65-70% Win Rate). Si le Stacking améliore l'Accuracy de 92% → 95-96%, on devrait **coller mieux au Kalman** et retrouver naturellement la rentabilité."

**Approche**: Ensemble Learning pur - Combiner les 3 experts pour retrouver la Vérité (Kalman)

---

## 🏗️ ARCHITECTURE

### Niveau 1: Les 3 Modèles de Base

| Modèle | Rôle | Caractéristique |
|--------|------|-----------------|
| **MACD** | Tendance lourde | Stable mais en retard dans les virages |
| **RSI** | Vitesse pure | Réactif mais nerveux |
| **CCI** | Volatilité | Détecte les extremes |

**Chacun prédit**: Direction (UP/DOWN) + Force (STRONG/WEAK)

---

### Niveau 2: Meta-Modèle

**Inputs (X_meta)**:
```
X_meta = [
    p_macd_dir,    # Proba Direction MACD (0-1)
    p_macd_force,  # Proba Force MACD (0-1)
    p_rsi_dir,     # Proba Direction RSI (0-1)
    p_rsi_force,   # Proba Force RSI (0-1)
    p_cci_dir,     # Proba Direction CCI (0-1)
    p_cci_force,   # Proba Force CCI (0-1)
]
Shape: (n, 6)
```

**Cible (Y_meta)**:
```
Y_meta = kalman_dir  # Label Direction Original (0 ou 1)
Shape: (n, 1)
```

**Objectif**: Apprendre à combiner les 6 signaux pour retrouver le Kalman original.

---

### Règles Automatiques Apprises

Le meta-modèle apprendra automatiquement des patterns comme:

```python
Si RSI_dir change ET MACD_dir stable:
    → Écouter RSI (virage anticipé)

Si MACD_dir + RSI_dir + CCI_dir tous d'accord:
    → Confiance maximale (suivre le consensus)

Si CCI_force WEAK ET MACD_dir change:
    → Ignorer MACD (faux signal en volatilité faible)

Si RSI_force STRONG ET MACD_force WEAK:
    → Retournement imminent (écouter RSI)
```

---

## 🚀 WORKFLOW COMPLET

### ✅ Étape 0: Prérequis

**Vérifier que vous avez**:
1. Les 3 datasets dual_binary_kalman.npz
2. Les 3 modèles entraînés (.pth)

**Si manquants**, exécuter:

```bash
# 1. Générer datasets
python src/prepare_data_purified_dual_binary.py --assets BTC ETH BNB ADA LTC

# 2. Entraîner les 3 modèles
python src/train.py --data data/prepared/dataset_btc_eth_bnb_ada_ltc_macd_dual_binary_kalman.npz --epochs 50
python src/train.py --data data/prepared/dataset_btc_eth_bnb_ada_ltc_rsi_dual_binary_kalman.npz --epochs 50
python src/train.py --data data/prepared/dataset_btc_eth_bnb_ada_ltc_cci_dual_binary_kalman.npz --epochs 50
```

---

### ✅ Étape 1: Générer les Méta-Features

**Script**: `src/generate_meta_features.py`

```bash
python src/generate_meta_features.py --assets BTC ETH BNB ADA LTC
```

**Ce que fait ce script**:
1. Charge les 3 modèles entraînés (.pth)
2. Charge les 3 datasets correspondants
3. Génère les prédictions (probabilités) pour Train/Val/Test
4. Sauvegarde les méta-features

**Outputs**:
```
data/meta/
  ├── meta_features_train.npz  # X_meta: (n, 6), Y_meta: (n, 1)
  ├── meta_features_val.npz
  └── meta_features_test.npz
```

**Durée**: ~2-3 min (génération des prédictions)

---

### ✅ Étape 2: Entraîner le Meta-Modèle

**Script**: `src/train_stacking.py`

**3 modèles disponibles** (tester du plus simple au plus complexe):

#### 2.1 Logistic Regression (Baseline)

```bash
python src/train_stacking.py --model logistic
```

**Avantages**:
- ✅ Rapide (~10 secondes)
- ✅ Interprétable (poids des features)
- ✅ Baseline de référence

**Attendu**: Si linéaire suffit, devrait atteindre 94-95%

---

#### 2.2 Random Forest

```bash
python src/train_stacking.py --model rf
```

**Avantages**:
- ✅ Capture interactions non-linéaires
- ✅ Robuste
- ✅ Feature importance

**Attendu**: Si non-linéaire, devrait atteindre 95-96%

---

#### 2.3 MLP (Neural Network)

```bash
python src/train_stacking.py --model mlp --device cuda
```

**Avantages**:
- ✅ Capture patterns très non-linéaires
- ✅ Flexible

**Attendu**: Si très complexe, devrait atteindre 96%+

---

### ✅ Étape 3: Évaluer les Résultats

**Comparer les 3 modèles**:

| Modèle | Train Acc | Val Acc | Test Acc | Interprétable | Temps |
|--------|-----------|---------|----------|---------------|-------|
| Logistic | ? | ? | ? | ✅ Oui | ~10s |
| Random Forest | ? | ? | ? | ⚠️ Moyen | ~30s |
| MLP | ? | ? | ? | ❌ Non | ~2 min |

**Choisir le modèle** avec le meilleur Test Acc sans overfit (gap Train/Test < 5%).

---

## 📊 GAIN ATTENDU

### Scénario Optimiste

| Métrique | Actuel | Attendu | Gain |
|----------|--------|---------|------|
| **Accuracy Direction** | 92% | **95-96%** | +3-4% |
| **Corrélation avec Kalman** | ~0.75 | **~0.90** | +20% |
| **Win Rate Trading** | 14% | **55-65%** | **+41-51%** 🎯 |

**Justification**:
- Si on colle mieux au Kalman (95-96% Accuracy)
- Et que le Kalman est rentable (65-70% Win Rate Oracle)
- Alors l'IA devrait retrouver 80-90% de cette rentabilité

---

## 🎯 COMPARAISON AVEC PROFITABILITY RELABELING

| Approche | Objectif | Cible | Résultat Attendu |
|----------|----------|-------|------------------|
| **Profitability** | Nettoyer Force | Labels relabelés | Oracle +6-8% Win Rate |
| **Stacking** | Combiner experts | **Kalman Direction** | **IA +41-51% Win Rate** |

**Stacking = Solution au Proxy Learning Failure** 🎯

---

## 🔍 ANALYSE DES RÉSULTATS

### Si Logistic Regression Suffit (94-95%)

**Interprétation**: La combinaison optimale est **linéaire**.

**Analyse des poids**:
```python
Poids positifs élevés → Feature importante pour UP
Poids négatifs élevés → Feature importante pour DOWN

Exemple:
  MACD_dir:   +0.45  (fort signal UP si MACD prédit UP)
  RSI_dir:    +0.30  (signal UP modéré)
  RSI_force:  -0.20  (si Force faible, ignore RSI)
```

**Règles apprises**: Pondération simple des 3 experts.

---

### Si Random Forest Meilleur (95-96%)

**Interprétation**: Interactions **non-linéaires** importantes.

**Analyse Feature Importance**:
```python
Feature Importance:
  RSI_dir:     0.25  (le plus important pour virages)
  MACD_dir:    0.20  (tendance principale)
  CCI_force:   0.15  (détection extremes)
  ...
```

**Règles apprises**: Décisions en arbre (ex: SI RSI_dir > 0.6 ET MACD_force < 0.3 ALORS...)

---

### Si MLP Nécessaire (96%+)

**Interprétation**: Patterns **très complexes** nécessaires.

**Hypothèse**: Le modèle apprend des interactions d'ordre supérieur (ex: RSI×MACD×CCI).

---

## 🚨 CRITÈRES DE SUCCÈS

| Critère | Objectif | Verdict |
|---------|----------|---------|
| **Test Accuracy** | ≥ 95% | ✅ / ❌ |
| **Gap Train/Test** | < 5% | ✅ / ❌ |
| **Amélioration vs Baseline** | +3-4% | ✅ / ❌ |

**Si 3/3 ✅** → Stacking validé, tester en backtest

---

## 📋 TROUBLESHOOTING

### Problème: Test Acc < 94%

**Causes possibles**:
- Les 3 modèles de base sont trop similaires (redondants)
- Pas assez de diversité dans les prédictions

**Solutions**:
- Vérifier que les 3 modèles ont des performances différentes
- Ajouter des features (volatilité, volume)

---

### Problème: Overfit (Train 98%, Test 93%)

**Causes**: Meta-modèle trop complexe (MLP)

**Solutions**:
- Revenir à Logistic ou Random Forest
- Augmenter dropout MLP
- Réduire hidden size MLP

---

### Problème: Amélioration Faible (+1-2%)

**Causes**: Les 3 modèles font les mêmes erreurs

**Solutions**:
- Vérifier la diversité des modèles
- Entraîner les modèles de base avec des architectures différentes

---

## 🎓 LITTÉRATURE - Ensemble Learning

**Stacking** (Wolpert, 1992):
> "Combine multiple models to achieve better performance than any single model."

**Avantages**:
- Réduit biais et variance
- Exploite la diversité des modèles
- Robuste aux erreurs individuelles

**Exemples célèbres**:
- Netflix Prize (2009): Équipe gagnante utilisait Stacking
- Kaggle: 80% des solutions top utilisent Ensemble Learning

---

## 🏁 PROCHAINES ÉTAPES (Si Succès)

### Étape 4: Backtest Complet

Comparer Win Rate:
- MACD seul: 14%
- RSI seul: 12%
- CCI seul: 13%
- **Stacking**: 55-65% ? 🎯

---

### Étape 5: Combiner avec Profitability Relabeling

**Approche hybride**:
1. **Stacking** pour améliorer Direction (92% → 95%)
2. **Profitability Relabeling** pour nettoyer Force

**Gain total attendu**: Win Rate 14% → **65-70%** (Oracle-like) 🏆

---

## ✅ CHECKLIST D'EXÉCUTION

- [ ] Datasets générés (3 fichiers .npz)
- [ ] Modèles entraînés (3 fichiers .pth)
- [ ] Méta-features générées (generate_meta_features.py)
- [ ] Meta-modèle entraîné (train_stacking.py)
- [ ] Test Accuracy ≥ 95%
- [ ] Backtest Win Rate > 50%

---

**C'est la méthode la plus pure pour vérifier l'hypothèse: Est-ce que l'union fait la force pour retrouver le Kalman ?** 🚀

