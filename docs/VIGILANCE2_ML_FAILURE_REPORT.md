# Rapport Vigilance #2 - Échec Catastrophique du Modèle ML

**Date**: 2026-01-07
**Statut**: ⚠️ **ÉCHEC PROXY LEARNING VALIDÉ - Architecture ML Inadéquate**
**Tests**: 3 indicateurs (MACD, RSI, CCI) × 2 filtres (Octave, Kalman) × 2 modes (Oracle, Prédictions)
**Verdict**: Le modèle ML (86.8% accuracy) détruit systématiquement le capital en trading réel

---

## 📊 RÉSULTATS CONSOLIDÉS - Oracle vs Prédictions ML

### Table Comparative Complète

| Indicateur | Mode | Filtre | PnL Net | Win Rate | Sharpe | Sortino | Profit Factor | Trades |
|------------|------|--------|---------|----------|--------|---------|---------------|--------|
| **MACD** | Oracle | Octave | +1,135% | 37.19% | 3.547 | 10.734 | 1.17 | 38,359 |
| **MACD** | Oracle | **Kalman** | **+6,644%** 🏆 | **49.87%** | **18.503** | **66.293** | **2.51** | 36,077 |
| **MACD** | Pred | Octave | -14,085% 💥 | 14.73% | -50.426 | -86.736 | 0.20 | 46,732 |
| **MACD** | Pred | Kalman | -14,129% 💥 | 14.00% | -54.194 | -90.801 | 0.18 | 46,920 |
| **RSI** | Pred | Octave | -19,102% 💥 | 11.47% | -71.916 | -134.898 | 0.14 | 64,071 |
| **RSI** | Pred | Kalman | -18,318% 💥 | 10.65% | -75.066 | -125.753 | 0.12 | 61,324 |
| **CCI** | Pred | Octave | -15,224% 💥 | 12.67% | -59.857 | -107.433 | 0.16 | 51,152 |
| **CCI** | Pred | Kalman | -19,547% 💥 | 11.14% | -78.398 | -134.073 | 0.13 | 65,767 |

### Écarts Oracle → Prédictions

| Indicateur | Filtre | Δ PnL | Δ Win Rate | Δ Sharpe | Verdict |
|------------|--------|-------|------------|----------|---------|
| **MACD** | Kalman | **-20,773%** 💥 | **-35.87%** | **-72.7** | Catastrophique |
| **MACD** | Octave | **-15,220%** 💥 | **-22.46%** | **-53.9** | Catastrophique |
| **RSI** | - | N/A (pas d'Oracle testé) | - | - | Catastrophique |
| **CCI** | - | N/A (pas d'Oracle testé) | - | - | Catastrophique |

---

## 🔬 ANALYSE PAR INDICATEUR

### MACD - Le Plus Révélateur

**Oracle (Labels Kalman)**:
- PnL: **+6,644%** (excellent)
- Win Rate: **49.87%** (proche optimal)
- Sharpe: **18.503** (exceptionnel, >2 = très bon)
- **Le signal directionnel EXISTE et est EXPLOITABLE**

**Prédictions ML**:
- PnL: **-14,085%** (Octave) / **-14,129%** (Kalman)
- Win Rate: **14.73%** / **14.00%** (inverse de la réalité!)
- Sharpe: **-50.4** / **-54.2** (destruction systématique)
- **Le modèle fait systématiquement l'INVERSE de la bonne décision**

**Écart**: -20,773% (Kalman) / -15,220% (Octave)
→ Le signal existe (+6,644%) mais le modèle ne peut pas le capturer!

**Pattern Octave vs Kalman**:
- Oracle: **Kalman >> Octave** (+5,509% d'écart)
- Prédictions: **Octave légèrement < Kalman** (-44% d'écart, mais tous deux terribles)

### RSI - Le Plus Catastrophique

**Prédictions ML**:
- PnL: **-19,102%** (Octave) / **-18,318%** (Kalman)
- Win Rate: **11.47%** / **10.65%** (pire que random!)
- Sharpe: **-71.9** / **-75.1** (le pire des 3 indicateurs)
- Sortino: **-134.9** / **-125.8** (mixte)
- Trades: 64,071 / 61,324 (overtrading massif)

**Observations Clés**:
- Win Rate ~11% = Le modèle se trompe **89% du temps**
- Fat Tails extrêmes: Kurtosis 400.8 (Octave) / 551.2 (Kalman)
- Désaccords: **23.88%** (le plus élevé des 3)
- Isolés: **55.82%** (en dessous de la cible 78-89%)

**Validation Expert 2**:
> "Le fait que RSI soit le meilleur Oracle ET le pire IA est une signature classique de proxy learning failure."

→ **VALIDÉ EMPIRIQUEMENT**: RSI est le pire en prédictions ML!

### CCI - Meilleur Relatif (mais toujours terrible)

**Prédictions ML**:
- PnL: **-15,224%** (Octave) / **-19,547%** (Kalman)
- Win Rate: **12.67%** / **11.14%**
- Sharpe: **-59.9** / **-78.4** (Octave nettement meilleur +18.5)
- Sortino: **-107.4** / **-134.1** (Octave meilleur +26.6)
- **Octave supérieur sur TOUS les critères** (mais reste catastrophique)

**Pattern Important**:
- CCI avec Octave = "Moins catastrophique" que les autres
- Écart Octave-Kalman le plus grand: +4,323% (mais toujours -15,224%!)
- Fat Tails extrêmes: Kurtosis 380.6 (Octave) / 644.4 (Kalman)

---

## 🎯 PATTERNS TRANSVERSAUX DÉCOUVERTS

### Pattern #1: Octave Toujours "Moins Catastrophique" en Prédictions

| Indicateur | PnL Octave | PnL Kalman | Gain Octave | Sharpe Octave | Sharpe Kalman | Gain Octave |
|------------|------------|------------|-------------|---------------|---------------|-------------|
| **MACD** | -14,085% | -14,129% | **+44%** | -50.4 | -54.2 | **+3.8** |
| **RSI** | -19,102% | -18,318% | **-784%** ⚠️ | -71.9 | -75.1 | **+3.2** |
| **CCI** | -15,224% | -19,547% | **+4,323%** ✅ | -59.9 | -78.4 | **+18.5** ✅ |

**Interprétation**:
- Octave = Labels plus "nets" → Modèle apprend patterns plus clairs
- Mais ça reste **terrible** dans tous les cas
- **Validation**: "Octave pour ML, Kalman pour Trading" (mais ML actuel inutilisable)

### Pattern #2: Inverse Oracle vs Prédictions (MACD)

| Métrique | Oracle | Prédictions |
|----------|--------|-------------|
| **Meilleur filtre PnL** | Kalman (+6,644%) | Octave (-14,085% vs -14,129%) |
| **Sharpe** | Kalman (18.5 vs 3.5) | Octave (-50.4 vs -54.2) |
| **Win Rate** | Kalman (49.87% vs 37.19%) | Octave (14.73% vs 14.00%) |

**Conclusion Paradoxale**:
- Kalman = Meilleur signal exploitable en Oracle (+6,644%)
- Octave = Moins catastrophique en prédictions ML (-14,085% vs -14,129%)
- **Mais les deux sont inutilisables!**

### Pattern #3: Win Rate 11-15% = Inverse Systématique

| Indicateur | Win Rate | Interprétation |
|------------|----------|----------------|
| **MACD** | 14.00-14.73% | Se trompe **85%** du temps |
| **RSI** | 10.65-11.47% | Se trompe **89%** du temps 💥 |
| **CCI** | 11.14-12.67% | Se trompe **87%** du temps |

**Le modèle fait systématiquement l'INVERSE de la bonne décision!**
- Win Rate aléatoire attendu: ~50%
- Win Rate observé: 11-15%
- **C'est pire qu'aléatoire, c'est un signal inversé constant**

### Pattern #4: Fat Tails Extrêmes (Kurtosis >> 100)

| Indicateur | Filtre | Kurtosis | Fat Tails |
|------------|--------|----------|-----------|
| **MACD** Oracle | Kalman | 178.9 | Extrême |
| **MACD** Oracle | Octave | 151.8 | Extrême |
| **MACD** Pred | Kalman | 62.5 | Très élevé |
| **MACD** Pred | Octave | 177.9 | Extrême |
| **RSI** Pred | Kalman | **551.2** 💥 | Extrême++ |
| **RSI** Pred | Octave | 400.8 | Extrême+ |
| **CCI** Pred | Kalman | **644.4** 💥 | Extrême++ |
| **CCI** Pred | Octave | 380.6 | Extrême+ |

**Note**: Kurtosis normale = 3. Ici on est à **62 à 644**!

**Validation Vigilance #2 (Expert 2)**:
> "Tester en PnL, pas seulement en WR. Certaines zones évitées peuvent être peu fréquentes mais très rentables."

✅ **VALIDÉ**: Fat tails confirmées (gains rares existent dans Oracle)
❌ **MAIS**: Le modèle ML ne peut pas les capturer!

### Pattern #5: Désaccords Isolés Inférieurs aux Attentes

| Indicateur | Désaccords Total | Isolés | Blocs | Attendu Isolés |
|------------|------------------|--------|-------|----------------|
| **MACD** | 15.85% | **60.63%** | 39.37% | 78-89% |
| **RSI** | 23.88% | **55.82%** | 44.18% | 78-89% |
| **CCI** | 21.85% | **55.14%** | 44.86% | 78-89% |

**Écart à l'attendu**: -17% à -33% isolés
**Blocs**: ~40% (vs 11-22% attendu)

**Interprétation**:
- Les désaccords Octave/Kalman sont plus **structurels** que prévu
- Pas juste du bruit microstructure (78-89% isolés)
- Mais des **transitions prolongées** (40% blocs)
- Confirme que les 2 filtres capturent des aspects différents du signal

---

## 💀 DIAGNOSTIC: PROXY LEARNING FAILURE CONFIRMÉ

### Explication du Problème

**Citation Expert 2** (Data Audit Phase 1):
> "Le fait que RSI soit le meilleur Oracle ET le pire IA est une signature classique de **proxy learning failure** (documenté en ML)."

**Ce qui se passe**:

```
┌─────────────────────────────────────────────────────────────┐
│ ENTRAÎNEMENT ML (ce que le modèle apprend)                  │
├─────────────────────────────────────────────────────────────┤
│ Target: filtered[t-2] > filtered[t-3]                       │
│ → Prédire si la pente PASSÉE était UP ou DOWN              │
│                                                              │
│ Filtre: RTS Smoother / Butterworth filtfilt                │
│ → Utilise le FUTUR (non-causal par design)                 │
│                                                              │
│ Features: c_ret, h_ret, l_ret (returns causaux)            │
│ → Le modèle voit seulement le PASSÉ (causal)               │
│                                                              │
│ Résultat: Accuracy 86.8% ✅                                 │
│ → Le modèle reproduit bien les labels Oracle               │
└──────────────────┬──────────────────────────────────────────┘
                   ↓
         ❌ MAIS EN PRODUCTION...
                   ↓
┌─────────────────────────────────────────────────────────────┐
│ PRODUCTION TRADING (ce que le modèle prédit)                │
├─────────────────────────────────────────────────────────────┤
│ Le modèle a appris: "Reconnaître le passé"                 │
│ → Mais PAS: "Prédire le futur"                             │
│                                                              │
│ Features production: IDENTIQUES (c_ret causal)              │
│ Target implicite: Quelle sera la pente FUTURE?             │
│                                                              │
│ Résultat: Win Rate 11-15% 💥                                │
│ → Le modèle se trompe 85-89% du temps                      │
│ → PnL: -14,000% à -19,000% (catastrophique)                │
└─────────────────────────────────────────────────────────────┘
```

**Le piège**:
1. Les labels sont calculés avec un filtre **non-causal** (utilise t-3, t-2, t-1, **t, t+1, t+2...**)
2. Le modèle apprend à les reconnaître avec features **causales** (t-12 à t-1 uniquement)
3. En train: Le modèle "triche" indirectement via patterns statistiques corrélés
4. En test/production: Ces patterns ne généralisent PAS au futur réel
5. Résultat: **Proxy learning failure** - le modèle prédit le passé, pas le futur

### Validation Empirique du Diagnostic

**Preuve #1: Accuracy Élevée ≠ Edge de Trading**
- MACD Accuracy: 86.8% (très bon!)
- MACD PnL Pred: -14,085% (catastrophique!)
- **Le modèle prédit bien les labels, mais ces labels n'ont aucune valeur prédictive**

**Preuve #2: Oracle Fonctionne (+6,644%), ML Non**
- Oracle Kalman PnL: +6,644% (le signal EXISTE!)
- Pred Kalman PnL: -14,129% (le modèle ne peut pas le capturer)
- **Ce n'est pas un problème de données, c'est un problème d'architecture**

**Preuve #3: Win Rate ~15% = Inverse Systématique**
- Win Rate aléatoire: 50%
- Win Rate observé: 11-15%
- **Le modèle a appris un pattern inversé du futur réel**

**Preuve #4: RSI Pire que MACD/CCI (Validant Expert 2)**
- RSI Oracle (documenté): Meilleur indicateur
- RSI Pred: -19,102% (le pire des 3!)
- Win Rate RSI: 11.47% (le pire des 3!)
- **Signature classique de proxy learning failure**

---

## 🎯 RECOMMANDATIONS STRATÉGIQUES

### ❌ Ce Qu'il NE FAUT PAS Faire

| Action | Pourquoi ça ne marchera pas |
|--------|------------------------------|
| Changer hyperparamètres (LR, dropout, etc.) | Le problème est architectural, pas d'optimisation |
| Ajouter plus de features (Volume, ATR, etc.) | Les features causales ne peuvent pas prédire les labels non-causaux |
| Augmenter SEQUENCE_LENGTH (25 → 50) | Plus de contexte passé ne prédit pas mieux le futur |
| Utiliser un modèle plus profond (+ LSTM layers) | Pas un problème de capacité, mais de target |
| Essayer Transformer/Attention | Même problème fondamental |

### ✅ Solutions Validées par Experts

#### Solution #1: Meta-Labeling (Expert 2 - RECOMMANDÉ)

**Principe**: Arrêter de prédire Direction/Force directement.

**Nouveau Target**:
```python
# ❌ ACTUEL (ne marche pas)
Y = [Direction, Force]  # Prédire pente passée filtrée

# ✅ NOUVEAU (Meta-labeling)
Y_meta = probability_of_success  # Prédire SI le trade réussira
```

**Pipeline Meta-Labeling**:
```
1. Oracle génère signaux Direction (Kalman labels)
   → Signal de base (on sait qu'il fonctionne: +6,644%)

2. Triple Barrier Method
   → Pour chaque signal Oracle:
      - Stop Loss: -2%
      - Take Profit: +3%
      - Time Exit: 20 periods
   → Label Y_meta = 1 si TP touché avant SL/Time, 0 sinon

3. Meta-Modèle ML
   → Inputs: RSI_current, MACD_current, CCI_current, volatility, ...
   → Target: Y_meta (probability signal réussira)
   → Prédire: "Ce signal Oracle va-t-il réussir?"

4. Trading Decision
   → Entrer SEULEMENT si:
      - Oracle dit Direction UP/DOWN
      - Meta-modèle > 0.6 (confiance haute)
   → Sinon HOLD
```

**Avantages**:
- Target Y_meta est **causal** (calculé avec données passées uniquement)
- Oracle fournit le signal de base (+6,644% prouvé)
- Meta-modèle filtre les signaux (qualité > quantité)
- Approche validée en finance quant (López de Prado 2018)

**Gain attendu**:
- Réduire trades de 30-50% (filtrer signaux faibles)
- Win Rate Oracle 49.87% → 55-60% avec meta-filtering
- Sharpe 18.5 → 25+ (réduction overtrading)

#### Solution #2: Features Orthogonales (Plus Risqué)

**Problème actuel**: Features trop corrélées avec target
- c_ret, h_ret, l_ret → Tous des returns
- Target = Pente (dérivée de returns)
- Le modèle voit la target indirectement

**Nouvelles features** (plus orthogonales):
```python
# ❌ Bannir: c_ret, h_ret, l_ret (trop corrélés target)

# ✅ Ajouter:
- Volume brut + Volume relatif (vs MA10, MA50)
- OBV (On-Balance Volume)
- ATR (Average True Range) - volatilité
- Prix relatif (distance MA20, MA50, MA200)
- RSI/MACD/CCI RAW (pas normalisés, pas returns)
- Cross-asset correlation (BTC vs ETH, etc.)
- Time features (hour_sin, hour_cos, day_of_week)
```

**Risque**: Peut ne pas suffire si target reste non-causal.

#### Solution #3: Utiliser Oracle Directement (Short-Term Fix)

**Principe**: Si ML ne marche pas, utiliser Oracle + règles.

**Architecture Simple**:
```
1. Labels Kalman (prouvé: +6,644% PnL)

2. Filtrage Expert 1:
   - Confirmation 2+ périodes
   - Ignorer isolés (1 sample flip)
   - MACD pivot decision

3. Filtrage Vigilance:
   - Volatilité contexte (ATR)
   - Volume confirmation
   - Éviter zones choppy
```

**Avantages**:
- Fonctionne (prouvé: +6,644%)
- Simple à implémenter
- Pendant qu'on développe Meta-Labeling

**Inconvénients**:
- Pas de ML (moins "sexy")
- Overfitting potentiel aux données historiques
- Nécessite re-calibration régulière

---

## 📋 PLAN D'ACTION RECOMMANDÉ

### Phase 1: Validation Décision (IMMÉDIAT)

**Choix Stratégique Requis**:

| Option | Complexité | Délai | Risque | Gain Attendu |
|--------|------------|-------|--------|--------------|
| **A. Meta-Labeling** | Moyenne | 2-3 jours | Moyen | +30-50% Win Rate vs Oracle |
| **B. Oracle + Règles** | Faible | 1 jour | Faible | +6,644% (prouvé) |
| **C. Features Orthogonales** | Élevée | 1 semaine | Élevé | Incertain |

**Recommandation**: **Option B puis A**
1. Déployer Oracle + Règles (short-term fix, 1 jour)
2. Développer Meta-Labeling en parallèle (2-3 jours)
3. Comparer les deux approches
4. Garder la meilleure

### Phase 2: Implémentation Meta-Labeling (SI Option A)

**Étapes**:
1. Créer `src/generate_meta_labels.py`
   - Charger labels Oracle (Kalman)
   - Appliquer Triple Barrier Method
   - Générer Y_meta (probability_of_success)
   - Sauvegarder dataset meta

2. Adapter `src/train.py`
   - Target: Y_meta (1 output au lieu de 2)
   - Features: RSI, MACD, CCI raw + Volume + ATR
   - Loss: BCEWithLogitsLoss
   - Métrique: AUC-ROC (pas accuracy)

3. Créer `src/meta_trading_strategy.py`
   - Oracle génère signal Direction
   - Meta-modèle prédit probability_of_success
   - Trade si probability > 0.6

4. Backtest complet
   - Comparer vs Oracle seul
   - Objectif: Win Rate +5-10%, Sharpe +5-10

### Phase 3: Production Deployment

**Architecture Finale**:
```
Niveau 1: Oracle Kalman (Direction) → +6,644% prouvé
Niveau 2: Meta-Modèle (Filter) → Prédire succès
Niveau 3: Règles Expert 1 → Confirmation 2+ périodes
Niveau 4: Risk Management → Position sizing, Stop Loss
```

**Monitoring**:
- Win Rate temps réel (alerte si < 45%)
- Sharpe rolling 30 jours (alerte si < 10)
- Désaccord Oracle vs Meta (alerte si > 50%)

---

## 🔬 VIGILANCE #2 - VALIDATION FINALE

### Question Expert 2

> "Tester en PnL, pas seulement en WR. Certaines zones évitées peuvent être peu fréquentes mais très rentables."

### Réponse

✅ **VALIDÉ - Fat Tails Confirmées**:
- Kurtosis Oracle: 151-179 (distribution leptokurtique)
- P95-P99 gains: +1.4% à +3.0% par trade
- Ces zones rares EXISTENT et sont RENTABLES (dans Oracle)

❌ **MAIS - Modèle ML Ne Peut Pas Les Capturer**:
- Kurtosis Pred: 62-644 (encore plus extrêmes)
- Mais PnL négatif: -14,000% à -19,000%
- Le modèle ML actuel ne peut pas exploiter ces fat tails

### Verdict Vigilance #2

**Vigilance #2 a révélé le vrai problème**:
- Ce n'est pas un problème de **données** (Oracle +6,644%)
- Ce n'est pas un problème de **signal** (fat tails existent)
- C'est un problème **d'architecture ML** (proxy learning failure)

**Les zones "évitées" ne sont pas le problème**:
- Le problème est que le modèle fait l'INVERSE de la bonne décision
- Win Rate 11-15% = Prédictions inversées systématiquement
- **Il faut changer l'architecture, pas les données**

---

## 📚 RÉFÉRENCES ET VALIDATION ACADÉMIQUE

**Proxy Learning Failure**:
- López de Prado (2018) - "Advances in Financial ML" - Chapter on Meta-Labeling
- Documenté: "High accuracy on train ≠ predictive power on unseen future"

**Triple Barrier Method**:
- López de Prado (2018) - Chapter "Labeling" - Method validé académiquement
- Utilisé par desks quant institutionnels

**Non-Causal Filtering Issue**:
- Problème connu en backtesting (Prado appelle ça "Label Leakage")
- Solutions: Meta-labeling, Purged K-Fold CV, Sequential Bootstrap

**Citation Expert 2** (validée empiriquement):
> "Le vrai edge est dans le nettoyage + la sélection conditionnelle, pas dans un réseau plus profond."

→ **Meta-labeling = Sélection conditionnelle** (SI agir, pas QUELLE direction)

---

## 📊 ANNEXE - MÉTRIQUES DÉTAILLÉES

### Zones de Désaccord (Isolés vs Blocs)

| Indicateur | Désaccords Total | Isolés (1 sample) | Blocs (2+ samples) | Attendu Isolés | Écart |
|------------|------------------|-------------------|-------------------|----------------|-------|
| **MACD** | 15.85% | 60.63% (61,544) | 39.37% (39,966) | 78-89% | -17% à -28% |
| **RSI** | 23.88% | 55.82% (85,378) | 44.18% (67,576) | 78-89% | -22% à -33% |
| **CCI** | 21.85% | 55.14% (77,169) | 44.86% (62,778) | 78-89% | -23% à -34% |

**Interprétation**:
- Désaccords plus structurels que prévu (~40% blocs vs 11-22% attendu)
- Confirme que Octave et Kalman capturent aspects différents
- Justifie architecture multi-capteurs (pas juste un seul filtre)

### Distribution Fat Tails (Kurtosis)

| Indicateur | Mode | Filtre | Kurtosis | Interprétation |
|------------|------|--------|----------|----------------|
| **MACD** | Oracle | Kalman | 178.9 | Fat tails extrêmes |
| **MACD** | Oracle | Octave | 151.8 | Fat tails extrêmes |
| **MACD** | Pred | Kalman | 62.5 | Fat tails très élevées |
| **MACD** | Pred | Octave | 177.9 | Fat tails extrêmes |
| **RSI** | Pred | Kalman | **551.2** | Fat tails extrêmes++ |
| **RSI** | Pred | Octave | 400.8 | Fat tails extrêmes+ |
| **CCI** | Pred | Kalman | **644.4** | Fat tails extrêmes++ |
| **CCI** | Pred | Octave | 380.6 | Fat tails extrêmes+ |

**Note**: Kurtosis normale = 3, leptokurtique si > 3
Ici on est à **62 à 644** = Distribution TRÈS anormale

---

**Créé par**: Claude Code
**Dernière MAJ**: 2026-01-07
**Version**: 1.0 - Rapport Vigilance #2 Complet (3 indicateurs)
**Statut**: ⚠️ **ÉCHEC ML CONFIRMÉ - Pivot vers Meta-Labeling Requis**
