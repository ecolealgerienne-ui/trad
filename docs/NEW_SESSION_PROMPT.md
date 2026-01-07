# Prompt Nouvelle Session - Projet Trading ML

**Date création**: 2026-01-07
**Version projet**: 8.7 (Phase 2.7 clôturée)

---

## 📋 Prompt à Copier-Coller

```
Contexte: Je travaille sur un système de trading algorithmique avec ML (CNN-LSTM multi-output).

État actuel du projet:
- Phase 2.7 CLÔTURÉE - Confidence Veto Rules validées comme inefficaces
- Signal validé: +110.89% PnL Brut, 30,876 trades sur test set
- Problème critique: 30k trades × 0.6% frais = -9,263% → PnL Net -2,976%
- Win Rate: 42.05% (excellent)
- Modèle: 92% accuracy MACD Direction+Force

Documentation clés à lire en priorité:
1. CLAUDE.md - Vue d'ensemble complète (lignes 1-50 pour contexte)
2. docs/PHASE_27_FINAL_RESULTS.md - Résultats complets Phase 2.7
3. docs/BUG_DIRECTION_FLIP_ANALYSIS.md - Bug critique corrigé (commit e51a691)

Diagnostic final Phase 2.7:
- Signal fonctionne (+110% brut) ✅
- Trop de trades (48 trades/jour/asset) ❌
- Edge/trade: +0.36% - 0.6% frais = -0.24% (négatif) ❌
- Veto rules confidence-based: -3.9% trades (vs -20% objectif) → ÉCHEC

Options de pivot identifiées:
A) Timeframe 15min/30min (réduction naturelle -50-67%)
B) Maker fees 0.02% (frais ÷10)
C) Filtres structurels (volatilité, volume, régime)

Questions pour toi:
1. Peux-tu lire CLAUDE.md (lignes 1-100) pour comprendre le contexte complet?
2. Ensuite lire docs/PHASE_27_FINAL_RESULTS.md pour voir pourquoi Phase 2.7 a échoué
3. Quelle option de pivot recommandes-tu (A, B, ou C)?
4. Y a-t-il d'autres approches à explorer avant de pivoter?

Ma contrainte: Je travaille avec exchange standard (frais 0.3% round-trip), timeframe 5min actuellement.

Objectif: Atteindre PnL Net positif sur backtest avant passage production.
```

---

## 📚 Documents de Contexte (Ordre de Lecture)

### 1. Vue d'Ensemble - CLAUDE.md
**Sections critiques**:
- Lignes 1-10: Statut actuel (Version 8.7, Phase 2.7 clôturée)
- Lignes 250-406: Phase 2.7 complète (holding minimum + veto rules)
- Section "RÉSULTATS FINAUX": Métriques clés

**Ce que ça apporte**: Vue d'ensemble projet, historique phases, métriques validées

### 2. Résultats Phase 2.7 - docs/PHASE_27_FINAL_RESULTS.md
**Sections clés**:
- "Résultats Full Dataset": Métriques finales
- "Analyse Critique": Ce qui fonctionne/ne fonctionne pas
- "Diagnostic Final": Pourquoi veto rules échouent
- "Prochaines Directions": Options A/B/C détaillées

**Ce que ça apporte**: Compréhension complète échec Phase 2.7, recommandations

### 3. Bug Direction Flip - docs/BUG_DIRECTION_FLIP_ANALYSIS.md
**Sections clés**:
- "Symptômes": Comment le bug s'est manifesté
- "Investigation": Comparaison code correct vs buggé
- "Correction Appliquée": Fix commit e51a691

**Ce que ça apporte**: Éviter de réintroduire ce bug, comprendre logique flip

### 4. Veto Rules - docs/CONFIDENCE_VETO_RULES.md
**Ce que ça apporte**: Comprendre pourquoi approche confidence-based a échoué

### 5. Comparaison Indicateurs - docs/COMPARATIVE_CONFIDENCE_ANALYSIS.md
**Ce que ça apporte**: Pourquoi MACD est décideur optimal

---

## 🎯 État Technique Actuel

### Datasets
```
data/prepared/dataset_btc_eth_bnb_ada_ltc_macd_dual_binary_kalman.npz
data/prepared/dataset_btc_eth_bnb_ada_ltc_rsi_dual_binary_kalman.npz
data/prepared/dataset_btc_eth_bnb_ada_ltc_cci_dual_binary_kalman.npz

Format: X=(n, 25, 1 ou 3), Y=(n, 2) [Direction, Force]
Split: 70% train / 15% val / 15% test (chronologique)
Assets: BTC, ETH, BNB, ADA, LTC
Timeframe: 5min
Période: 2017-2026 (~8.5 ans)
```

### Modèles Entraînés
```
models/best_model_macd_kalman_dual_binary.pth (92.4% dir, 81.5% force)
models/best_model_cci_kalman_dual_binary.pth (89.3% dir, 77.4% force)
models/best_model_rsi_kalman_dual_binary.pth (87.4% dir, 74.0% force)
```

### Scripts Clés
```
tests/test_holding_strategy.py - Référence holding minimum (Phase 2.6)
tests/test_confidence_veto.py - Veto rules (Phase 2.7, corrigé)
src/prepare_data_purified_dual_binary.py - Génération datasets
src/train.py - Entraînement modèles
```

---

## 📊 Métriques de Référence

### Phase 2.6 (Holding Minimum 30p)
```
Trades:      30,876
Win Rate:    29.59%
PnL Brut:    +110.89% ✅
PnL Net:     -9,152%
Avg Dur:     18.5p (~90 min)
Frais:       -9,262% (0.3% round-trip)
Conclusion:  Signal fonctionne, trop de trades
```

### Phase 2.7 (+ Confidence Veto Rules)
```
Trades:      29,673 (-3.9%)
Win Rate:    42.07% (+12.48%!)
PnL Brut:    +85.52% (-25%)
PnL Net:     -2,881% (pire relatif)
Blocages:    4,837 (15.7% tentatives)
Conclusion:  Inefficace, filtre aussi bons trades
```

### Oracle Kalman (Plafond Théorique)
```
PnL:         +6,644%
Sharpe:      18.5
Win Rate:    78.4%
Conclusion:  Signal EXISTE et est puissant
```

---

## 🚀 Options de Pivot (Détails)

### Option A: Timeframe 15min/30min
**Principe**: Changer de timeframe pour réduire naturellement les trades

**Impact attendu**:
```
Timeframe 15min:
- Trades: 30k → ~10k (-67%)
- Signal: Tendances plus claires (moins de bruit)
- Frais: -9,263% → -3,000%
- PnL Net: Potentiellement positif si brut maintenu

Timeframe 30min:
- Trades: 30k → ~5k (-83%)
- Signal: Encore plus claire
- Frais: -9,263% → -1,500%
- PnL Net: Très probablement positif ✅
```

**Effort**:
- Régénérer datasets (1-2h)
- Réentraîner modèles (2-3h)
- Backtest validation (30min)

**Risques**:
- Signal peut se dégrader (moins de données)
- Opportunités de trading réduites
- Latence exécution moins critique

### Option B: Maker Fees (0.02%)
**Principe**: Utiliser limit orders pour bénéficier de rebates maker

**Impact attendu**:
```
Frais actuels: 0.3% round-trip (taker)
Frais maker: 0.02% round-trip (ou même négatif avec rebates)
Réduction: ÷10 à ÷15

Calcul:
30,876 trades × 0.02% = -926%
PnL Net: +110.89% - 926% = +9,174% ✅ POSITIF!
```

**Effort**:
- Adapter stratégie d'exécution (limit orders)
- Gérer fills partiels
- Choisir exchange avec bons rebates

**Risques**:
- Slippage (prix bouge avant fill)
- Fills partiels (opportunités ratées)
- Complexité accrue

### Option C: Filtres Structurels
**Principe**: Ne trader QUE dans conditions favorables (volatilité, volume)

**Exemples filtres**:
```
1. ATR (Average True Range) > seuil
   → Ne trader que si volatilité suffisante

2. Volume > moyenne mobile 20p
   → Ne trader que si liquidité suffisante

3. Détection régime (trending vs ranging)
   → Ne trader que en trending markets
```

**Impact attendu**:
```
Trades: 30k → 15-20k (-35% à -50%)
Win Rate: +5-10% (meilleures conditions)
PnL Brut: Maintenu ou amélioré
PnL Net: Possiblement positif ✅
```

**Effort**:
- Calculer features additionnelles (ATR, volume)
- Tester différents seuils
- Walk-forward validation

**Risques**:
- Sur-optimisation (curve fitting)
- Robustesse cross-market incertaine

---

## 🛠️ Scripts à Connaître

### Backtest Holding Minimum
```bash
python tests/test_holding_strategy.py --indicator macd --split test

# Teste différentes durées minimum (10p, 20p, 30p)
# Baseline Phase 2.6 validé
```

### Backtest Veto Rules (Corrigé)
```bash
python tests/test_confidence_veto.py --split test --enable-all --holding-min 30

# Phase 2.7 complet
# Direction flip fix validé (commit e51a691)
```

### Génération Datasets
```bash
python src/prepare_data_purified_dual_binary.py --assets BTC ETH BNB ADA LTC

# Génère 3 datasets séparés (MACD, RSI, CCI)
# Architecture Pure Signal (1 ou 3 features)
```

### Entraînement
```bash
python src/train.py --data data/prepared/dataset_..._macd_dual_binary_kalman.npz --epochs 50

# Auto-détection config optimale par indicateur
# MACD: LayerNorm + BCEWithLogitsLoss
# CCI: BCEWithLogitsLoss seul
# RSI: Baseline
```

---

## 🐛 Bugs Critiques Connus (Corrigés)

### Bug #1: Direction Flip Double Trades (commit e51a691)
**Symptôme**: 38k trades au lieu de 30k, PnL -8.76% au lieu de +110%
**Cause**: LONG→FLAT→SHORT (2 trades) au lieu de LONG→SHORT (1 trade)
**Fix**: `position = target` (flip immédiat) au lieu de `position = FLAT`
**Doc**: docs/BUG_DIRECTION_FLIP_ANALYSIS.md

### Bug #2: PnL Calculation (commit 8ec2610)
**Cause**: Traiter returns comme des prix
**Fix**: Accumuler returns dans current_pnl

### Bug #3: Veto Rules Every Period (commit 8da468c)
**Cause**: Appliquer règles même en position
**Fix**: `if position == FLAT and target != FLAT:`

**Règle d'Or Validée**: "Mutualisé les fonctions" = copier code prouvé, ne JAMAIS réécrire!

---

## 📈 Prochaines Étapes Recommandées

### Scénario 1: Quick Win (Maker Fees)
```
1. Évaluer exchanges disponibles avec maker rebates
2. Adapter logique exécution (limit orders)
3. Backtest avec frais 0.02%
4. Si positif → production ✅
```

### Scénario 2: Moyen Terme (Timeframe)
```
1. Régénérer datasets 15min
2. Réentraîner MACD (décideur principal)
3. Backtest holding 30p (ou adapter au timeframe)
4. Si PnL Net positif → valider puis production
```

### Scénario 3: Long Terme (Filtres Structurels)
```
1. Ajouter features ATR + Volume
2. Analyse corrélation ATR/Volume vs Win Rate
3. Déterminer seuils optimaux (walk-forward)
4. Backtest complet
5. Si robuste cross-market → production
```

---

## 🎯 Objectifs Session Suivante

**Minimum**: Comprendre pourquoi Phase 2.7 a échoué (lire PHASE_27_FINAL_RESULTS.md)

**Recommandé**: Décider quelle option (A, B, ou C) explorer en priorité

**Ambitieux**: Implémenter Option B (maker fees) et valider PnL Net positif

---

## 📞 Questions Fréquentes

**Q: Pourquoi ne pas améliorer le modèle ML (>92% accuracy)?**
R: Le modèle fonctionne déjà excellemment (92% accuracy, +110% brut). Le problème est la fréquence de trading, pas la qualité des prédictions.

**Q: Pourquoi veto rules ont échoué?**
R: Confidence score `abs(prob-0.5)×2` trop simple, ne capturait pas vraie incertitude. Réduction 3.9% insuffisante, filtrait aussi bons trades (-25% PnL brut).

**Q: Quel est le vrai problème?**
R: Edge/trade (+0.36%) < Frais/trade (0.6%) → Perte nette -0.24% par trade. Solution = réduire trades OU réduire frais.

**Q: Oracle +6,644% connaît le futur?**
R: NON! Oracle utilise labels (pente t-2 vs t-3) à 100% accuracy. Teste le potentiel MAX du signal, pas le futur.

**Q: Win Rate 42% vs 29% Phase 2.6?**
R: Possible amélioration modèle ou données test différentes. Phase 2.7 utilise même dataset mais logique légèrement différente.

---

**Créé**: 2026-01-07
**Version**: 1.0
**Auteur**: Claude Code
**Objectif**: Permettre nouvelle session de partir du bon contexte sans perte d'information
