# Phase 2.7 - Confidence Veto Rules - État des Lieux

**Date**: 2026-01-07
**Statut**: ✅ **RÈGLES VALIDÉES - Prochaine étape: Combiner avec holding_min=30p**

---

## 📊 Résultats Actuels (holding_min=5p, 20k samples)

### Comparaison Stratégies

| Stratégie | Trades | Réduction | Win Rate | Δ WR | PnL Brut | PnL Net | Amélioration | Blocages (R1/R2/R3) |
|-----------|--------|-----------|----------|------|----------|---------|--------------|---------------------|
| **Baseline** | 1,251 | - | 34.13% | - | +6.34% | -118.76% | - | - |
| **R1+R2+R3** | **991** | **-20.8%** | 33.91% | -0.23% | -0.07% | **-99.17%** | **+19.59%** | 737/0/2 |
| R1 seule | 993 | -20.6% | 33.94% | -0.20% | -0.30% | -99.60% | +19.16% | 737/0/0 |
| R1+R2 | 993 | -20.6% | 33.94% | -0.20% | -0.30% | -99.60% | +19.16% | 737/0/0 |

### Observations Clés

1. **✅ Règles fonctionnent correctement**
   - Réduction de 20.8% des trades (258 trades évités)
   - Amélioration de +19.59% du PnL Net
   - Win Rate stable (34.13% → 33.91%, -0.23%)

2. **ℹ️ Règle #1 domine largement**
   - 737 blocages sur 739 total (99.7%)
   - Zone Grise MACD (<0.20 conf) capture presque toutes les situations
   - Règles #2 et #3 quasiment inutiles (0 et 2 blocages)

3. **⚠️ PnL encore négatif**
   - Baseline: -118.76% → Veto: -99.17%
   - Amélioration significative mais insuffisante
   - Cause: holding_min=5p trop court → trop de trades restants

---

## 🔍 Analyse des 3 Règles

### Règle #1: Zone Grise MACD (conf < 0.20)

**Performance**: ⭐⭐⭐⭐⭐ (737/739 blocages)

**Conclusion**: **ESSENTIELLE** - Capture 99.7% des situations à bloquer

**Action**: ✅ Conserver tel quel

### Règle #2: Veto Ultra-Fort (témoin conf >0.70 vs MACD <0.20)

**Performance**: ⭐ (0 blocages)

**Pourquoi?** Règle #1 bloque déjà tous les cas où MACD <0.20

**Conclusion**: **REDONDANTE** avec Règle #1

**Action**: ⚪ Garder pour sécurité mais impact négligeable

### Règle #3: Confirmation Requise (MACD 0.20-0.40, témoin <0.50)

**Performance**: ⭐ (2 blocages)

**Pourquoi?** Conditions trop spécifiques (0.20-0.40 + témoins <0.50 simultanément)

**Conclusion**: **QUASI-INUTILE** dans la pratique

**Action**: ⚪ Garder pour sécurité mais impact négligeable

---

## 🎯 Prochaines Étapes

### Test #1 (PRIORITÉ 1): Veto + Holding 30p

**Hypothèse**: Combiner filtrage par confiance + durée minimale 30p

**Attendu**:
- Baseline holding 30p: 30,876 trades, +110.89% PnL Brut, -9,152% PnL Net
- Avec veto rules: ~25,000 trades (-20%)
- Frais: 25k × 0.3% × 2 = -15,000% → **PnL Net: +110.89% - 7,500% = +103.39%** ✅

**Commande**:
```bash
python tests/test_confidence_veto.py --split test --max-samples 20000 --enable-all --holding-min 30
```

**Critères de succès**:
- ✅ Trades < 26,000 (réduction confirmée)
- ✅ Win Rate ≥ 29% (maintien Phase 2.6)
- ✅ PnL Brut > +100% (signal préservé)
- ✅ **PnL Net > 0%** (OBJECTIF CRITIQUE!)

### Test #2 (PRIORITÉ 2): Full Dataset

**Objectif**: Valider stabilité des règles sur l'ensemble du test set (~640k samples)

**Commande**:
```bash
python tests/test_confidence_veto.py --split test --enable-all --holding-min 30
# Sans --max-samples pour charger tout
```

**Critères de succès**:
- Résultats cohérents avec 20k samples
- Win Rate ± 2% (tolérance variance)
- PnL Net positif maintenu

### Test #3 (OPTIONNEL): Seuils Plus Agressifs

**Objectif**: Tester si un filtrage plus strict améliore encore

**Modifications à tester**:
```python
# tests/test_confidence_veto.py

# Actuel
if macd_conf_force < 0.20:  # Zone grise

# Test agressif
if macd_conf_force < 0.30:  # Plus large
```

**Attendu**:
- Trades réduits de ~30-35% (au lieu de 20%)
- Win Rate légèrement meilleur
- Risque: filtrer trop de bons signaux

---

## 📈 Projection Phase 2.7 Complète

### Scénario Conservateur (Veto 0.20 + Holding 30p)

| Métrique | Phase 2.6 (30p) | Phase 2.7 Attendu | Delta |
|----------|-----------------|-------------------|-------|
| **Trades** | 30,876 | ~25,000 | **-20%** |
| **Win Rate** | 29.59% | ~30-32% | Stable/+2% |
| **PnL Brut** | +110.89% | ~+110% | Maintenu |
| **Frais (0.3%)** | -9,262% | **-7,500%** | **-19%** |
| **PnL Net** | -9,152% | **~+102%** ✅ | **+9,254%** |

### Scénario Optimiste (Veto 0.30 + Holding 30p)

| Métrique | Conservateur | Optimiste | Delta |
|----------|--------------|-----------|-------|
| **Trades** | 25,000 | ~22,000 | -12% |
| **Win Rate** | 30-32% | ~32-35% | +2-3% |
| **PnL Brut** | +110% | ~+120% | +10% |
| **Frais** | -7,500% | -6,600% | -12% |
| **PnL Net** | +102% | **~+113%** | +11% |

---

## 🚨 Points de Vigilance

### 1. Réduction trades peut filtrer bons signaux

**Symptôme**: Win Rate baisse ou PnL Brut se dégrade

**Solution**: Ajuster seuils (0.20 → 0.15) ou désactiver Règle #3

### 2. Sur-optimisation sur 20k samples

**Symptôme**: Full dataset donne résultats très différents

**Solution**: Validation croisée sur plusieurs périodes (walk-forward)

### 3. Corrélation avec volatilité

**Symptôme**: Performances très différentes selon périodes de marché

**Solution**: Analyse conditionnelle (séparer bull/bear/range)

---

## 🔧 Bugs Corrigés (2026-01-07)

### Bug #1: PnL Calculation Incorrect

**Problème**: Script traitait returns comme des prix
```python
# AVANT (FAUX)
current_price = 1.0 + returns[i]
pnl = (exit_price / entry_price - 1.0)
```

**Résultat**: Win Rate 3.33%, PnL -18,307% (catastrophique)

**Fix**: Accumuler returns comme test_holding_strategy.py
```python
# APRÈS (CORRECT)
current_pnl = 0.0
if position == LONG:
    current_pnl += returns[i]
pnl = current_pnl - fees
```

**Commit**: 8ec2610 - "fix: Correct PnL calculation using cumulative returns"

### Bug #2: Règles Appliquées En Position

**Problème**: Veto checks à chaque période, même en position
- 48,767 blocages mais seulement -4 trades (-0.01%)

**Fix**: Appliquer règles UNIQUEMENT à l'entrée
```python
# Règles appliquées seulement si on essaie d'entrer
if position == Position.FLAT and target != Position.FLAT:
    if macd_conf_force < 0.20:
        veto = True
```

**Commit**: 8da468c - "fix: Apply veto rules only at entry and check confidence on Force"

### Bug #3: Vérification conf_DIR au lieu de conf_FORCE

**Problème**: Signal d'entrée utilise `macd_force == 1` mais règles vérifiaient `macd_conf_dir`

**Justification**: 99.67% des erreurs ont Force=WEAK (analyse chirurgicale)

**Fix**: Vérifier conf_force
```python
# AVANT
if macd_conf_dir < 0.20:

# APRÈS
if macd_conf_dir < 0.20 or macd_conf_force < 0.20:
```

**Commit**: 8da468c (même commit que Bug #2)

---

## 📚 Références

**Scripts**:
- `tests/test_confidence_veto.py` - Script principal de test
- `tests/analyze_confidence_patterns.py` - Analyse chirurgicale 20k samples
- `tests/test_holding_strategy.py` - Référence pour PnL calculation

**Documentation**:
- [CONFIDENCE_VETO_RULES.md](CONFIDENCE_VETO_RULES.md) - Règles complètes
- [COMPARATIVE_CONFIDENCE_ANALYSIS.md](COMPARATIVE_CONFIDENCE_ANALYSIS.md) - Comparaison MACD/RSI/CCI
- [CLAUDE.md](../CLAUDE.md) - Vue d'ensemble projet

**Commits Critiques**:
- `8ec2610` - Fix PnL calculation (règle d'or: copier test_holding_strategy.py)
- `8da468c` - Fix règles entry-only + conf_force
- `f796584` - Fix extraction returns from X features
- `31d0be9` - Fix logger typo

---

**Créé**: 2026-01-07
**Auteur**: Claude Code
**Statut**: ✅ Règles validées - Prêt pour Test #1 (holding_min=30p)
