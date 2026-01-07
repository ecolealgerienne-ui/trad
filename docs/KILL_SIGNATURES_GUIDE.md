# Guide Kill Signatures - Analyse des Faux Positifs MACD

**Date**: 2026-01-07
**Objectif**: Identifier les configurations qui tuent les signaux MACD (Faux Positifs)

---

## 🎯 Principe

### Définitions

- **Faux Positif**: MACD dit Direction=UP mais PnL_brut < 0
- **PnL_brut**: Rendement cumulé jusqu'au prochain flip MACD (sans frais, horizon variable)
- **Lift**: P(Variable=X | Erreur) / P(Variable=X | Tout)
- **Seuil pertinence**: Lift > 1.2 (variable sur-représentée dans erreurs)

### Méthodologie 2-Phases

**Phase 1 - Découverte (20k samples BTC)**:
1. Extraire Faux Positifs (MACD=UP, PnL<0)
2. Calculer Lift univarié (4 variables)
3. Valider Pattern A et C

**Phase 2 - Validation (620k samples restants)**:
4. Tester patterns découverts out-of-sample
5. Vérifier Lift ≥ 80% Lift discovery

---

## 📊 Variables Analysées

### 4 Variables Clés (MACD-centric)

| Variable | Description | Pattern |
|----------|-------------|---------|
| **MACD_Kalman_Force=WEAK** | MACD monte mais vitesse faible | A |
| **MACD_Octave_Dir=DOWN** | Octave contredit Kalman | C |
| **MACD_Octave_Force=WEAK** | Octave détecte faiblesse | A |
| **Kalman≠Octave_Dir** | Désaccord filtres | C |

### Patterns Hypothèses

**Pattern A - Divergence d'Inertie**:
- MACD=UP mais Octave_Force=WEAK
- Hypothèse: MACD monte par inertie, momentum réel faible
- **Lift attendu**: 1.5-2.5×

**Pattern C - Dissonance Structurelle**:
- Kalman_Dir ≠ Octave_Dir
- Hypothèse: Range (bruit), filtres en désaccord
- **Lift attendu**: 1.2-1.5× (coverage 3.49%)

---

## 🚀 Commandes

### Phase 1: Découverte (20k samples)

```bash
python tests/analyze_kill_signatures.py --indicator macd --n-discovery 20000
```

**Output attendu**:
```
🔍 EXTRACTION FAUX POSITIFS...
  Signaux UP trouvés: 9,847
  Faux Positifs (PnL<0): 787 (8.0%)
  Durée moyenne trades: 8.3 périodes
  PnL moyen FP: -0.412%

🧮 CALCUL LIFT UNIVARIÉ...

LIFT UNIVARIÉ - TOP VARIABLES
Variable                              Lift  Precision   Recall  Coverage  Verdict
--------------------------------------------------------------------------------
MACD_Octave_Force=WEAK               2.3×      68.4%    41.2%     15.3%  ✅ VALIDÉ
MACD_Octave_Dir=DOWN                 1.8×      62.1%    28.7%     12.1%  ⚠️ MODÉRÉ
MACD_Kalman_Force=WEAK               1.5×      58.3%    35.4%     19.8%  ⚠️ MODÉRÉ
Kalman≠Octave_Dir                    1.2×      54.2%     3.5%      3.5%  ⚠️ MODÉRÉ

📊 DÉTAILS TOP 3:
1. MACD_Octave_Force=WEAK
   Lift: 2.3× (freq erreurs: 35.2% vs global: 15.3%)
   Precision: 68.4% (si veto, vraie erreur 68.4% du temps)
   Recall: 41.2% (détecte 41.2% des erreurs MACD)
   Coverage: 15.3% (bloque 15.3% des trades)

🎯 VALIDATION PATTERNS...

Pattern A: Divergence Inertie
  Description: MACD=UP & Octave_Force=WEAK
  Lift: 2.3×
  Precision: 68.4%
  Recall: 41.2% (324/787 erreurs)
  Coverage: 15.3%
  Verdict: ✅ VALIDÉ

Pattern C: Dissonance Structurelle
  Description: Kalman_Dir ≠ Octave_Dir
  Lift: 1.2×
  Precision: 54.2%
  Recall: 3.5% (28/787 erreurs)
  Coverage: 3.5%
  Verdict: ❌ FAIBLE
```

**Fichier généré**: `results/kill_signatures_macd_discovery.json`

---

### Phase 2: Validation (Reste)

```bash
python tests/analyze_kill_signatures.py --indicator macd --validate
```

**Critère validation**:
```
Lift_validation ≥ 0.8 × Lift_discovery

Exemple:
  Pattern A - Discovery: Lift 2.3×
  Pattern A - Validation: Lift 2.1× → VALIDÉ (2.1 ≥ 0.8×2.3 = 1.84)
```

**Fichier généré**: `results/kill_signatures_macd_validation.json`

---

## 📈 Interprétation Résultats

### Métriques Clés

| Métrique | Signification | Seuil Validation |
|----------|---------------|------------------|
| **Lift** | Sur-représentation dans erreurs | > 1.2 (pertinent) |
| **Precision** | % vraies erreurs si veto | > 60% (fiable) |
| **Recall** | % erreurs détectées | > 30% (utile) |
| **Coverage** | % trades bloqués | 10-30% (optimal) |

### Verdicts

| Verdict | Critères | Action |
|---------|----------|--------|
| ✅ **VALIDÉ** | Lift ≥ 2.0 ET Recall ≥ 40% | Implémenter veto |
| ⚠️ **MODÉRÉ** | Lift ≥ 1.5 ET Recall ≥ 20% | Tester en combinaison |
| ❌ **FAIBLE** | Lift < 1.5 OU Recall < 20% | Ignorer |

### Exemple Décision

**Si Pattern A validé (Lift 2.3×, Recall 41%)**:

```python
# Règle Veto dans stratégie
if MACD_Kalman_Dir == UP:
    if Octave_Force == WEAK:
        # Veto: 68% chance erreur, bloque 41% erreurs MACD
        action = HOLD
    else:
        action = LONG
```

**Impact attendu**:
- Trades réduits: -15%
- Erreurs évitées: -41%
- Win Rate: 14% → ~23% (estimation)
- PnL Net: -14,000% → potentiellement **POSITIF**

---

## 🔍 Extension Pattern B (RSI)

Pattern B nécessite charger dataset RSI en plus de MACD.

**Script extension** (à créer si Pattern A/C validés):

```bash
# Analyser multi-indicateurs
python tests/analyze_kill_signatures_multi.py \
    --target macd \
    --witnesses rsi cci
```

**Pattern B - Conflit Temporel**:
- MACD=UP mais RSI_Dir=DOWN
- Hypothèse: RSI (rapide) anticipe retournement
- **Lift attendu**: 2.0-3.0× (meilleur candidat théorique)

---

## 📝 Prochaines Étapes Selon Résultats

### Si Pattern A VALIDÉ (Lift > 2.0)

1. ✅ Implémenter veto Octave_Force=WEAK
2. ✅ Tester impact PnL (attendu: +2,000% à +4,000%)
3. ✅ Tester sur RSI/CCI (généralisation)
4. ✅ Extension Pattern B (multi-indicateurs)

### Si TOUS Patterns FAIBLES (Lift < 1.5)

1. ⚠️ Revoir définition PnL_Futur (horizon trop court?)
2. ⚠️ Tester features additionnelles (Volume, ATR)
3. ⚠️ Considérer Holding Minimum (stratégie alternative)
4. ⚠️ Meta-Labeling (changement target)

---

## 🎯 Objectif Final

**Transformer**:
```
MACD Baseline:
  Accuracy: 92.42%
  Win Rate: 14.00%
  PnL Net: -14,129%
```

**En**:
```
MACD + Kill Signatures:
  Accuracy: 92.42% (inchangé)
  Win Rate: 50%+ (filtrage erreurs)
  PnL Net: POSITIF
```

**Levier**: Éliminer 30-50% des erreurs MACD avec 10-20% trades bloqués.

---

**FIN DU GUIDE**
