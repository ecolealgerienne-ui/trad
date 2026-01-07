# Tests Multi-Indicateurs avec Filtres Croisés

**Date**: 2026-01-07
**Objectif**: Réduire le nombre de trades (30k → 15-20k) pour atteindre rentabilité nette

---

## 🎯 Contexte

### Découverte Phase 2.6 - Signal Fonctionne!

**Résultats Holding 30 périodes (MACD)**:
- Trades: 30,876 (-34% vs baseline 46,920)
- Win Rate: **29.59%** (+15.59% vs baseline 14%)
- PnL Brut: **+110.89%** ✅ **POSITIF!**
- PnL Net: -9,152% ❌ (frais: -9,262%)

**Diagnostic**:
- ✅ Le signal FONCTIONNE (PnL Brut prouve ça)
- ✅ Le modèle ML est bon (92% accuracy valide)
- ❌ Problème = **TROP DE TRADES** × frais 0.3% détruit le PnL net

**Solution**: Réduire encore les trades en utilisant RSI+CCI comme filtres témoins

---

## 📋 Stratégie Multi-Indicateurs

### Principe

**Architecture hiérarchique**:
```
MACD (Décideur Principal)
  ↓ Direction + Force
  ↓
RSI + CCI (Témoins/Filtres)
  ↓ Veto si désaccord fort
  ↓
DÉCISION FINALE
```

### Règles de Trading

**Entrée**:
- MACD Direction=UP ET Force=STRONG → Target LONG
- MACD Direction=DOWN ET Force=STRONG → Target SHORT
- Force=WEAK → Target FLAT (attente)

**Sortie avec Holding**:
```python
# PRIORITÉ 1: Retournement Direction MACD
if direction_flip and target != position:
    exit_and_reverse()  # Immédiat, même si < 5p

# PRIORITÉ 2: Force=WEAK
elif Force == WEAK:
    if duration < 5p:
        continue_trade()  # IGNORER signal, continuer
    else:  # >= 5p
        exit_trade()      # Sortie autorisée
```

**Holding fixe**: 5 périodes (~25 min)

---

## 🔬 8 Combinaisons de Filtres

Chaque combinaison teste un mix de filtres (Kalman/Octave) pour les 3 indicateurs:

| Code | MACD Filter | RSI Filter | CCI Filter | Description |
|------|-------------|------------|------------|-------------|
| **KKK** | Kalman | Kalman | Kalman | Triple Kalman (conservateur) |
| **KKO** | Kalman | Kalman | Octave | MACD/RSI Kalman, CCI Octave |
| **KOK** | Kalman | Octave | Kalman | MACD/CCI Kalman, RSI Octave |
| **KOO** | Kalman | Octave | Octave | MACD Kalman, RSI/CCI Octave |
| **OKK** | Octave | Kalman | Kalman | MACD Octave, RSI/CCI Kalman |
| **OKO** | Octave | Kalman | Octave | RSI Kalman, MACD/CCI Octave |
| **OOK** | Octave | Octave | Kalman | CCI Kalman, MACD/RSI Octave |
| **OOO** | Octave | Octave | Octave | Triple Octave (agressif) |

**Hypothèses**:
- **Kalman**: Plus conservateur, moins de faux signaux, peut filtrer plus
- **Octave**: Plus agressif, capture mieux les retournements
- Mix optimal = MACD (décideur) avec filtres complémentaires

---

## 🚀 Commande d'Exécution

```bash
python tests/test_multi_indicator_filters.py --split test
```

**Durée estimée**: ~2-3 minutes (charge 6 datasets, teste 8 combinaisons)

---

## 📊 Métriques Analysées

Pour chaque combinaison:

| Métrique | Description | Objectif |
|----------|-------------|----------|
| **Trades** | Nombre total de trades | **15,000-20,000** (réduction -50%) |
| **Win Rate** | % de trades gagnants | **30-40%** (maintien/amélioration) |
| **PnL Brut** | Rendement sans frais | Positif (signal fonctionne) |
| **PnL Net** | Rendement après frais 0.3% | **POSITIF** ✅ |
| **Sharpe Ratio** | Rendement ajusté risque | **>1.0** (robuste) |
| **Profit Factor** | Gains/Pertes | >1.1 (souhaité) |
| **Avg Duration** | Durée moyenne trade | ~10-15 périodes |

---

## 📈 Résultats Attendus

### Scénario Idéal (Succès)

```
✅ Meilleure Combinaison: OKO
   Trades: 18,234 (-41% vs holding 30p)
   Win Rate: 32.4%
   PnL Brut: +105.23%
   PnL Net: +9.42% ✅ POSITIF!
   Sharpe Ratio: 1.8
   Profit Factor: 1.12
```

**Interprétation**:
- Réduction trades suffisante pour absorber frais
- Win Rate maintenu autour 30%
- **STRATÉGIE VALIDÉE** → Passage en production

### Scénario Moyen (Amélioration Partielle)

```
⚠️  Meilleure Combinaison: KKO
   Trades: 24,500 (-21% vs holding 30p)
   Win Rate: 31.2%
   PnL Brut: +98.12%
   PnL Net: -2,345% ❌ Encore négatif
   Sharpe Ratio: 0.8
```

**Interprétation**:
- Amélioration mais insuffisante
- Besoin de tests supplémentaires:
  - Holding 7-10p (au lieu de 5p)
  - Filtrage additionnel (volatilité, volume)
  - Ajustement seuils Force

### Scénario Négatif (Échec)

```
❌ Toutes Combinaisons: PnL Net < 0
   Meilleure: -5,234%
   Trades: 26,000-35,000
   Win Rate: 28-33%
```

**Interprétation**:
- Filtrage par indicateurs INSUFFISANT
- Problème structurel plus profond
- **Action**: Pivot vers Meta-Labeling (changement de target)

---

## 🔍 Analyse Post-Test

### Questions à Répondre

1. **Quelle combinaison réduit le plus les trades?**
   - Objectif: <20,000 trades
   - Si insuffisant: augmenter holding ou filtrage additionnel

2. **Le filtrage maintient-il le Win Rate?**
   - Attendu: 28-35% (vs 29.59% holding 30p)
   - Si chute <25%: filtres trop agressifs

3. **Kalman vs Octave: Lequel filtre mieux?**
   - Comparer KKK vs OOO (extrêmes)
   - Identifier pattern optimal (ex: MACD Octave + RSI/CCI Kalman)

4. **PnL Net positif atteint?**
   - Si OUI: ✅ Succès, documenter et valider
   - Si NON: Analyser écart (combien de trades en trop?)

---

## 📝 Prochaines Étapes Selon Résultats

### Si PnL Net POSITIF trouvé (Succès)

1. ✅ **Valider sur autres splits** (train, val)
2. ✅ **Walk-forward analysis** (stabilité temporelle)
3. ✅ **Tests robustesse**:
   - Variation frais (0.1%, 0.2%, 0.3%)
   - Sensibilité holding (3p, 5p, 7p)
4. ✅ **Documentation stratégie** complète
5. ✅ **Préparation production**

### Si Amélioration PARTIELLE (Encore Négatif)

1. ⚠️ **Identifier meilleure combinaison** (Sharpe max)
2. ⚠️ **Tests supplémentaires**:
   - Holding 7p ou 10p (au lieu de 5p)
   - Seuils Force adaptatifs
   - Filtrage volatilité (ATR, vol_rolling)
3. ⚠️ **Analyse des erreurs restantes**:
   - Distribution temporelle
   - Contexte de marché (trending vs ranging)

### Si TOUS Négatifs (Échec)

1. ❌ **Pivot Meta-Labeling** (changement de target):
   ```python
   # Au lieu de prédire Direction/Force
   # Prédire: Probabilité de succès du trade
   Y_meta = probability_profitable_trade
   ```
2. ❌ **Analyser limites fondamentales**:
   - Le signal 5min est-il trop bruité?
   - Frais 0.3% insurmontables à cette échelle?
3. ❌ **Considérer alternatives**:
   - Timeframe 15min/30min (moins de trades)
   - Maker fees 0.02% (si exchange supporte)
   - Agrégation multi-assets (correlation)

---

## 🎯 Critères de Validation

### Critère #1: Rentabilité
- ✅ **VALIDÉ** si PnL Net > 0 sur test set
- ⚠️ **LIMITE** si -5% < PnL Net < 0
- ❌ **ÉCHEC** si PnL Net < -5%

### Critère #2: Réduction Trades
- ✅ **EXCELLENT** si <20,000 trades (-35%+)
- ⚠️ **ACCEPTABLE** si 20,000-25,000 trades (-20%)
- ❌ **INSUFFISANT** si >25,000 trades (<-20%)

### Critère #3: Win Rate
- ✅ **EXCELLENT** si Win Rate ≥32%
- ⚠️ **ACCEPTABLE** si 28% ≤ Win Rate < 32%
- ❌ **INSUFFISANT** si Win Rate <28%

### Critère #4: Sharpe Ratio
- ✅ **EXCELLENT** si Sharpe >1.5
- ⚠️ **ACCEPTABLE** si 1.0 ≤ Sharpe ≤ 1.5
- ❌ **INSUFFISANT** si Sharpe <1.0

---

## 📌 Rappel Important

**NE PAS OUBLIER**:
- L'objectif n'est PAS de créer un nouveau modèle
- L'objectif est de **RÉDUIRE LA FRÉQUENCE DE TRADING** sans perdre le signal
- Le modèle fonctionne (92% accuracy, PnL Brut +110.89%)
- Le problème est purement **économique** (trop de trades × frais)

**Outils de réduction**:
1. ✅ Holding minimum (déjà testé: 5p fixe)
2. 🔧 Filtrage multi-indicateurs (EN COURS: 8 combinaisons)
3. ⏭️ Filtrage volatilité/volume (si 1+2 insuffisants)
4. ⏭️ Meta-Labeling (si tout échoue)

---

**FIN DU DOCUMENT**
