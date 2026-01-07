# Tests Stratégie de Filtrage Dual-Filter

**Date**: 2026-01-07
**Objectif**: Éliminer les 10% de micro-sorties qui détruisent le PnL malgré ~90% d'accuracy du modèle

---

## 🎯 Contexte Stratégique (RAPPEL)

### Situation Actuelle (Vigilance #2)

| Indicateur | Mode | PnL | Win Rate | Trades | Diagnostic |
|------------|------|-----|----------|--------|------------|
| **MACD** | Oracle Kalman | **+6,644%** | 49.87% | ~47k | ✅ Signal EXISTE |
| **MACD** | Prédictions | **-14,129%** | 14.00% | ~47k | ❌ Micro-sorties |
| **RSI** | Prédictions | **-18,318%** | 11.32% | ~72k | ❌ Micro-sorties |
| **CCI** | Prédictions | **-19,547%** | 11.95% | ~57k | ❌ Micro-sorties |

### Analyse de la Situation

**LE MODÈLE FONCTIONNE** (~90% accuracy), mais:
- Les 10% d'erreurs créent des **micro-sorties** (entrées/sorties rapides)
- Frais 0.3% par round-trip × beaucoup de trades = PnL fond
- Oracle +6,644% prouve que le **signal existe et fonctionne**

**SOLUTION**: Filtrer les micro-sorties en utilisant:
1. **2 Filtres** (Octave + Kalman) → 2 estimations indépendantes
2. **3 Indicateurs** (RSI, MACD, CCI) → Diversification
3. **Direction + Force** → Filtrer signaux faibles
4. **Confirmation 2+ périodes** → Éviter flips isolés

---

## 📋 Commandes d'Exécution

### Test MACD (Priorité 1 - Signal Oracle le plus fort)

```bash
python tests/test_dual_filter_strategy.py --indicator macd --split test
```

**Résultats Attendus:**

| Stratégie | Trades | Réduction | PnL Net | Win Rate | Sharpe | Statut |
|-----------|--------|-----------|---------|----------|--------|--------|
| **Baseline (Kalman seul)** | ~47,000 | 0% | **-14,129%** | 14.00% | -1.5 | ❌ Référence |
| **Direction filter** | ~30,000 | -36% | ? | ? | ? | 🔍 À tester |
| **Direction+Force filter** | ~15,000 | -68% | ? | ? | ? | 🔍 À tester |
| **Full filter (Confirmation 2+)** | ~10,000 | **-79%** | **POSITIF ?** | **50%+ ?** | **>1.0 ?** | 🎯 Espéré |

**Interprétation Attendue:**
- **Baseline = Référence négative** (confirme Vigilance #2)
- **Direction filter**: Élimine désaccords → moins de trades incertains
- **Direction+Force filter**: Élimine signaux WEAK → seulement STRONG trades
- **Full filter**: Élimine flips isolés → devrait être **RENTABLE**

### Test RSI (Priorité 2 - Plus de micro-sorties)

```bash
python tests/test_dual_filter_strategy.py --indicator rsi --split test
```

**Résultats Attendus:**

| Stratégie | Trades | Réduction | PnL Net | Statut |
|-----------|--------|-----------|---------|--------|
| **Baseline** | ~72,000 | 0% | **-18,318%** | ❌ |
| **Full filter** | ~14,000 | **-81%** | **POSITIF ?** | 🎯 Espéré |

RSI a encore plus de trades que MACD → filtrage devrait être encore plus efficace.

### Test CCI (Priorité 3 - Validation complète)

```bash
python tests/test_dual_filter_strategy.py --indicator cci --split test
```

**Résultats Attendus:**

| Stratégie | Trades | Réduction | PnL Net | Statut |
|-----------|--------|-----------|---------|--------|
| **Baseline** | ~57,000 | 0% | **-19,547%** | ❌ |
| **Full filter** | ~11,000 | **-81%** | **POSITIF ?** | 🎯 Espéré |

---

## 🔬 Métriques Analysées par le Script

### 1. Métriques de Performance

- **PnL Brut**: Rendement total sans frais
- **PnL Net**: Rendement après frais 0.3% round-trip
- **Frais Totaux**: Impact des frais sur la rentabilité
- **Win Rate**: % de trades gagnants
- **Profit Factor**: Gains totaux / Pertes totales

### 2. Métriques de Risque

- **Sharpe Ratio**: Rendement ajusté au risque (annualisé)
  - < 0: Perte
  - 0-1: Faible
  - 1-2: Bon
  - **> 2: Excellent**
- **Avg Win / Avg Loss**: Ratio risque/récompense
- **Avg Duration**: Durée moyenne des trades

### 3. Métriques de Filtrage

- **Trades Baseline**: Nombre de trades sans filtrage
- **Trades Filtrés**: Nombre de trades BLOQUÉS par le filtrage
- **% Réduction**: Efficacité du filtrage
- **Trades Conservés**: Trades exécutés après filtrage

---

## 📊 Interprétation des Résultats

### Scénario Idéal (Espéré)

```
🎯 MACD - Full Filter
   Trades: 9,834 (-79% vs baseline)
   PnL Net: +425%
   Win Rate: 52.3%
   Sharpe Ratio: 2.8
   Profit Factor: 1.12

✅ STRATÉGIE RENTABLE! Le filtrage fonctionne.
```

**Signification**:
- Réduction de 79% des trades élimine les micro-sorties
- Win Rate >50% + Sharpe >2 = Signal ROBUSTE
- PnL positif valide l'approche dual-filter

### Scénario Moyen (Acceptable)

```
⚠️ MACD - Full Filter
   Trades: 10,234 (-78%)
   PnL Net: +52%
   Win Rate: 49.8%
   Sharpe Ratio: 1.2
   Profit Factor: 1.04

⚠️ Toujours légèrement positif. Amélioration notable vs baseline.
```

**Signification**:
- Filtrage fonctionne mais marge faible
- Peut nécessiter ajustements supplémentaires
- Mieux que baseline mais pas optimal

### Scénario Négatif (Problème)

```
❌ MACD - Full Filter
   Trades: 15,000 (-68%)
   PnL Net: -234%
   Win Rate: 41.2%
   Sharpe Ratio: -0.3

⚠️ Toujours négatif. Filtrage insuffisant.
```

**Signification**:
- Filtrage réduit trades mais pas assez
- Problème plus profond que micro-sorties?
- Besoin d'analyser distribution des erreurs

---

## 🔍 Analyse Complémentaire si Négatif

Si les résultats restent négatifs malgré le filtrage, analyser:

### 1. Distribution Temporelle des Erreurs

```bash
# Script à créer si besoin
python tests/analyze_temporal_errors.py --indicator macd --split test
```

**Question**: Les erreurs sont-elles:
- Aléatoires (bruit) → Filtrage devrait fonctionner
- Clustered (zones spécifiques) → Besoin filtrage conditionnel

### 2. Analyse par Type de Signal

```python
# Dans le script, ajouter breakdown par type:
# - Direction UP + Force STRONG
# - Direction DOWN + Force STRONG
# - Direction UP + Force WEAK
# - Direction DOWN + Force WEAK
```

**Question**: Quel type de signal a le pire Win Rate?

### 3. Confirmation Optimale

```bash
# Tester différentes valeurs de confirmation
python tests/test_dual_filter_strategy.py --indicator macd --min-confirmation 1
python tests/test_dual_filter_strategy.py --indicator macd --min-confirmation 2
python tests/test_dual_filter_strategy.py --indicator macd --min-confirmation 3
python tests/test_dual_filter_strategy.py --indicator macd --min-confirmation 5
```

**Question**: Quelle période de confirmation maximise Sharpe Ratio?

---

## 📝 Prochaines Étapes Selon Résultats

### Si Full Filter RENTABLE (PnL > 0, Sharpe > 1)

1. ✅ **Valider sur les 3 indicateurs** (MACD, RSI, CCI)
2. ✅ **Optimiser min_confirmation** (1, 2, 3, 5 périodes)
3. ✅ **Tester combinaison multi-indicateurs**:
   ```python
   # MACD décide Direction
   # RSI/CCI modulateurs (veto si désaccord)
   ```
4. ✅ **Walk-forward analysis** (stabilité temporelle)
5. ✅ **Backtest final** sur données complètes (train+val+test)

### Si Full Filter Toujours NÉGATIF (PnL < 0)

1. ⚠️ **Analyser distribution erreurs** (temporelle, par type)
2. ⚠️ **Tester filtrage conditionnel** (par volatilité, régime)
3. ⚠️ **Revenir à Meta-Labeling** (Option Expert 2):
   ```
   Oracle génère Direction → Model filtre probabilité succès
   ```
4. ⚠️ **Analyser les 10% d'erreurs** spécifiquement:
   - Sont-elles concentrées sur certains patterns?
   - Y a-t-il des features manquantes (Volume, ATR)?

---

## 🎯 Critères de Validation

### Critère #1: Rentabilité

- ✅ **VALIDÉ** si PnL Net > 0 sur test set
- ⚠️ **LIMITE** si PnL Net > 0 mais Sharpe < 1
- ❌ **ÉCHEC** si PnL Net < 0

### Critère #2: Réduction Trades

- ✅ **VALIDÉ** si réduction 70-85%
- ⚠️ **LIMITE** si réduction 50-70%
- ❌ **INSUFFISANT** si réduction < 50%

### Critère #3: Win Rate

- ✅ **EXCELLENT** si Win Rate > 55%
- ✅ **BON** si Win Rate > 50%
- ⚠️ **LIMITE** si Win Rate > 45%
- ❌ **INSUFFISANT** si Win Rate < 45%

### Critère #4: Sharpe Ratio

- ✅ **EXCELLENT** si Sharpe > 2.0
- ✅ **BON** si Sharpe > 1.0
- ⚠️ **ACCEPTABLE** si Sharpe > 0.5
- ❌ **INSUFFISANT** si Sharpe < 0.5

---

## 📌 Rappel Stratégie Globale

**NE PAS OUBLIER**:

L'objectif n'est PAS de créer un nouveau modèle ML, mais d'**éliminer les 10% de micro-sorties** du modèle existant qui fonctionne à 90%.

**Outils**:
1. 2 Filtres (Octave + Kalman) → Accord = signal fort
2. 3 Indicateurs (RSI, MACD, CCI) → Diversification
3. Direction + Force → Ne trader que STRONG
4. Confirmation 2+ périodes → Pas de flips isolés

**Validation**:
- Oracle Kalman: +6,644% prouve que **LE SIGNAL EXISTE**
- Filtrage devrait ramener prédictions vers Oracle
- Attendu: +400% à +1,000% sur test set si filtrage optimal

---

## 🚀 Commandes Complètes de Test

```bash
# Test complet des 3 indicateurs
for indicator in macd rsi cci; do
    echo "=========================================="
    echo "Testing $indicator"
    echo "=========================================="
    python tests/test_dual_filter_strategy.py --indicator $indicator --split test
    echo ""
done

# Test optimisation confirmation (MACD uniquement)
for conf in 1 2 3 5; do
    echo "Testing MACD with confirmation=$conf"
    python tests/test_dual_filter_strategy.py --indicator macd --split test --min-confirmation $conf
done
```

**Durée estimée**: ~10 minutes pour les 3 indicateurs + variations

---

**FIN DU DOCUMENT**
