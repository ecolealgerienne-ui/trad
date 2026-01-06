# Frais Conservateurs - Simulation Pessimiste

**Date**: 2026-01-06
**Statut**: ✅ IMPLÉMENTÉ
**Philosophie**: "Qui peut le plus, peut le moins"

---

## 🎯 PRINCIPE FONDAMENTAL

**Si votre stratégie est rentable avec 0.3% de frais simulés, elle sera une MACHINE DE GUERRE avec les vrais frais Binance (0.1-0.2%).**

En simulant des conditions **PIRES** que la réalité, vous évitez les mauvaises surprises en production.

---

## 💰 DÉCOMPOSITION DES FRAIS

### Frais par Side (un côté: entrée OU sortie)

| Composant | Valeur | Explication |
|-----------|--------|-------------|
| **Binance Standard** | 0.1% (0.001) | Frais de trading Binance spot |
| **Slippage Estimé** | 0.05% (0.0005) | Décalage prix signal → exécution réelle |
| **TOTAL PAR SIDE** | **0.15% (0.0015)** | Somme conservatrice |

### Frais par Trade Complet (aller-retour: entrée + sortie)

| Action | Frais | Total |
|--------|-------|-------|
| Entrée (FLAT → LONG/SHORT) | 0.15% | - |
| Sortie (LONG/SHORT → FLAT) | 0.15% | - |
| **TOTAL ROUND-TRIP** | - | **0.3% (0.003)** |

---

## 🔧 PARAMÈTRES --fees

Le script `test_dual_binary_trading.py` utilise maintenant **0.15% par défaut**.

### Configurations Disponibles

```bash
# Configuration CONSERVATRICE (RECOMMANDÉ)
python tests/test_dual_binary_trading.py --indicator macd --split test
# Par défaut: --fees 0.15 → 0.3% total

# Configuration RÉALISTE (Binance sans slippage)
python tests/test_dual_binary_trading.py --indicator macd --split test --fees 0.1
# → 0.2% total

# Configuration OPTIMISTE (Maker fees Binance)
python tests/test_dual_binary_trading.py --indicator macd --split test --fees 0.02
# → 0.04% total (nécessite ordres limit qui ne mangent pas le livre)
```

### Recommandation par Contexte

| Phase | Fees Recommandés | Raison |
|-------|------------------|--------|
| **Développement** | 0.15% (0.3% total) | Test conservateur |
| **Validation** | 0.1% (0.2% total) | Binance réaliste |
| **Optimisation** | 0.02-0.05% | Maker fees si stratégie le permet |

---

## 📊 IMPACT DU SLIPPAGE

### Qu'est-ce que le Slippage ?

**Définition**: Décalage entre le prix au moment du signal et le prix d'exécution réelle.

**Causes**:
- Latence réseau (quelques ms)
- Spread bid/ask
- Volatilité intra-seconde
- Ordre market vs limit

**Exemple concret**:
```
Signal généré: BTC = 50,000 USDT
Ordre envoyé:  BTC = 50,005 USDT (+5 USDT = +0.01%)
Ordre exécuté: BTC = 50,025 USDT (+25 USDT = +0.05%)
→ Slippage total: 0.05%
```

### Slippage par Timeframe

| Timeframe | Slippage Estimé | Volatilité |
|-----------|-----------------|------------|
| **5min** (actuel) | **0.05%** | Modérée |
| **1min** | **0.1%** | Élevée |
| **15min** | **0.03%** | Faible |
| **1h** | **0.01%** | Très faible |

**Note**: Plus le timeframe est court, plus le slippage est important.

---

## 🧪 RÉSULTATS COMPARATIFS (Attendus)

### Avec Logique Originale (Exit on Force WEAK)

Sur **72,377 trades** (test set, MACD predictions) :

| Fees Config | Par Side | Total RT | Fees Totaux | PnL Brut | PnL Net | Verdict |
|-------------|----------|----------|-------------|----------|---------|---------|
| **Conservateur** | 0.15% | 0.3% | **21,713%** | +49.84% | **-21,663%** | ❌ Non rentable |
| **Réaliste** | 0.1% | 0.2% | **14,475%** | +49.84% | **-14,425%** | ❌ Non rentable |
| **Optimiste** | 0.02% | 0.04% | **2,895%** | +49.84% | **-2,845%** | ⚠️ Proche breakeven |

**Conclusion**: Même avec maker fees (0.02%), **72k trades est TROP** pour un edge de +50%.

---

## 🎯 OBJECTIF DE RENTABILITÉ

### Calcul du Nombre Max de Trades

Avec PnL Brut = +49.84% et fees conservateurs (0.3% total):

```
PnL Net = PnL Brut - (N_trades × fees_per_trade)
0 = +49.84% - (N_trades × 0.3%)
N_trades_max = 49.84 / 0.3 = 166 trades
```

**Pour être rentable avec fees conservateurs (0.3%), il faut < 166 trades au total !**

### Comparaison Actuel vs Objectif

| Métrique | Actuel | Objectif | Facteur |
|----------|--------|----------|---------|
| **Trades** | 72,377 | **< 166** | **÷436** |
| **Fees** | 21,713% | **< 50%** | **÷434** |
| **PnL Net** | -21,663% | **> 0%** | - |

**Il faut diviser le nombre de trades par 436× pour être rentable avec fees conservateurs.**

---

## 🛠️ SOLUTIONS POUR RÉDUIRE LES TRADES

### Option 1: Augmenter Confirmation Temporelle

```bash
# min_confirmation = 20 (au lieu de 2)
python tests/test_dual_binary_trading.py \
    --indicator macd \
    --split test \
    --min-confirmation 20
```

**Attendu**: Trades ÷10-15

### Option 2: Augmenter Seuil Force

```bash
# threshold_force = 0.8 (au lieu de 0.5)
python tests/test_dual_binary_trading.py \
    --indicator macd \
    --split test \
    --threshold-force 0.8
```

**Attendu**: Trades ÷3-5, mais PnL Brut peut baisser

### Option 3: Timeframe Plus Long (15min/30min)

- Tendances plus longues
- Moins de bruit
- Moins de trades naturellement

**Attendu**: Trades ÷5-10

### Option 4: Combiner Plusieurs Filtres

```bash
python tests/test_dual_binary_trading.py \
    --indicator macd \
    --split test \
    --min-confirmation 10 \
    --threshold-force 0.7
```

**Attendu**: Trades ÷30-50

---

## 📋 CHECKLIST AVANT PRODUCTION

- [ ] Backtest avec fees **0.15%** (conservateur) → PnL Net > 0%
- [ ] Backtest avec fees **0.1%** (réaliste) → PnL Net > +10%
- [ ] Backtest avec fees **0.02%** (optimiste) → PnL Net > +20%
- [ ] Nombre de trades < 500 (pour être robuste)
- [ ] Win Rate > 40%
- [ ] Profit Factor > 1.5
- [ ] Max Drawdown < 20%

**Si tous les critères sont remplis avec fees 0.15%, la stratégie est PRODUCTION-READY.**

---

## 🎓 LEÇONS APPRISES

### 1. Slippage est RÉEL et IMPORTANT

- Ignorer le slippage = erreur classique de débutant
- Sur crypto 5min, slippage ~0.05% est conservateur
- Sur crypto 1min, slippage peut atteindre 0.1-0.2%

### 2. Pessimisme > Optimisme

- Simuler des conditions pires que la réalité
- Si rentable dans le pire cas → robustesse garantie
- "Qui peut le plus, peut le moins"

### 3. Edge vs Fees

```
Edge par trade = PnL Brut / N_trades
Fees par trade = constant (0.2-0.3%)

Rentabilité = (Edge > Fees) ET (N_trades raisonnable)
```

**Sans réduire drastiquement les trades, impossible d'être rentable.**

### 4. Nombre de Trades > Edge

Avec notre edge actuel (+50% sur 72k trades):
- Edge/trade = 0.069% (69 basis points)
- **C'est BON** mais mangé par les fees (200-300 basis points)

**Il faut 3-4× moins de trades pour que l'edge émerge.**

---

## 🚀 PROCHAINES ÉTAPES

1. ✅ Implémenter fees conservateurs (0.15% défaut) - **FAIT**
2. ⏳ Tester avec min_confirmation élevé (10-20)
3. ⏳ Tester avec threshold_force élevé (0.7-0.8)
4. ⏳ Préparer données 15min/30min
5. ⏳ Combiner filtres pour réduire trades ÷50+

**Objectif**: Passer sous 500 trades avec PnL Net > 0% (fees 0.15%)

---

**Créé par**: Claude Code
**Date**: 2026-01-06
**Commit**: 149da6a
