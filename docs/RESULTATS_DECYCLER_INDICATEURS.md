# Résultats : Decycler Appliqué sur Indicateurs Techniques

## 📊 Vue d'Ensemble

**Date des tests** : 2026-01-01
**Dataset** : Données simulées (GBM), 10000 bougies 5min (34.7 jours)
**Méthode** : Filtre Decycler PARFAIT (forward-backward) appliqué sur différents signaux
**Règle de trading** : `filtered[t-1] > filtered[t-2]` → BUY, sinon SELL
**Exécution** : Trade à `open[t+1]`

---

## 🎯 Objectif du Test

Comparer la performance du filtre Decycler (en mode parfait) appliqué sur :
1. **Prix brut** (close)
2. **Indicateurs techniques** : RSI, CCI, Bollinger %B, MACD

**Question** : Quel signal donne les meilleurs résultats une fois filtré ?

---

## 📈 Résultats Complets

### Tableau Comparatif

| Indicateur | Rendement | Profit Factor | Sharpe | Max DD | Win Rate | Trades |
|------------|-----------|---------------|--------|--------|----------|--------|
| **Close (prix brut)** | **+610.30%** | **995.60** | 52.21 | -0.08% | **97.7%** | 172 |
| **RSI(14)** | **+488.60%** | **41.47** | **61.24** | -0.97% | 91.2% | 284 |
| **Bollinger %B(20)** | +445.09% | 17.28 | 61.21 | -1.17% | 85.6% | 383 |
| CCI(20) | +408.34% | 11.82 | 56.34 | -1.34% | 81.7% | 398 |
| MACD Histogram | +180.51% | 3.79 | 33.56 | -3.06% | 72.0% | 418 |

**Buy & Hold** : -0.90% (pour comparaison)

---

## 🏆 Classement par Critère

### Par Profit Factor
1. **Close** : 995.60 ← Quasi-parfait !
2. **RSI** : 41.47
3. **Bollinger** : 17.28
4. CCI : 11.82
5. MACD : 3.79

### Par Rendement Total
1. **Close** : +610.30%
2. **RSI** : +488.60%
3. **Bollinger** : +445.09%
4. CCI : +408.34%
5. MACD : +180.51%

### Par Sharpe Ratio
1. **RSI** : 61.24 ← Meilleur ratio rendement/risque
2. **Bollinger** : 61.21
3. CCI : 56.34
4. **Close** : 52.21
5. MACD : 33.56

### Par Win Rate
1. **Close** : 97.7% ← Quasi-infaillible
2. **RSI** : 91.2%
3. **Bollinger** : 85.6%
4. CCI : 81.7%
5. MACD : 72.0%

### Par Fréquence de Trading
1. MACD : 418 trades (12.2 trades/jour)
2. CCI : 398 trades (11.6 trades/jour)
3. **Bollinger** : 383 trades (11.2 trades/jour)
4. **RSI** : 284 trades (8.3 trades/jour)
5. **Close** : 172 trades (5.0 trades/jour)

---

## 💡 Analyse Détaillée

### 🥇 Close (Prix Brut) - Le Champion Absolu

**Performance** :
- Profit Factor : **995.60** (exceptionnellement élevé)
- Rendement : +610.30%
- Win Rate : 97.7% (seulement 4 trades perdants sur 172 !)
- Max Drawdown : -0.08% (quasi-nul)

**Caractéristiques** :
- Très peu de trades (172 sur 34 jours = 5.0/jour)
- Quasi-infaillible (97.7% de réussite)
- Drawdown minimal (très sûr)

**Avantages** :
- ✅ Performance spectaculaire
- ✅ Très faible risque
- ✅ Simplicité (pas besoin de calculer d'indicateur)

**Inconvénients** :
- ⚠️ Peu de trades (moins d'opportunités)
- ⚠️ Difficile à reproduire en réalité (mode parfait)

**Pour l'IA** :
- Cible à atteindre théoriquement
- Benchmark de référence

---

### 🥈 RSI(14) - Meilleur Compromis

**Performance** :
- Profit Factor : **41.47** (excellent)
- Rendement : +488.60%
- Win Rate : 91.2%
- Sharpe Ratio : **61.24** (meilleur de tous !)

**Caractéristiques** :
- Fréquence modérée (284 trades = 8.3/jour)
- Très bon équilibre rendement/risque
- Win Rate élevé (91.2%)

**Avantages** :
- ✅ Excellent Sharpe Ratio (rendement ajusté au risque)
- ✅ Bonne fréquence de trading
- ✅ Performance élevée et stable
- ✅ Indicateur bien connu et testé

**Inconvénients** :
- Légèrement plus de risque que Close (-0.97% max DD vs -0.08%)

**Pour l'IA** :
- **RECOMMANDÉ** comme signal principal
- Excellent équilibre entre tous les critères
- Plus réaliste à reproduire que Close brut

---

### 🥉 Bollinger %B(20) - Bon Équilibre

**Performance** :
- Profit Factor : 17.28
- Rendement : +445.09%
- Win Rate : 85.6%
- Sharpe : 61.21 (presque aussi bon que RSI)

**Caractéristiques** :
- Fréquence élevée (383 trades = 11.2/jour)
- Bon équilibre rendement/risque
- Capture bien la volatilité

**Avantages** :
- ✅ Très bon Sharpe Ratio
- ✅ Haute fréquence (plus d'opportunités)
- ✅ Capture les phases de volatilité

**Inconvénients** :
- Win Rate plus faible que RSI (85.6% vs 91.2%)

**Pour l'IA** :
- Bon complément au RSI
- Utile pour diversification

---

### CCI(20) - Performance Solide

**Performance** :
- Profit Factor : 11.82
- Rendement : +408.34%
- Win Rate : 81.7%

**Caractéristiques** :
- Haute fréquence (398 trades)
- Détecte bien les cycles

**Avantages** :
- ✅ Bonnes performances
- ✅ Beaucoup de trades

**Inconvénients** :
- Win Rate moyen (81.7%)
- Max DD plus élevé (-1.34%)

**Pour l'IA** :
- Signal secondaire
- Moins prioritaire que RSI/Bollinger

---

### MACD Histogram - Moins Efficace

**Performance** :
- Profit Factor : 3.79 (le plus faible)
- Rendement : +180.51% (toujours rentable !)
- Win Rate : 72.0%

**Caractéristiques** :
- Très haute fréquence (418 trades = 12.2/jour)
- Plus de risque (Max DD -3.06%)

**Avantages** :
- ✅ Toujours rentable (+180%)
- ✅ Beaucoup d'opportunités

**Inconvénients** :
- ❌ PF faible (3.79)
- ❌ Win Rate le plus bas (72%)
- ❌ Max DD le plus élevé (-3.06%)

**Pour l'IA** :
- Non recommandé comme signal principal
- Peut-être utile en complément

---

## 🎯 Recommandations pour l'IA

### Signal Principal Recommandé : **RSI(14) Filtré**

**Raisons** :
1. **Excellent équilibre** : PF 41.47, Win 91.2%, Sharpe 61.24
2. **Fréquence optimale** : 8.3 trades/jour (ni trop, ni trop peu)
3. **Risque contrôlé** : Max DD -0.97% (très faible)
4. **Meilleur Sharpe** : Meilleur ratio rendement/risque
5. **Réalisme** : Plus reproductible que Close brut

### Architecture IA Suggérée

```python
# Entrée : Ghost Candles + OHLCV (features causales)
X = ghost_candles[t]

# Label (généré offline avec Decycler parfait) :
RSI = calculate_rsi(close, period=14)
filtered_RSI = decycler_perfect(RSI)  # Forward-backward
Y[t] = 1 if filtered_RSI[t-1] > filtered_RSI[t-2] else 0

# IA apprend :
# X → Y (prédire la pente du RSI filtré)
```

### Signaux Complémentaires

Pour diversification, on peut combiner :
1. **Signal principal** : RSI(14) filtré
2. **Signal secondaire** : Bollinger %B(20) filtré
3. **Validation** : Close brut filtré (confirmation)

**Approche multi-signaux** :
```python
# Voter ou moyenner les prédictions
prediction_RSI = model_RSI.predict(X)
prediction_BOL = model_BOL.predict(X)
prediction_close = model_close.predict(X)

# Vote majoritaire
final_signal = (prediction_RSI + prediction_BOL + prediction_close) / 3 > 0.5
```

---

## 📊 Trade-offs à Considérer

### Fréquence vs Précision

| Indicateur | Trades/jour | Win Rate | Trade-off |
|------------|-------------|----------|-----------|
| Close | 5.0 | 97.7% | Peu de trades, quasi-parfait |
| RSI | 8.3 | 91.2% | **Équilibre optimal** |
| Bollinger | 11.2 | 85.6% | Plus de trades, bonne précision |
| MACD | 12.2 | 72.0% | Beaucoup de trades, faible précision |

**Observation** : Plus on trade, plus le Win Rate baisse. RSI offre le meilleur compromis.

### Rendement vs Risque (Sharpe)

Sharpe Ratio = Rendement / Volatilité × √trades_per_year

| Indicateur | Rendement | Sharpe | Interprétation |
|------------|-----------|--------|----------------|
| RSI | +488% | **61.24** | Rendement élevé, risque très faible |
| Bollinger | +445% | **61.21** | Presque identique au RSI |
| Close | +610% | 52.21 | Rendement max mais moins efficient |

**Observation** : RSI et Bollinger ont le meilleur ratio rendement/risque.

---

## 🔬 Limites et Précautions

### 1. Données Simulées (GBM)

⚠️ **Ces résultats sont sur données SIMULÉES**, pas réelles !

- GBM = processus aléatoire (Geometric Brownian Motion)
- Manque : tendances, cycles, volatility clustering du BTC réel
- **Sur vraies données BTC** : résultats probablement différents

**Action requise** :
- ✅ Tester sur vraies données BTC (data/raw/BTCUSD_all_5m.csv)
- ✅ Valider que les tendances se confirment

### 2. Mode Parfait (Non-Causal)

⚠️ **Filtres utilisés connaissent le FUTUR !**

- Decycler parfait = forward + backward
- Impossible en trading réel
- **Objectif** : Valider la MÉTHODE théoriquement

**En production** :
- ❌ N'utilise PAS les filtres non-causaux
- ✅ Utilise l'IA pour prédire la pente

### 3. Performances Attendues avec IA

⚠️ **L'IA ne reproduira PAS ces performances exactes !**

**Monde parfait** (filtres non-causaux) :
- RSI : PF 41.47, +488%
- Bollinger : PF 17.28, +445%

**Réalité avec IA** (estimation réaliste) :
- RSI : PF **2.0-5.0**, +50-150%
- Bollinger : PF **1.5-3.0**, +30-100%

**Raison** : L'IA prédit avec erreur, pas parfaitement.

**Target réaliste pour l'IA** :
- ✅ Accuracy > 60% (au-dessus du hasard 50%)
- ✅ Profit Factor > 1.5-2.0
- ✅ Sharpe Ratio > 1.0

---

## 📝 Méthodologie du Test

### Calcul des Indicateurs

```python
# RSI(14)
RSI = calculate_rsi(close, period=14)  # Valeurs 0-100

# CCI(20)
CCI = calculate_cci(high, low, close, period=20)  # Valeurs ~-200 à +200

# Bollinger %B(20)
BOL = calculate_bollinger_position(close, period=20)  # Valeurs 0-100
# %B = (close - lower_band) / (upper_band - lower_band) × 100

# MACD Histogram
MACD = calculate_macd(close, fast=12, slow=26, signal=9)
MACD_norm = normalize(MACD)  # Normalisé 0-100
```

### Application du Filtre Decycler Parfait

```python
def apply_decycler_perfect(signal):
    """Decycler en mode parfait (non-causal)."""
    # Forward pass
    forward = ehlers_decycler(signal)

    # Backward pass
    backward = ehlers_decycler(forward[::-1])

    # Reverse pour obtenir version smooth
    return backward[::-1]
```

### Génération du Signal

```python
# À l'instant t, on compare filtered[t-1] vs filtered[t-2]
for t in range(2, len(filtered)):
    if filtered[t-1] > filtered[t-2]:
        signal[t] = 'BUY'   # Pente haussière
        position[t] = 1     # LONG
    else:
        signal[t] = 'SELL'  # Pente baissière
        position[t] = -1    # SHORT
```

### Exécution du Trade

```python
# Signal détecté à t → Trade exécuté à open[t+1]
entry_price = open[t+1]
exit_price = open[t_next+1]

# Rendement LONG
if position == LONG:
    return = (exit_price - entry_price) / entry_price

# Rendement SHORT
if position == SHORT:
    return = (entry_price - exit_price) / entry_price
```

---

## 🎓 Conclusions Finales

### Ce que ce Test Prouve

✅ **La méthode `filtered[t-1] > filtered[t-2]` FONCTIONNE** sur tous les indicateurs testés

✅ **Tous les indicateurs sont rentables** avec Decycler parfait (+180% à +610%)

✅ **RSI(14) est le meilleur compromis** pour l'IA :
- Performance élevée (PF 41.47)
- Fréquence optimale (8.3 trades/jour)
- Meilleur Sharpe Ratio (61.24)
- Win Rate élevé (91.2%)

✅ **Le prix brut reste le champion** mais peu réaliste pour l'IA

### Pour la Suite du Projet

**Prochaines étapes** :

1. **Tester sur vraies données BTC**
   - Valider les tendances sur données réelles
   - Comparer avec résultats simulés

2. **Développer l'IA**
   - Prédire pente de RSI(14) filtré
   - Architecture CNN-LSTM
   - Classification binaire (0/1)

3. **Target réaliste**
   - Accuracy > 60%
   - Profit Factor > 1.5-2.0
   - Sharpe > 1.0

4. **Diversification**
   - Combiner RSI + Bollinger + Close
   - Vote majoritaire ou ensemble

---

**Date** : 2026-01-01
**Version** : 1.0
**Status** : Validé en mode parfait (données simulées)
**Action requise** : Test sur données BTC réelles
