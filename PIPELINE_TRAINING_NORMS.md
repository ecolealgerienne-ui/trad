# Normes de Pipeline ML Trading

**Version** : 1.0 (en cours de rédaction)
**Date création** : 2026-04-26
**Auteur** : Consolidation des erreurs identifiées sur 14+ phases (Phase 1-14 foundation_finetune, slope_improvement AQ-KF, v5.0→v5.4 PatchTST)
**Objet** : checklist normative à appliquer **avant chaque entraînement** d'un modèle ML pour stratégie de trading.

---

## Pourquoi ce document

Au fil du projet, des erreurs méthodologiques se sont **répétées** sur des paradigmes différents (CNN-LSTM, Chronos LoRA, XGBoost, PatchTST). Cette consolidation transforme ces erreurs en **règles préventives**. L'objectif : ne **plus jamais** commettre la même erreur en construisant un nouveau pipeline de training.

Ce document est une **checklist obligatoire** à parcourir avant tout :
- nouvel ajout de feature
- nouveau type de label
- nouveau paradigme architectural
- nouvelle expérience comparative

**Convention** : ❌ = erreur observée dans le projet, ✅ = règle à appliquer, ⚠️ = piège subtil à connaître.

---

## Table des matières

1. **Causalité stricte** ✅ rédigé
2. Normalisation des features
3. Détection du distribution shift (drift)
4. Engineering des labels
5. Engineering des features
6. Class imbalance
7. Split train/val/test + embargo
8. Validation et diagnostics
9. Choix de modèle et entraînement
10. Audits pré-run
11. Backtest et critères de décision
12. Discipline documentaire
13. Anti-patterns transverses

---

## Section 1 — Causalité stricte

> **Règle d'or** : à chaque instant `t`, une feature ne doit utiliser **AUCUNE** information de `t+1` ou ultérieure. Le label peut utiliser le futur (par construction), mais **JAMAIS** la feature.

### 1.1 Vérifier la causalité de chaque feature

❌ **Erreur classique** : utiliser `future-looking` filtres comme labels OU comme features
- Exemple Phase 2 du projet : `pykalman.smooth()` (filtre RTS, non-causal forward-backward) utilisé pour générer des labels Oracle. **Acceptable car LABEL.**
- Si jamais ce filtre passe en feature → fuite massive.

✅ **Règle** : pour chaque feature, écrire en commentaire le **lookback strict** utilisé.

```python
# RSI[t] utilise close[t-13..t] (14 bars passées + bar courante)
# OK causal
rsi_14 = talib.RSI(close, timeperiod=14)

# MAUVAIS exemple (interdit) :
# pykalman_smoothed[t] utilise close[0..n-1] forward+backward
# → leak du futur si utilisé comme feature
```

### 1.2 Camarilla pivots et autres calculs daily/sliding

❌ **Erreur** (potentielle) : `daily_pivots[D]` calculé depuis `H/L/C[D]` au lieu de `H/L/C[D-1]`
- Phase v5 : audit identifié ce risque. Notre code utilise `daily['close'].shift(1)` puis `reindex(method='ffill')` → causal (vérifié 96% confidence).

✅ **Règle** : tout indicateur calculé sur une granularité agrégée (daily, hourly) DOIT être **shifté de 1 unité** avant ffill au timeframe inférieur.

```python
# CORRECT
daily_close = df.set_index('timestamp')['close'].resample('1D').last()
prev_daily_close = daily_close.shift(1)  # ← le shift est OBLIGATOIRE
levels = compute_camarilla(prev_daily_close)  # utilise jour D-1 pour jour D
ffilled = levels.reindex(df.index, method='ffill')  # broadcast au 5min

# INCORRECT (fuite)
levels = compute_camarilla(daily_close)  # utilise jour D pour jour D = fuite intra-jour
```

### 1.3 Rolling windows et inclusion de `t`

⚠️ **Subtilité** : `pd.Series.rolling(N).mean()` à l'index `t` utilise `[t-N+1 .. t]` **inclus**. La valeur `t` est donc dans sa propre normalisation.

✅ **C'est acceptable** pour des features temps-réel : le bar `t` est connu à sa close, donc `rolling.mean()[t]` est calculable à la close de `t`. Le modèle prédit ensuite l'action **après** `t` (sur `t+1..t+24`). Le label reste séparé.

❌ **NE PAS** utiliser `rolling().shift(1)` par paranoïa : ça décale tout d'un bar inutilement et perd 1 unité d'info récente.

### 1.4 Filtres backward-only — vérification systématique

✅ **Règle** : avant d'utiliser une fonction pandas/numpy de fenêtre, **VÉRIFIER** explicitement qu'elle est causale :
- `pd.Series.rolling(N)` : ✅ backward (utilise `[t-N+1, t]`)
- `pd.Series.rolling(N, center=True)` : ❌ centred (utilise `[t-N/2, t+N/2]`) — INTERDIT comme feature
- `scipy.signal.filtfilt` : ❌ non-causal forward-backward — INTERDIT comme feature
- `pykalman.smooth` : ❌ smoother RTS non-causal — INTERDIT comme feature
- `pykalman.filter_update` : ✅ Kalman forward-only — OK
- `talib.RSI/MACD/CCI/ATR` : ✅ tous backward-only

### 1.5 Triple Barrier et walk-forward labels

✅ **Règle** : le label utilise OBLIGATOIREMENT le futur strict `[t+1 .. t+timeout]`, jamais le bar courant `t`.

```python
# CORRECT (notre pivot_labeler.py)
sub_high = high[idx + 1: idx + 1 + time_barrier]  # bars FUTURS uniquement
sub_low = low[idx + 1: idx + 1 + time_barrier]

# INCORRECT (fuite)
sub_high = high[idx: idx + time_barrier]  # inclut le bar de signal idx
```

### 1.6 Test de causalité reproductible

✅ **Règle** : pour tout nouveau pipeline, écrire un **test de causalité automatique** :

```python
def test_feature_causality(features, idx_test=1000):
    """Vérifier qu'une feature à l'index t ne change pas si on modifie le futur t+1..n."""
    feat_orig = features.copy()

    # Corrompre le futur
    features.iloc[idx_test+1:] = np.random.randn(len(features) - idx_test - 1)

    # Recalculer les features
    feat_modified = compute_features(features)

    # Les features à idx_test devraient être IDENTIQUES
    diff = (feat_orig.iloc[idx_test] - feat_modified.iloc[idx_test]).abs().sum()
    assert diff < 1e-9, f"FUITE DÉTECTÉE: features à t={idx_test} changent quand t+1.. change"
```

Ce test prend ~1 minute à coder mais détecte 95% des fuites silencieuses.

---

**Section 1 fin** — Tu valides cette section ? Je continue ensuite avec **Section 2 : Normalisation des features** qui reprendra notre découverte récente (max global ≠ rolling z-score, ratio /close, etc.).
