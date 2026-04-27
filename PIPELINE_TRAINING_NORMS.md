# Normes Pipeline ML Trading — Checklist

**Version** : 3.0 — checklist pratique
**Objet** : éviter les erreurs récurrentes dans les pipelines de training pour stratégies de trading.
**Format** : ✅ règle — ❌ piège — ⚠️ subtilité.
**Référence détaillée** : voir `PIPELINE_NORMS_DETAILED.md`.

À parcourir AVANT chaque nouvel entraînement. Si une case n'est pas validée, ne pas lancer.

---

## 1. Causalité

✅ Une feature à `t` n'utilise QUE des données `[..., t-1, t]`.
✅ Le label peut utiliser le futur `[t+1..t+H]` (par construction).
✅ Pour features daily/hourly broadcastées au timeframe inférieur : `daily.shift(1)` puis `ffill()`.

❌ `pd.rolling(N, center=True)`, `scipy.filtfilt`, `pykalman.smooth()` comme features.
❌ Stats globales (mean, std, max) calculées sur tout le dataset → fit train-only.
❌ `bfill()`, `interpolate()` linéaire/temps comme features.

⚠️ `pd.rolling(N).mean()` à `t` inclut `t` (`[t-N+1..t]`) — c'est OK, pas besoin de `.shift(1)`.

**Test obligatoire** : corrompre `df[idx+1:]` aléatoirement, recalculer features, vérifier que `feature[idx]` est inchangé.

---

## 2. Stationarité des features

✅ Features bornées par construction (RSI, Stoch, ADX, MFI, %B) : OK telles quelles.
✅ Features price-scale non bornées (MACD line, trend slope, ATR brut) : transformer en ratio `/close × 100` ou rolling z-score.
✅ Cumulatives (OBV, equity) : prendre la pente rolling ou les diffs.

❌ Alimenter le modèle avec `close` brut, MACD en USD, slope de régression sur prix non normalisée.
❌ Considérer "borné" comme garantie de stationnarité (la distribution peut drift même si la borne reste).

⚠️ Le test ADF + KPSS combiné est plus fiable qu'un seul test (hypothèses opposées, consensus = confiance).

---

## 3. Normalisation

✅ `scaler.fit(X_train)` → `scaler.transform(X_val)` et `scaler.transform(X_test)`. JAMAIS de fit global.
✅ Z-score par défaut. Robust scaling si outliers fréquents. Min-Max uniquement pour features déjà bornées.
✅ Transformers time series (PatchTST etc.) : RevIN per-instance en + de la normalisation train-only.

❌ Croire que la normalisation **fixe le drift**. Diviser par `max_global` rescale mais ne change pas la **forme** des distributions train vs test (vérifié empiriquement : KS reste identique à 4 décimales).
❌ Refit le scaler sur train+val après hyperparam tuning → invalide les hyperparams.

---

## 4. Détection de drift (avant entraînement)

✅ KS test train vs test sur chaque feature avant tout training. Bloquer si KS > 0.20.
✅ Si KS critique sur une feature price-scale : transformer (ratio /close ou rolling z-score) plutôt qu'ignorer.

❌ Lancer un training sans avoir vérifié que train et test ont des distributions comparables.

⚠️ Seuils heuristiques : KS < 0.10 stable ; 0.10–0.20 modéré ; > 0.20 critique.

```python
from scipy.stats import ks_2samp
def check_drift(X_train, X_test, threshold=0.20):
    bad = [c for c in X_train.columns
           if ks_2samp(X_train[c], X_test[c])[0] > threshold]
    if bad:
        raise ValueError(f"Drift critique sur: {bad}")
```

---

## 5. Labels & Triple Barrier

✅ Calculer le **breakeven Win Rate** AVANT le training :
```
breakeven_WR = |mean_loss| / (mean_win + |mean_loss|)
```
Si breakeven > 60% → labels mal designés (TP/SL asymétriques en pratique).

✅ Frais et slippage **inclus dans le PnL des labels** (`pnl_after_fees_pct`).
✅ Triple Barrier sur prix (TP/SL en `entry ± k×ATR`) : clean RR symétrique. JAMAIS sur signal_low/high pour le SL si on veut un RR contrôlé.

❌ Croire qu'un Win Rate observé > 50% = profitable. Avec fees et RR < 1:1, breakeven peut être 70%+.
❌ Choisir TP/SL sans sanity check Oracle PnL (cf. § 9).

---

## 6. Class balance

✅ Mesurer Class=1 ratio sur train. Calculer `pos_weight = (1 - class1) / class1` pour BCEWithLogitsLoss.
✅ Métriques alignées sur l'objectif business : `precision @ top K%`, AUC, PR AUC. **Pas** uniquement accuracy (trompeuse en imbalance).

❌ Optimiser `accuracy` quand classe minoritaire = celle qui compte (le modèle prédit toujours majorité).
❌ Oversampling avec SMOTE sans purge embargo : crée de la fuite temporelle.

---

## 7. Split temporel + purge embargo

✅ Split **strictement chronologique** (jamais random pour time series).
✅ Embargo entre train→val→test ≥ horizon du label (si Triple Barrier H=24, embargo ≥ 24).
✅ Warmup ≥ max(rolling windows utilisés) + lookback de la fenêtre input.
   Exemple : feature uses `rolling(288)` + window input 96 → warmup ≥ 288 + 96 = 384.

❌ Cross-validation random `KFold` sur time series → leakage massif.
❌ Warmup insuffisant : les premiers events ont des features avec NaN dans la fenêtre lookback.

---

## 8. Diagnostic train vs val vs test

✅ Évaluer le model sur train, val, test avec **mêmes métriques**. Comparer.

| Pattern observé | Diagnostic | Action |
|---|---|---|
| Train AUC ≈ Val AUC ≈ 0.5 | Pas de signal appris | Drift features ou problème label |
| Train AUC >> Test AUC (gap > 0.10) | Overfitting OU distribution shift | Réduire capacité OU walk-forward |
| Train ≈ Val ≈ Test mais faible | Signal stable mais faible | Plafond information features |

❌ Conclure "le modèle a un signal" en regardant seulement le test. Sans comparer train, on ne sait pas si overfit ou plafond.

---

## 9. Sanity check Oracle PnL

✅ AVANT toute optimisation modèle, calculer l'Oracle PnL = somme des PnL des trades Label=1 uniquement (sélection parfaite).
✅ Vérifier Oracle annualisé > 50%/an. Si < 10%/an → labels mal calibrés (RR trop défavorable, ou frais trop hauts).

```python
import numpy as np
oracle_pnl = pnl_after_fees[y_true == 1].sum()
oracle_annualized = oracle_pnl / span_years
```

❌ Lancer du tuning hyperparams alors que l'Oracle lui-même est négatif → l'alpha n'existe pas dans les labels.

---

## 10. Audit pré-run obligatoire

Checklist à valider avant `python train.py` :

- [ ] **Causalité** : test feature corruption passé (§1)
- [ ] **Stationarité** : ADF/KPSS OK ou features transformées (§2)
- [ ] **Normalisation** : scaler fitted train-only (§3)
- [ ] **Drift** : KS test < 0.20 sur toutes features critiques (§4)
- [ ] **Breakeven WR** : calculé et < 60% (§5)
- [ ] **Class weights** : pos_weight passé à la loss (§6)
- [ ] **Split + embargo** : chronologique strict, embargo ≥ label horizon (§7)
- [ ] **Oracle PnL** : positif et annualisé > 50%/an (§9)

Si UN seul item n'est pas coché → **ne pas lancer**, fixer d'abord.

---

## Anti-patterns transverses (à connaître)

❌ **"Plus de features = mieux"** : ajouter des features redondantes (RSI 7/14/21 + Stochastic + Williams %R) gaspille la capacité du modèle. Préférer 1 représentant par cluster sémantique.

❌ **"Validation walk-forward = overkill"** : pour finance, c'est le standard, pas une option. Un seul split chronologique est insuffisant pour valider la robustesse aux régimes.

❌ **"Le modèle marche en production donc tout va bien"** : monitorer le drift quotidiennement (KS sur features clés), retrain selon décrochage (PSI > 0.25 ou perf drop > 20%).

❌ **"Min-max global pour rendre stationnaire"** : démontré empiriquement faux. La normalisation rescale, ne stationarise pas.

❌ **"Les indicateurs sont redondants donc inutiles"** : conclusion paradigme-dépendante. Une corrélation 1.0 sur des labels Kalman ne garantit pas la même chose sur des labels Triple Barrier différents.

---

## Workflow type d'un nouveau pipeline

1. **Explorer** : OHLCV → audit outliers/missing.
2. **Engineer features** : appliquer §2 (transformer non-stationnaires).
3. **Engineer labels** : Triple Barrier réaliste (§5), calculer breakeven WR.
4. **Sanity check Oracle** : §9.
5. **Build dataset** : split chronologique + purge embargo (§7).
6. **Drift check** : §4. Bloquer si critique.
7. **Train baseline** : régression logistique d'abord, puis modèle complexe.
8. **Diagnostic** : §8 (train vs val vs test).
9. **Backtest réaliste** : fees, slippage, sizing dynamique.
10. **Décision** : déployer si critères STATUS chiffrés atteints.
