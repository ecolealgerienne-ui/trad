# STATUS v5.0 — PatchTST OHLCV-only Enriched (Triple Barrier sur pivots)

**Date** : 2026-04-26
**Asset** : BTC (single asset, BTCUSD 5min)
**Branche** : `claude/post-foundation-finetune-v14-PiOSL`
**Statut global** : ❌ **CLÔTURÉ — ÉCHEC EMPIRIQUEMENT VALIDÉ**
**Conclusion** : Mur OHLCV-only public confirmé sur 3 runs indépendants → **pivot v6 (données externes)**
**Approche précédente** : v4 = `experiments/foundation_finetune/` (clos Phase 14)

---

## Objectif

Construire un classifieur binaire **événementiel** (pas continu) qui valide ou rejette un signal d'entrée scalping 5min via PatchTST sur ~22 channels OHLCV-derived non-redondants, avec cible Triple Barrier sur niveaux pivot Camarilla.

**Question centrale** : une combinaison de représentations **fondamentalement différentes** des indicateurs continus (RSI/MACD/CCI testés en v1-v4) — bougies japonaises catégorielles + estimateurs microstructure + niveaux pivot + contexte multi-TF — permet-elle de casser le plafond ~44% precision top 1% identifié en Phase 14, en restant strictement dans BTCUSD OHLCV public ?

---

## Contexte v1 → v4

| Version | Approche | Verdict |
|---------|----------|---------|
| v1-v3 | RSI/CCI/MACD direction + dual-binary + clock-injected | Plafond 85-92% accuracy mais 33% WR trading |
| v3.0 | LSTM CNN crossfeat 30min | 49% Trans MACD, mais PnL négatif |
| v4.0 | FLKS sub-step + AQ-KF + XGBoost FLKS slopes | 96.3% accuracy, +870% PnL en backtest mais surfit |
| `foundation_finetune` Phase 1-14 | Chronos LoRA + meta-labeling Triple Barrier sur pente indicateur | **Mur empirique** : precision plafonne ~44% top 1% |

### Findings Phase 14 verrouillés
- RSI/MACD/CCI = **3 projections du même signal latent** (Pearson 1.0, recouvrement erreurs 80.6%)
- ATR + volume_spike + vc_score apportent plus que les indicateurs (XGBoost gain 212 vs 35)
- 71% du test = RANGE_LOW_VOL non-tradable
- Aucune architecture (CNN-LSTM, Chronos, XGBoost, RF, Logistic) ne casse le mur

> Le mur est dans la **nature de l'information**, pas dans l'architecture.

---

## Décision stratégique (2026-04-26)

| Décision | Justification |
|---|---|
| ❌ Pas de données externes (funding, OI, order book) cette itération | Choix utilisateur : exploiter d'abord à fond OHLCV |
| ✅ Rester BTCUSD 5min OHLCV public uniquement | Disponible immédiatement, validation rapide |
| ✅ Changer la **nature** de représentation, pas seulement l'archi | Phase 2.13 prouve que stacking d'indicateurs continus échoue |
| ✅ Reformuler la cible (Triple Barrier sur niveaux prix) | Phase 2.18 : labels alignés avec stratégie de trading réelle |
| ✅ Sampling **événementiel** (pas continu) | Élimine 71% RANGE_LOW_VOL non-tradable par construction |

### Hypothèse v5.0
Une combinaison de :
1. Bougies japonaises **catégorielles** (multi-hot)
2. Estimateurs **microstructure** dérivables d'OHLC (Corwin-Schultz, Yang-Zhang, Amihud)
3. Niveaux **discrets** (Pivot Camarilla, VWAP, Volume Profile)
4. Contexte **multi-timeframe** explicite (1h, 4h)
5. Cible Triple Barrier **réaliste** sur pivots

peut casser le plafond Phase 14. Si non → preuve que BTCUSD OHLCV seul est saturé → pivot v6 vers données externes.

---

## Architecture verrouillée

### Backbone
- **PatchTST** Channel-Independent ([Yu et al. ICLR 2023](https://arxiv.org/abs/2211.14730))
- Fenêtre input : **96 bougies** (8h sur 5min)
- Patches : **8 patches × 12 bougies** chacun (1h)

### Label
- **Triple Barrier Method ATR-adaptatif** :
  - **TP** : `entry ± 1.0 × ATR` (long/short) — adapte à la volatilité du moment, contrairement à Camarilla qui produit des labels incohérents par régime
  - **SL** : `bas_signal − 0.5×ATR` (long) ou `haut_signal + 0.5×ATR` (short)
  - **Time barrier** : 24 bougies (2h max)
  - **Label = 1** si TP touché avant SL et avant timeout
  - **Label = 0** sinon (SL ou timeout négatif)
  - Camarilla reste comme **feature input** (`dist_camarilla_nearest_norm`), le modèle peut l'utiliser pour décider
- Cible binaire (pas régression)

### Sampling
- **Event-driven** uniquement (pas de fenêtre glissante)
- Trigger = combinaison `pattern bougie reversal + proximité pivot + volume_zscore > 1.5`
- Volume estimé : 500-3000 events sur la période test (vs ~880k bougies brutes)

### Features (~22 channels OHLCV-only)

| Groupe | Channels | Description |
|---|---|---|
| **A — Bougies japonaises** | 8 | Patterns multi-hot (TA-Lib top 5-6) + body_ratio + upper/lower_wick_ratio + close_location_value + gap_norm |
| **B — Microstructure** | 5 | corwin_schultz_spread, garman_klass_vol, yang_zhang_vol, amihud_illiq, volume_zscore_20p |
| **C — Niveaux & contexte** | 5 | dist_vwap_session, dist_camarilla_nearest, dist_poc_5d, dist_high_20p, dist_low_20p (toutes normalisées par ATR) |
| **D — Multi-TF** | 4 | trend_1h_slope, trend_4h_slope, vol_1h_zscore, dist_vwap_daily |
| **E — Optionnel itération 2** | 3 | permutation_entropy_50p, hurst_dfa_100p, pacf_lag5 |

**Total itération 1** : 22 channels (groupes A+B+C+D)

---

## Roadmap d'implémentation

| # | Module | Description | Statut | Date |
|---|--------|-------------|--------|------|
| 0 | `STATUS_v5.0.md` | Document de suivi (ce fichier) | ✅ | 2026-04-26 |
| 1 | `experiments/patchtst_v5/README.md` | Création structure + README projet | ✅ | 2026-04-26 |
| 2 | `feature_builder.py` | Calcul des 22 channels (A+B+C+D) depuis CSV BTCUSD | ✅ | 2026-04-26 |
| 3 | `event_detector.py` | Détection des triggers (pattern + niveau + volume) | ✅ | 2026-04-26 |
| 4 | `pivot_labeler.py` | Triple Barrier ATR-adaptatif (TP/SL/timeout) sur chaque event | ✅ | 2026-04-26 |
| 5 | `dataset_builder.py` | Extraction fenêtres 96×N par event → NPZ train/val/test | ✅ | 2026-04-26 |
| 6 | `model.py` + `train.py` | PatchTST CI + boucle entraînement avec early stopping val AUC | ✅ | 2026-04-26 |
| 7 | `evaluate.py` | Threshold sweep + top-K%% sweep + calibration + per-segment | ✅ | 2026-04-26 |
| 8 | `backtest_realistic.py` | Backtest event-driven (Sharpe, MaxDD, Calmar, equity curves) | ✅ | 2026-04-26 |
| 9 | Décision finale | Validation v5 / pivot v6 selon critères de succès | ✅ | 2026-04-26 |

---

## Verdict final v5.0 (2026-04-26)

### ❌ ÉCHEC empiriquement validé sur 3 runs indépendants

| Run | Configuration | Top 1% precision | Best Sharpe | Verdict |
|---|---|---|---|---|
| 1 | sl_mode=from_signal, 24 ch | 63% (faux positif label asymétrique) | -1.08 | Trompeur |
| 2 | sl_mode=from_entry RR 1:1, 22 ch | **40.7%** (anti-prédictif au sommet) | -2.50 | Mur exposé |
| **3** | **+ Group E (entropy/Hurst/PACF), 27 ch** | **38.9%** (output collapse <0.55) | **-4.68** | **Mur renforcé** |

### Convergence avec Phase 14 du foundation_finetune

| Approche | Top 1% precision |
|---|---|
| Chronos LoRA + 22 features + Triple Barrier (Phase 14) | ~44% |
| PatchTST CI + 22 channels + RR 1:1 (v5 run 2) | 40.7% |
| PatchTST CI + 27 channels + Group E (v5 run 3) | 38.9% |

3 architectures, 3 formulations, 3 jeux de features → convergence systémique. **Le plafond est dans l'information, pas dans le modèle.**

### Ce qui a été éliminé comme cause possible (audits indépendants)

- ✅ Architecture PatchTST CI fidèle paper (audit 1)
- ✅ Formules numériques correctes (audit 2 — 85% confidence)
- ✅ Aucune fuite, splits propres, purge OK (audit 3 — 96% confidence)
- ✅ Triple Barrier walk-forward défensif (`np.argmax` guardé par `.any()`)
- ✅ RevIN per-sample, no cross-sample leakage
- ✅ Class balance, BCE pos_weight, AdamW + scheduler tous corrects
- ✅ 27 channels couvrant 5 axes informationnels (catégoriels, microstructure, niveaux, multi-TF, statistique)

### Décision : pivot v6 — données externes orthogonales

L'élimination par v5.0 de l'hypothèse OHLCV-only ouvre la voie à v6 avec sources d'information **vraiment orthogonales** :

| Source | API | Coût |
|---|---|---|
| Funding rate Binance perpétuels | binance.com/api/v3/funding-rate | Gratuit |
| Open Interest + ΔOI | binance.com/api/v3/futures-data | Gratuit |
| Long/Short ratio (top traders + global) | binance.com/api/v3/futures-data/topLongShortAccountRatio | Gratuit |
| Liquidations | Coinglass / Binance liquidation stream | Partiel gratuit |
| Premium index (futures - spot) | Binance | Gratuit |

Ces signaux sont structurellement absents d'OHLCV (positionnement, sentiment dérivés, événements forcés). Ils ont une vraie probabilité de casser le mur.

**Prochaine session** : créer `STATUS_v6.0.md` + `experiments/v6_external_data/` avec le pipeline d'ingestion + intégration au framework PatchTST existant.

Légende : ⏳ Pending — 🔄 In progress — ✅ Done — ❌ Blocked

---

## Findings hérités à NE PAS retester

Acquis empiriques validés sur 14 phases. v5.0 ne doit pas les retester :

| # | Finding | Source | Implication v5.0 |
|---|---|---|---|
| 1 | RSI/MACD/CCI ≈ même signal (Pearson 1.0) | Phase 2.13 | Aucun de ces 3 comme channel principal |
| 2 | Direction-Only > Dual-Binary (Force inutile) | Phase 2.8 | Cible binaire suffit (TP-touché vs non) |
| 3 | Kalman GLOBAL > Sliding Window | Phase 2.10 | Si filtre nécessaire, GLOBAL only |
| 4 | RANGE_LOW_VOL = 71% test = non-tradable | Phase 14 | Sampling événementiel résout par construction |
| 5 | Triple Barrier sur pente indicateur ne casse pas le mur | Phase 2.18 | TB doit être sur **prix** (pivots), pas indicateur |
| 6 | Octave Sliding Window catastrophique | 2026-01-08 | Aucun filtre sliding window |
| 7 | Stacking RSI/CCI/MACD échoue (0/9 succès) | 2026-01-06 | Pas de méta-modèle entre indicateurs corrélés |
| 8 | best_lag = +1 plafonne autocorr 0.93 | Phase 14 | Pas de prédiction t→t+1 sur close |
| 9 | KALMAN_PROCESS_VAR=0.01 régime toxique | slope_improvement | Si Kalman utilisé, σ²=1.155 (MLE) |
| 10 | Suroptimisation des seuils sur même split | v4.0 OOS | Walk-forward ou validation sur split distinct obligatoire |

---

## Critères de succès

### Métriques cibles (test out-of-sample)

| Métrique | Plancher | Cible | Idéal |
|---|---|---|---|
| Precision @ top 1% confidence | > 50% | > 60% | > 65% |
| Precision @ top 10% confidence | > 45% | > 55% | > 60% |
| Number of events triggered | > 500 | 1000-3000 | — |
| AUC ROC | > 0.55 | > 0.60 | > 0.65 |
| Backtest PnL Net (taker 0.04%) | positif | > +20% / an | > +50% / an |
| Sharpe Ratio annualisé | > 1.0 | > 2.0 | > 3.0 |
| Max Drawdown | < -25% | < -15% | < -10% |

### Critères de décision finale

| Scénario | Métriques | Action |
|---|---|---|
| ✅ **Succès** | Precision top 10% > 55% ET PnL net > +20% / an | Industrialisation, optimisation hyperparams, multi-asset |
| ⚠️ **Mitigé** | Precision 45-55% top 10%, PnL net ~0% | Tester groupe E (entropie/Hurst/PACF) puis revoir |
| ❌ **Échec** | Precision < 45% top 10% OU PnL net négatif | **Mur OHLCV-only confirmé** → pivot v6 (funding/OI externes) |

---

## Journal de décisions

| Date | Décision | Justification |
|------|----------|---------------|
| 2026-04-26 | Architecture PatchTST + Triple Barrier sur pivots verrouillée | Réduire bruit (event-driven) + nature représentation différente (catégoriel + microstructure) |
| 2026-04-26 | Pas de données externes (funding/OI/L2) | Choix utilisateur — exploiter d'abord à fond OHLCV existant |
| 2026-04-26 | 22 channels (A+B+C+D), groupe E reporté | Limiter complexité initiale, valider d'abord la mécanique |
| 2026-04-26 | Cible binaire TP-vs-non plutôt que régression | Aligné scalping pratique : "ce trade est-il profitable ?" |
| 2026-04-26 | Camarilla pivots préférés à Classic/Fibonacci | Niveaux plus serrés (H1/L1 typique 0.3-0.7%) adaptés 5min |
| 2026-04-26 | Aucune feature continue type RSI/MACD/CCI | Phase 2.13 prouve la redondance, ROI nul |
| 2026-04-26 | Drop CDLDOJI du trigger event_detector | Pattern non-directionnel (13.84% bars), signal s'auto-annule en somme signée |
| 2026-04-26 | Drop volume_zscore filter (option C) | Le modèle apprend à filtrer les trades parasites via prediction confidence — pas un filtre dur |
| 2026-04-26 | Triple Barrier ATR-adaptatif au lieu de Camarilla H1/L1 | Camarilla pur produit labels incohérents par régime de volatilité (trop tight en bull, trop loose en range). ATR adapte par construction. Camarilla reste comme **input feature** |
| 2026-04-26 | SL `from_entry` (symétrique) au lieu de `from_signal` (low/high) | Premier run avec SL=signal_low−0.5×ATR donnait WR top 1% = 63% mais PnL net négatif. Diagnostic: pour Engulfing patterns (73% events), bar large → signal_low loin de close → SL effective ≈ 1.5-2×ATR vs TP 1×ATR (asymétrie défavorable). `from_entry` impose RR 1:1 contrôlé, breakeven WR ≈ 50% (au lieu de ~58%) |
| 2026-04-26 | Audit code : drop channel dupliqué + warmup 300→400 + log purge | Audit indépendant a identifié 3 issues. (1) `dist_vwap_daily_norm` était identique à `dist_vwap_session_norm` (collinéarité parfaite) → remplacé par `dist_open_daily_norm` (distance à l'open quotidien, ancrage différent). (2) warmup_bars 300 insuffisant (vol_1h_zscore rolling 288 + lookback 96 → minimum 384) → 400 garantit zéro NaN dans les fenêtres. (3) Purge embargo dataset_builder loggé explicitement |
| 2026-04-26 | Run 2 (from_entry, RR 1:1) montre AUC 0.51, top 1% precision 40.7% < baseline → Option A activée | Run 1 (from_signal) montrait WR 63% top 1% mais SL=signal_low créait un biais directionnel sur Engulfing patterns. Run 2 avec from_entry expose que le modèle n'a pas de signal réel sur les 22 channels. Avant déclarer échec, **ajouter group E** (permutation_entropy_50p, hurst_dfa_100p, pacf_lag5) — features statistiques capturant signature stochastique différente |
| 2026-04-26 | Group E activé par défaut | 3 features statistiques calculées sur log_returns : permutation entropy (Bandt-Pompe m=3 window=50), Hurst exponent (R/S window=100 multi-lags), PACF lag 5 (Yule-Walker window=50). Total channels = 22 continus + 5 patterns = 27 |

---

## Risques identifiés

| # | Risque | Probabilité | Mitigation |
|---|--------|-------------|------------|
| 1 | Mur Phase 14 reste actif (info OHLC saturée) | **Moyenne** | Décision honnête de pivot v6 si critères ❌ |
| 2 | Trop peu d'events triggered (sampling trop strict) | Faible | Relaxer trigger conditions itérativement |
| 3 | Class imbalance Triple Barrier (Label 1 < 30%) | Moyenne | Class weights, focal loss, threshold tuning |
| 4 | Overfitting PatchTST sur ~3000 events | Moyenne | Walk-forward, dropout élevé, early stopping |
| 5 | Bougies japonaises rares en 5min crypto | Faible | Mesurer fréquence patterns en step exploratoire |
| 6 | Pivot Camarilla H1/L1 trop proche du prix → SL touché systématiquement | Moyenne | Calibrer ratio TP/SL sur ATR moyen ; tester H2/L2 |
| 7 | Suroptimisation des seuils trigger (data snooping) | Élevée | Walk-forward strict ou hold-out split distinct |

---

## Stack technique

- **Python 3.11+**
- **PyTorch 2.x** + `transformers` (PatchTST disponible HuggingFace)
- **TA-Lib** ou `pandas-ta` (patterns bougies)
- **NumPy / Pandas / SciPy**
- **scikit-learn** (calibration + métriques)
- **Données** : `data_trad/BTCUSD_all_5m.csv` (879,710 bougies, 2017-08 → 2026-01)

---

## Liens et références

- Approche précédente : [experiments/foundation_finetune/README.md](experiments/foundation_finetune/README.md) (Phase 1-14, clos)
- Calibration Kalman : [experiments/slope_improvement/final_report.md](experiments/slope_improvement/final_report.md)
- Statut v4 : [STATUS_v4.0.md](STATUS_v4.0.md)
- Findings consolidés : [CLAUDE.md](CLAUDE.md)
- López de Prado, *Advances in Financial Machine Learning* (2018) — Triple Barrier, Meta-Labeling
- Yu et al., *PatchTST: A Time Series is Worth 64 Words* (ICLR 2023)
- Corwin & Schultz, *A Simple Way to Estimate Bid-Ask Spreads from Daily High and Low Prices* (J. Finance 2012)
- Yang & Zhang, *Drift-independent volatility estimation* (J. Business 2000)
- Amihud, *Illiquidity and stock returns* (J. Financial Markets 2002)
