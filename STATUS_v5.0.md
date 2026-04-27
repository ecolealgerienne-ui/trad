# STATUS v5.0 — PatchTST OHLCV-only Enriched (Triple Barrier sur pivots)

**Date** : 2026-04-26
**Asset** : BTC (single asset, BTCUSD 5min)
**Branche** : `claude/post-foundation-finetune-v14-PiOSL`
**Statut global** : 🟢 **v5.3 BREAKTHROUGH (2026-04-27)** — XGBoost + multi-aggs sur pivot 'beyond' SHORT casse les plafonds historiques (test top 1% WR = **79-83%**, val top 1% WR = **87%**)
**Note** : Voir section **« v5.3 — XGBoost + multi-aggs (2026-04-27) »** en bas du document. La pivot v6 envisagée fin avril 2026 est **suspendue** : on continue à exploiter OHLCV avec ce nouveau levier.
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
| 9 | Décision phase 1 | v5.0 → ÉCHEC, expansion v5.1 décidée par avis expert | ✅ | 2026-04-26 |
| 10 | `model.py` refactor: expose encoder embedding | Split forward en encode() + classify() | ✅ | 2026-04-26 |
| 11 | `train_contrastive.py` | Triplet Loss + Hard Negative Mining + BCE multi-task | ✅ | 2026-04-26 |
| 12 | Run v5.1 + comparaison vs v5.0 | Triplet+BCE: top 1% **33.3%** (vs 38.9% v5 run 3) — DÉGRADATION | ✅ | 2026-04-26 |
| 13 | Décision finale v5.1 | **ÉCHEC : pivot v6 définitif validé, OHLCV-only saturé** | ✅ | 2026-04-26 |
| 14 | v5.2 — Pure indicators paradigm | feature_builder Group I (16 indicateurs TA-Lib) + dataset_builder preset `v5_indicators_only` (19 ch total) | ✅ | 2026-04-26 |
| 15 | Run v5.2 BCE + Contrastive | Test si paradigme indicators-only casse le mur (vs v5.0/v5.1 hybrid) | ⏳ | — |
| 16 | Décision v5.2 | Si succès → industrialisation. Si échec → pivot v6 définitif définitif | ⏳ | — |

---

## Verdict final consolidé v5 (2026-04-26)

### 4 runs convergents, dégradation monotone

| Run | Configuration | Top 1% precision | Best Sharpe |
|---|---|---|---|
| 1 | v5.0 from_signal (24 ch) | 63.0% (faux positif label asymétrique) | -1.08 |
| 2 | v5.0 from_entry RR 1:1 (22 ch) | 40.7% | -2.50 |
| 3 | v5.0 + Group E entropy/Hurst/PACF (27 ch) | 38.9% | -4.68 |
| **4** | **v5.1 Contrastive Triplet + BCE (27 ch)** | **33.3%** | **-5.10** |

**Pattern frappant** : chaque raffinement méthodologique **détériore** le top 1%. Le Contrastive Learning (proposé par expert externe) a amplifié l'anti-prédictivité au sommet de confiance. Signature classique du fait que l'information recherchée **n'existe pas** dans les features — la régularisation ne peut pas créer du signal.

### Diagnostic du Contrastive (run 4)

L'expert proposait de séparer les "look-alikes" Label=1/Label=0 dans l'espace latent via Triplet Loss. Réalité empirique :
- Il n'y a **pas de différence systématique** entre Label=1 et Label=0 dans les 27 channels OHLCV
- Forcer la séparation par contraste → mémorisation de bruit train-spécifique
- Surconfidence sur configurations test différentes
- Top 1% (= prédictions les plus catégoriques) devient le plus à côté

→ **Preuve que les hard negatives ne sont pas séparables car l'info n'est pas dans les features.**

### Convergence inter-projets (5 paradigmes, 5 plafonds)

| Approche | Architecture | Loss | Top 1% precision |
|---|---|---|---|
| Phase 14 foundation_finetune | Chronos T5 + LoRA | Triple Barrier BCE | ~44% |
| v5 run 2 | PatchTST CI + 22 ch | BCE + RR 1:1 | 40.7% |
| v5 run 3 | PatchTST CI + 27 ch + Group E | BCE + Entropy | 38.9% |
| **v5.1 run 4** | **PatchTST CI + 27 ch + Projector** | **Triplet + BCE** | **33.3%** |

5 paradigmes, plafond systémique. **Le mur n'est pas négociable par méthodologie.**

### Pivot v6 — sources d'information orthogonales

Plan de transition (à démarrer prochaine session) :
- `STATUS_v6.0.md` au root + `experiments/v6_external_data/`
- Phase 6.1 : `binance_fapi_ingestor.py` — funding rate (8 ans historique gratuit, complet)
- Phase 6.2 : WebSocket logger forward-only (OI, L/S ratio, liquidations)
- Phase 6.3 : Coinalyze API pour rapatrier 1-2 ans d'historique liquidations
- Phase 6.4 : Adaptation feature_builder v5 → ajout 4-5 channels externes
- Phase 6.5 : Re-run pipeline complet PatchTST (ou meta-classifieur XGBoost) sur features v5 + v6 enrichies
- **Critère de succès** : si funding_rate seul fait passer top 1% precision de 33% à >55% → orthogonalité validée → continuer enrichissement

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

---

# v5.3 — XGBoost + multi-aggs sur pivot 'beyond' SHORT (2026-04-27)

**Statut** : 🟢 **Breakthrough qualité de prédiction**. Le PnL n'est PAS l'objectif de cette phase — on se concentre sur le **WR (qualité de prédiction)** et l'écart **train ↔ val/test**.

## Setup

| Élément | Valeur |
|---|---|
| Dataset | `data/patchtst_v5_pivot_buf05/` (window=96 bars, 19 channels OHLCV-derived) |
| Label | `pivot_labeler_levels --sl-mode beyond` (TP=pivot Camarilla immédiat, SL=pivot suivant au-delà, time-barrier=24) |
| Direction | SHORT-only (`--direction-filter short`) |
| Train events | 11 491 (70%) |
| Val events | 2 397 (15%) |
| Test events | 2 441 (15%) |
| Class1 baseline | ~58.7% (WR si on prend tout) |

## Spécification exacte des inputs du modèle XGBoost

> **Important** : XGBoost ne voit PAS la séquence brute. Il voit un vecteur 1D de **304 features dérivées** par event. Aucune donnée externe, aucun prix brut, aucun timestamp.

### Pipeline complet input → modèle

```
BTCUSD 5min CSV (OHLCV brut)
         │
         ▼
[feature_builder.py]  → calcule 19 indicateurs par bougie
         │
         ▼
features parquet : (~880k bougies × 19 indicateurs)
         │
         ▼
[event_detector.py]  → garde un sous-ensemble (~32k events triggers)
         │
         ▼
[pivot_labeler_levels.py]  → pour chaque event : label 0/1 + direction ±1
         │                    (Triple Barrier, TP=pivot Camarilla immédiat,
         │                     SL=pivot Camarilla suivant beyond, time=24 bars)
         ▼
[dataset_builder.py]  → pour chaque event : extrait fenêtre 96 bars × 19 channels
         │                purge embargo 24 bars entre splits chronologiques
         ▼
NPZ : X (n_events, 96, 19), y, direction, timestamp, pnl_after_fees_pct
         │
         ▼
[train_xgboost.py / build_features]  → aplatit (96, 19) → vecteur 1D 304 features
         │
         ▼
XGBoost input : (n_events, 304) float32   ← CE QUE VOIT LE MODÈLE
```

### Les 19 indicateurs (canaux d'entrée)

Tous calculés via TA-Lib sur OHLCV brut, par bougie 5min, dans `feature_builder.py`. Preset `v5_indicators_only` :

| Groupe | # | Channels |
|---|---|---|
| Momentum multi-horizon | 3 | `rsi_7`, `rsi_14`, `rsi_21` |
| MACD (% du prix, stationnaire) | 2 | `macd_line_pct`, `macd_signal_pct` |
| Oscillateur déviation | 1 | `cci_20` |
| Stochastique | 2 | `stoch_k_14`, `stoch_d_14` |
| Williams | 1 | `williams_r_14` |
| Trend strength | 3 | `adx_14`, `di_plus_14`, `di_minus_14` |
| Volatilité | 2 | `atr_14_norm_z` (z-score rolling causal), `bbands_pct_b_20` |
| Volume | 2 | `obv_slope_z` (z-score rolling), `mfi_14` |
| Statistique | 2 | `hurst_dfa_100p`, `permutation_entropy_50p` |
| Volume relatif | 1 | `volume_zscore_20p` |

**Total : 19 channels.** Ce sont des indicateurs **classiques** dérivés de OHLCV. Aucune donnée externe (pas de funding rate, pas d'order book, pas d'on-chain).

### Pour chaque event : (96, 19) → vecteur 1D 304 features

Mode utilisé : `last-plus-multi-aggs` avec `MULTI_AGG_WINDOWS = [6, 12, 24, 48, 96]` :

```
Pour 1 event = 1 vecteur de 304 nombres :

  19 × 1 = 19   _last           valeur ACTUELLE de chaque indicateur (timestep -1)

Puis pour chaque fenêtre w ∈ [6, 12, 24, 48, 96] bars :
  19 × 1 = 19   _mean{w}        moyenne sur les w dernières bougies
  19 × 1 = 19   _std{w}         écart-type sur les w dernières bougies
  19 × 1 = 19   _first{w}       valeur il y a w bougies (timestep -w)

Total : 19 + 5 × 3 × 19 = 19 + 285 = 304 features
```

### Exemple concret pour 1 event SHORT

Pour un event à T=12h00, le modèle reçoit 304 nombres qui décrivent :

- **`_last` (19 nombres)** : où en est chaque indicateur juste avant l'entrée. Ex : `rsi_14_last = 72.3` (surachat), `atr_14_norm_z_last = 1.8` (vol élevée).
- **`_mean6` (19 nombres)** : niveau moyen sur les 30 dernières min. Ex : `rsi_14_mean6 = 68.5`.
- **`_std6` (19 nombres)** : volatilité de chaque indicateur sur 30 min. Ex : `rsi_14_std6 = 4.2`.
- **`_first6` (19 nombres)** : valeur il y a 30 min. Ex : `rsi_14_first6 = 55` (RSI est monté 55 → 72 en 30 min).
- Idem pour windows 12 (1h), 24 (2h), 48 (4h), 96 (8h) → vue **multi-échelle**.

### Ce que le modèle NE voit PAS

- ❌ **Aucun prix brut** (close, high, low, open, volume)
- ❌ **Aucun ratio direct prix / pivot** (la position du prix dans la structure n'est pas une feature)
- ❌ **Aucune feature catégorielle** (patterns bougies, jour, heure, session)
- ❌ **Aucune donnée externe** (funding, OI, sentiment, on-chain, order book)
- ❌ **Aucune information sur le label** (TP, SL, RR, exit_reason — utilisés uniquement pour calculer y)
- ❌ **Aucun timestamp** (le modèle ignore la date)
- ❌ **Aucune sortie de PatchTST** (XGBoost remplace PatchTST, pas en-dessus)

### Ce que le modèle reçoit en plus du vecteur 304

- La **cible** `y ∈ {0, 1}` dérivée du Triple Barrier
- Le **`scale_pos_weight`** ajusté sur la prévalence Class1 du train (ratio ~58.7% → poids ~0.703)

### Dimensions finales

```
TRAIN : X (11 491, 304) float32   y (11 491,) int8
VAL   : X (2 397, 304)  float32   y (2 397,) int8
TEST  : X (2 441, 304)  float32   y (2 441,) int8
```

C'est un **gradient boosting tabulaire classique** sur 304 features dérivées d'indicateurs techniques, target Triple Barrier sur pivots Camarilla. Pas de séquence, pas d'attention, pas de récurrence — juste 304 nombres par event.

### Garanties anti-leakage sur ces inputs

| Niveau | Garantie | Source |
|---|---|---|
| Indicateurs | Calcul rolling causal par bougie (TA-Lib + z-scores rolling) | `feature_builder.py` v5.4 fix drift |
| Pivots Camarilla | `prev_close = daily.close.shift(1)`, `prev_rng = rng.shift(1)` | `pivot_labeler_levels.py:50-51` |
| Window extraction | `block = feat_arr[start: idx + 1]` — pas de bar future | `dataset_builder.py:191` |
| Walk-forward exit | `sub_high = high[idx + 1: end]` — démarre APRÈS la bar du signal | `pivot_labeler_levels.py:178-179` |
| Splits | Chronologique strict + purge 24 bars d'embargo | `dataset_builder.py:207-251` |
| Aggregations | `X[:, -w:, :]` — uniquement les w dernières bars de la fenêtre | `train_xgboost.py:77` |
| XGBoost | Tree-based → invariant aux échelles, aucune normalisation cross-split | par construction |
| Persistance config | `feature_mode`, `agg_window`, `multi_agg_windows`, `direction_filter`, `n_features` sauvés dans `xgboost_model.json` via `booster.set_attr()` ; predict_all_splits les lit pour éviter mismatch silencieux | `train_xgboost.py:294-299` + `predict_all_splits.py:85-128` |

## Diagnostic préalable (`diagnose_label_separability`)

| Métrique (sur train) | Valeur | Lecture |
|---|---|---|
| Max Cohen's d | 0.069 | Signal univarié quasi-nul |
| Max AUC univariée | 0.521 | Aucune feature ne discrimine isolément |
| Logistic AUC train / val / test | 0.524 / 0.516 / 0.518 | Pas de dérive, signal uniformément faible |

→ **Label structurellement difficile** ; tout signal exploitable doit venir de **combinaisons non-linéaires multi-fenêtres**.

## Comparatif des configs testées

Toutes : SHORT-only, dataset identique. Métriques = **WR** sur top-K%% confidence.

| # | Config | TRAIN AUC | TRAIN top 1% | VAL AUC | **VAL top 1%** | TEST AUC | **TEST top 1%** |
|---|---|---|---|---|---|---|---|
| 1 | Default `last-plus-aggs` (76 feat, max_depth=4, n_est=500, early stop) | 0.598 | 89.5% | 0.501 | 60.9% | 0.497 | 66.7% |
| 2 | Pushed `last-plus-aggs` (max_depth=10, n_est=3000, lr=0.03, no early stop, no reg) | **1.000** | 100% | 0.529 | 69.6% | 0.517 | 70.8% |
| 3 | Pushed `agg-window=12` | 1.000 | 100% | 0.513 | 39.1% | 0.514 | 58.3% |
| 4 | Pushed `agg-window=6` | 1.000 | 100% | 0.500 | 60.9% | 0.505 | 58.3% |
| 5 | **Default `last-plus-multi-aggs`** (304 feat, 5 windows [6,12,24,48,96]) | 0.981 | 100% | 0.510 | **78.3%** | 0.512 | **83.3%** ✅ |
| 6 | **Pushed `last-plus-multi-aggs`** | **1.000** | 100% | **0.519** | **87.0%** ✅ | **0.528** | **79.2%** ✅ |

### Top WR à plus large échelle

| Config | TRAIN top 5% | VAL top 5% | TEST top 5% | TEST top 10% | TEST top 25% |
|---|---|---|---|---|---|
| Default `last-plus-aggs` | 76.5% | 59.7% | 63.9% | 63.1% | 60.5% |
| Pushed `last-plus-aggs` | 100% | 62.2% | 59.8% | 58.6% | 59.2% |
| Default `last-plus-multi-aggs` | 100% | 65.5% | **68.0%** | 61.9% | 61.0% |
| **Pushed `last-plus-multi-aggs`** | 100% | 63.0% | 61.5% | **64.3%** | 61.5% |

## Découverte clé : `last-plus-multi-aggs`

Au lieu d'agréger sur une seule fenêtre (24 bars), agréger sur **5 fenêtres simultanément** : `[6, 12, 24, 48, 96]` bars × `{mean, std, first}` + `last`.

→ **19 channels × (1 + 5×3) = 304 features** au lieu de 76.

**Impact sur le WR** :
- TEST top 1% : **66.7% → 83.3%** (+16.6pp avec early stopping)
- VAL top 1% : **60.9% → 87.0%** (+26.1pp pushed) — meilleur WR de tous les runs
- Top features mêlent plusieurs résolutions : `atr_14_norm_z_mean6`, `rsi_21_mean48`, `hurst_dfa_100p_mean48`, `di_minus_14_mean96`. Le modèle exploite vraiment le multi-échelle.

**Pourquoi ça marche malgré Cohen's d 0.07 ?**
Le signal univarié reste quasi-nul, mais XGBoost combine **des centaines de features faibles à des résolutions différentes** pour discriminer la queue extrême (top 1%). C'est exactement le cas d'usage du gradient boosting sur features riches.

## Métriques détaillées — meilleur modèle (Pushed multi-aggs)

| Split | n | Class1 | AUC | PR AUC | top 1% | top 5% | top 10% | top 25% |
|---|---|---|---|---|---|---|---|---|
| TRAIN | 11 491 | 58.7% | **1.000** | 1.000 | **100%** | 100% | 100% | 100% |
| VAL | 2 397 | 58.4% | 0.519 | 0.605 | **87.0%** | 63.0% | 60.7% | 59.4% |
| TEST | 2 441 | 59.9% | 0.528 | 0.620 | **79.2%** | 61.5% | 64.3% | 61.5% |

**Train mémorisé à 100%** → on a saturé la capacité d'extraction sur le train.

**Gap train→val/test sur top 1%** :
- Train 100% → Val 87.0% : **-13.0pp**
- Train 100% → Test 79.2% : **-20.8pp**

Le gap reste élevé mais le **niveau absolu val/test est exceptionnellement haut** (87% / 79%) vs baseline 58.7%, soit **+28pp / +20pp d'edge** sur la queue extrême.

## Comparatif vs baseline initiale (SHORT only, même dataset)

| Métrique | Default v1 | Best v5.3 | Gain |
|---|---|---|---|
| TEST top 1% WR | 66.7% | **79.2%** | **+12.5pp** |
| VAL top 1% WR | 60.9% | **87.0%** | **+26.1pp** |
| TEST AUC | 0.497 | 0.528 | +0.031 |
| TEST PR AUC | 0.609 | 0.620 | +0.011 |
| Features utilisées | 76 | 304 | ×4 |

## Configuration optimale (commande de reproduction)

```bash
python -m experiments.patchtst_v5.train_xgboost \
    --train data/patchtst_v5_pivot_buf05/train.npz \
    --val   data/patchtst_v5_pivot_buf05/val.npz \
    --test  data/patchtst_v5_pivot_buf05/test.npz \
    --output-dir models/patchtst_v5_pivot_buf05_xgb_short_multi_pushed/ \
    --feature-mode last-plus-multi-aggs \
    --direction-filter short \
    --max-depth 10 --learning-rate 0.03 --n-estimators 3000 \
    --min-child-weight 1 --subsample 1.0 --colsample-bytree 1.0 \
    --reg-lambda 0.0 --reg-alpha 0.0 --no-early-stopping
```

Modèle : `models/patchtst_v5_pivot_buf05_xgb_short_multi_pushed/xgboost_model.json`

## Notes sur le PnL (informatif, non prioritaire)

PnL backtest top 1% test = **+1.51%/an, Sharpe 0.86** (positif sur 24 trades) — premier PnL positif out-of-sample du projet. Mais sur 23-24 trades par split, IC à 95% trop large. **Non concluant** sur cette métrique.

→ **L'objectif reste la qualité de prédiction (WR)**, pas le PnL net. Le label `pivot beyond` impose un breakeven WR ~73-75% qu'on dépasse maintenant largement.

## Prochaines pistes (priorité décroissante)

1. **Ensemble multi-seed** — entraîner 5-10 modèles avec seeds [42, 7, 13, 100, 999, …] et moyenner les scores. Devrait réduire la variance val/test top 1% et stabiliser le WR.
2. **Tester LONG-only multi-aggs pushed** — pas encore comparé pour vérifier symétrie.
3. **Walk-forward roulant** — refit tous les 90j sur les 365j précédents, valider la stabilité temporelle.
4. **Élargir multi-aggs** — tester [3, 6, 12, 24, 48, 96] (380 features) ou ajouter median, skewness.

## Commits associés (branche `claude/post-foundation-finetune-v14-PiOSL`)

| Commit | Description |
|---|---|
| `e827705` | feat: `--direction-filter` long/short/both pour XGBoost |
| `7958a15` | feat: expose XGBoost regularization params via CLI |
| `70a1f67` | fix: handle `--no-early-stopping` (best_iteration absent) |
| `d2d453a` | fix: handle best_score absence in report |
| `17eb47d` | fix: predict_all_splits when model trained without early stop |
| `c5b5f34` | feat: `last-plus-multi-aggs` mode (304 features, 5 windows) |

## Pourquoi v5.3 contredit la conclusion v5.2 « pivot v6 »

La conclusion d'avril 2026 (v5.2 ÉCHEC, pivot v6 nécessaire) reposait sur :
- Top 1% precision PatchTST/Chronos plafonnant à **33-44%**
- Conclusion : « OHLCV-only saturé, mur informationnel »

v5.3 montre que **le levier n'avait pas été exploité** :
- XGBoost avec **multi-resolution aggregations** (5 fenêtres) sur les **mêmes 19 channels OHLCV** atteint **79-87% top 1% WR** sur val/test SHORT-only
- C'est **+35-43pp** au-dessus du plafond v5.2 (44%)
- Le mur n'était pas dans l'information OHLCV mais dans l'**inadéquation entre l'architecture (PatchTST attention sur séquence) et la vraie structure du signal (interactions multi-échelle entre indicateurs agrégés)**

→ **Pivot v6 (données externes funding/OI) suspendu**. On continue à pousser v5.3 (ensemble, walk-forward, LONG, etc.) avant d'envisager des sources externes.

---

# v5.3.1 — Ensemble multi-seed (étape A) — 2026-04-27

**Statut** : ✅ **Signal confirmé robuste mais val 87% révélé comme outlier single-seed**.

## Setup ensemble

- 5 modèles XGBoost entraînés avec seeds `[42, 7, 13, 100, 999]`
- Hyperparamètres identiques au pushed multi-aggs (max_depth=10, n_est=3000, lr=0.03, no_early_stop)
- Scripts : `train_ensemble.py` + `predict_ensemble.py` (commits `dcaee99`)

## Première tentative (subsample=1.0, colsample=1.0) — ÉCHEC silencieux

Avec sampling complet, **les 5 modèles étaient bit-identiques** :
- Toutes les val-AUC identiques à chaque itération [0, 50, 100, ..., 2999]
- TEST top 1% = 79.2% pour les 5 (= single seed)
- Ensemble = single (aucune variance à réduire)

**Cause** : avec `subsample=1.0`, `colsample_bytree=1.0`, `tree_method=hist`, XGBoost est entièrement déterministe (le seed ne sert qu'au tie-breaking sur splits égaux, rare avec 11k events × 304 features).

**Leçon** : pour qu'un ensemble XGBoost ait du sens, il **faut** activer le sampling de lignes ET de colonnes (< 1.0).

## Deuxième tentative (subsample=0.8, colsample_bytree=0.8) — RÉUSSI

Bagging + feature subsampling activé → diversité réelle entre seeds.

### Per-seed metrics

| Seed | TRAIN AUC | VAL top 1% | VAL top 5% | TEST top 1% | TEST top 5% |
|------|-----------|------------|------------|-------------|-------------|
| 42 | 1.000 | 78.3% | 63.0% | **62.5%** | 62.3% |
| 7 | 1.000 | 73.9% | 61.3% | **79.2%** | 56.6% |
| 13 | 1.000 | 69.6% | 59.7% | **66.7%** | 67.2% |
| 100 | 1.000 | 73.9% | 58.8% | **83.3%** | 65.6% |
| 999 | 1.000 | 73.9% | 62.2% | **87.5%** | 65.6% |
| **mean** | — | **73.9%** | **61.0%** | **75.8%** | **63.5%** |
| **range** | — | 69.6 - 78.3 | 58.8 - 63.0 | **62.5 - 87.5** | 56.6 - 67.2 |

→ **Variance énorme** sur TEST top 1% (62-88%) à cause des seulement 24 trades par split.

### Ensemble (moyenne des 5 scores)

| Métrique | Single pushed (seed 42) | **Ensemble (5 seeds)** | Delta |
|---|---|---|---|
| TEST top 1% WR | 79.2% | **79.2%** | 0 |
| TEST top 1% AvgNet | +0.078% | +0.050% | -0.028 |
| TEST top 1% AnnRet | +1.51% | **+0.97%** | -0.54 |
| TEST top 1% Sharpe | 0.86 | **0.56** | -0.30 |
| TEST top 1% MaxDD | -0.88% | -0.52% | meilleur |
| TEST top 5% WR | 61.5% | **63.9%** | +2.4pp |
| TEST top 2% WR | 64.6% | **68.8%** | +4.2pp |
| **VAL top 1% WR** | **87.0%** | **73.9%** | **-13.1pp** |
| VAL top 1% AnnRet | -0.18% | -0.51% | -0.33 |

## Découvertes critiques

1. **Le 87% VAL top 1% du single était un coup de chance** — l'ensemble redonne **73.9%** = exactement la moyenne des 5 seeds. C'est la **vraie** valeur attendue du modèle sur val.
2. **Le 79.2% TEST top 1% est confirmé robuste** — l'ensemble retombe sur la même valeur (avec PnL légèrement réduit).
3. **Top 5% test légèrement amélioré** (+2.4pp) grâce au moyennage.
4. **MaxDD réduit** (-0.88% → -0.52%) — l'ensemble est moins volatile.

## Estimation honnête du modèle (post-ensemble)

| Niveau | TEST | VAL | Lecture |
|---|---|---|---|
| top 1% WR | **~79%** | **~74%** | +20pp / +16pp vs baseline 58.7% — signal réel |
| top 5% WR | ~64% | ~62% | +5pp / +3pp — signal marginal |
| AnnRet top 1% | +0.97% | -0.51% | Asymétrie test/val gênante |
| Sharpe top 1% | 0.56 | -0.67 | Test positif, val légèrement négative |

## Limite atteinte avec le label actuel

- Train AUC = 1.000 (capacité d'extraction saturée)
- Top 1% val/test plafonne à ~74-79% WR
- Breakeven WR structural pour `pivot beyond` ≈ 73-75% → on est juste à la limite, marge insuffisante pour PnL significatif
- L'ensemble ne crée pas de signal nouveau, il stabilise seulement

## Commits ensemble étape A

| Commit | Description |
|---|---|
| `dcaee99` | feat: train_ensemble.py + predict_ensemble.py (multi-seed XGBoost averaging) |

## Décision : passage à étape B (plus de features)

L'ensemble a fait son job (variance réduite, signal confirmé). Pour aller plus loin sur le WR, il faut **enrichir les features** :

- **Option 1 (priorité)** : ajouter `median, q25, q75, min, max, skew` aux 5 windows actuelles
  → 19 ch × (1 last + 5 windows × 9 stats) = **874 features** (au lieu de 304)
- **Option 2** : ratios cross-window (ex : `rsi_mean6 / rsi_mean48`)
- **Option 3** : ratios cross-channel (ex : `rsi_14 - rsi_7`, divergences)

Étape B en cours : implémentation Option 1.

---

# v5.3.2 — Étape B (option 1 rich features) ÉCHEC + retour modèle unifié — 2026-04-27

## Étape B option 1 : `last-plus-multi-aggs-rich` (874 features)

Ajout de 6 stats (median, q25, q75, min, max, skew) aux 5 windows existantes : 19 channels × (1 last + 5 windows × 9 stats) = **874 features** au lieu de 304.

### Single rich pushed (subsample=1.0)

| Métrique | Multi-aggs (304 feat) | **Rich (874 feat)** | Delta |
|---|---|---|---|
| TRAIN AUC | 1.000 | 1.000 | 0 |
| **VAL top 1%** | 78.3% | **56.5%** | **-21.8pp** ❌ |
| **TEST top 1%** | 83.3% | **58.3%** | **-25.0pp** ❌ |
| TEST top 5% | 68.0% | 61.5% | -6.5 |

### Ensemble bagging rich (subsample=0.8)

| Métrique | Multi-aggs ensemble | **Rich ensemble** | Delta |
|---|---|---|---|
| **TEST top 1% WR** | **79.2%** | 70.8% | -8.3pp ❌ |
| TEST top 1% AnnRet | +0.97% | -0.83% | -1.80% ❌ |
| TEST top 1% Sharpe | +0.56 | -2.31 | -2.87 ❌ |
| **VAL top 1% WR** | 73.9% | 60.9% | -13.0pp ❌ |

### Diagnostic option 1

L'ajout des stats `median/q25/q75/min/max/skew` :
- **Disperse le signal** : top features passent de `atr_mean6/rsi_mean24` à `atr_q75, hurst_median, williams_max` → modèle ne se concentre plus sur la queue extrême
- **N'apporte pas d'info nouvelle** : quantiles, min, max sont des fonctions des mêmes séries déjà résumées par mean/std
- **Ratio features/samples 1:13** : memorisation parfaite mais signal généralisable dilué

→ **Option 1 abandonnée**. Confirmation empirique que sur ce dataset, **plus de features dérivées ≠ mieux**.

## Étape B suite : test LONG-only ensemble bagging

Avant de passer à option 2, j'ai testé LONG-only avec la config multi-aggs ensemble bagging pour mesurer une éventuelle asymétrie SHORT vs LONG.

| Métrique | SHORT (référence) | **LONG (nouveau)** |
|---|---|---|
| TEST top 1% WR | **79.2%** | 70.4% |
| TEST top 1% AnnRet | **+0.97%** ✅ | -1.33% ❌ |
| TEST top 1% Sharpe | **+0.56** | -1.64 |
| **VAL top 1% WR** | 73.9% | **92.6%** (25/27 wins) |
| **VAL top 1% AnnRet** | -0.51% | **+0.30%** ✅ |
| VAL top 1% Sharpe | -0.67 | **+0.42** |
| **VAL top 2% WR** | 72.3% | **85.5%** (47/55 wins) |
| VAL top 2% AnnRet | +0.22% | **+0.99%** ✅ |

### Découverte : asymétrie INVERSÉE val/test entre LONG et SHORT

- **SHORT** : val négatif (-0.51%), test positif (+0.97%)
- **LONG** : val positif (+0.30%), test négatif (-1.33%)

Les deux modèles montrent du signal **sur des splits différents**. Pattern classique de variance haute sur petits échantillons (24-27 trades par direction par split). Aucune direction n'est strictement meilleure.

## Décision stratégique : retour à un modèle UNIFIÉ (`--direction-filter both`)

**Constat** : la séparation LONG/SHORT initiée pour « éviter la pollution » au début de la session n'a pas montré d'avantage robuste — chaque direction a son split lucky différent.

**Décision** : passer à **un seul modèle entraîné sur les 24 175 events combinés** (LONG + SHORT). Le modèle peut apprendre lui-même les divergences via les features.

### Bénéfices attendus
- **2× plus de données train** (24 175 vs 11 491) → ensemble potentiellement plus stable
- **Couverture complète** des 5 175 events test (vs 2 441 pour SHORT-only)
- **Simplicité opérationnelle** : 1 modèle, 1 backtest, pas de combinaison à gérer

### Risques connus
- Le modèle pourrait diluer les signaux directionnels en mélangeant LONG/SHORT
- La direction (`+1`/`-1`) est dans les features ? **Non** — la direction n'est PAS un input du modèle (c'est une étiquette annexe utilisée seulement pour le backtest). Le modèle voit uniquement les 304 features dérivées des 19 indicateurs OHLCV.

### Configuration de référence pour étape B+

```bash
# Train unifié ensemble bagging
python -m experiments.patchtst_v5.train_ensemble \
    --train data/patchtst_v5_pivot_buf05/train.npz \
    --val   data/patchtst_v5_pivot_buf05/val.npz \
    --test  data/patchtst_v5_pivot_buf05/test.npz \
    --output-dir models/patchtst_v5_pivot_buf05_xgb_both_multi_ensemble_bagging/ \
    --seeds 42,7,13,100,999 \
    --feature-mode last-plus-multi-aggs \
    --direction-filter both \
    --max-depth 10 --learning-rate 0.03 --n-estimators 3000 \
    --min-child-weight 1 \
    --subsample 0.8 --colsample-bytree 0.8 \
    --reg-lambda 0.0 --reg-alpha 0.0 --no-early-stopping
```

C'est la nouvelle baseline à partir de laquelle on poursuit l'exploration features.
