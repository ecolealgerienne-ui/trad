# PatchTST v5.0 — Filtrage de signaux scalping 5min par Triple Barrier sur pivots

**Statut** : 🟡 Planification (étape 1/9 — structure)
**Asset** : BTCUSD 5min uniquement (OHLCV public, pas de données externes)
**Statut tracking** : voir [`/STATUS_v5.0.md`](../../STATUS_v5.0.md) pour roadmap, critères de succès et journal de décisions.

---

## TL;DR

Classifieur binaire **événementiel** qui valide ou rejette un signal d'entrée scalping :

```
EventDetector  →  PivotLabeler  →  DatasetBuilder  →  PatchTST  →  Backtest
   (trigger)        (TP/SL/timeout)   (96 × 22 channels)   (binaire)   (réaliste)
```

- **Sampling** : event-driven (pattern bougie + niveau pivot + spike volume), ~500-3000 events sur la période test (vs 880k bougies brutes)
- **Cible** : Triple Barrier sur Camarilla H1/L1 — Label 1 si TP touché avant SL avant 24 bougies
- **Features** : 22 channels OHLCV-derived **non-redondants** (bougies catégorielles, microstructure, niveaux, multi-TF)
- **Backbone** : PatchTST channel-independent, fenêtre 96 bougies, patches 8 × 12

---

## Pourquoi cette approche

Les 14 phases précédentes (`experiments/foundation_finetune/`, v1-v4) ont accumulé une preuve empirique convergente : les **indicateurs continus OHLC-derived** (RSI/MACD/CCI/BOL) sont 3 projections du même signal latent (Pearson 1.0, Phase 2.13). Aucune architecture testée (CNN-LSTM, Chronos LoRA, XGBoost, RF, Logistic) n'a dépassé ~44% precision top 1%.

**Le mur est dans la nature de l'information, pas dans l'architecture.**

v5.0 change la **nature** de la représentation, en restant strictement OHLCV public :
- Patterns de bougies japonaises **catégoriels** (pas continus)
- Estimateurs **microstructure** dérivables d'OHLC (Corwin-Schultz, Yang-Zhang, Amihud)
- Niveaux **discrets** (Camarilla pivots, VWAP, Volume Profile)
- Sampling **événementiel** (pas continu) — élimine 71% RANGE_LOW_VOL non-tradable

Si v5.0 plafonne au même niveau que Phase 14 → preuve définitive que BTCUSD OHLCV seul est saturé en information → pivot v6 vers données externes (funding, OI, order book).

---

## Architecture

### Backbone PatchTST channel-independent

```
Input shape : (batch, 22 channels, 96 bougies)
   ↓
Patching   : (batch, 22 channels, 8 patches × 12 bougies)
   ↓
Linear projection per channel → embedding dim D
   ↓
Transformer encoder (channel-independent, weights shared across channels)
   ↓
Concat across channels + Classification head (Sigmoid)
   ↓
Output : score [0, 1] = P(TP touché avant SL avant timeout)
```

### Stack des 22 channels (groupes A+B+C+D)

| Groupe | # | Channels | Source |
|---|---|---|---|
| **A — Bougies japonaises** (8) | 1-5 | candle_pattern_top5 (multi-hot Hammer / Engulfing / Doji / Pin Bar / Star) | TA-Lib |
| | 6 | body_ratio = `\|close-open\| / (high-low)` | OHLC |
| | 7 | upper_wick_ratio + lower_wick_ratio | OHLC |
| | 8 | close_location_value = `(close-low) / (high-low)` | OHLC |
| | 9 | gap_norm = `(open - prev_close) / atr` | OHLC |
| **B — Microstructure** (5) | 10 | corwin_schultz_spread (proxy bid-ask) | HL sur 2 bougies |
| | 11 | garman_klass_vol (vol intra-bougie) | OHLC |
| | 12 | yang_zhang_vol (vol drift-robuste) | OHLC + open gaps |
| | 13 | amihud_illiq = `\|return\| / volume` | C + V |
| | 14 | volume_zscore_20p | V |
| **C — Niveaux & contexte** (5) | 15 | dist_vwap_session_norm (par ATR) | OHLC + V |
| | 16 | dist_camarilla_nearest_norm | OHLC daily |
| | 17 | dist_poc_5d_norm (Point of Control) | Volume profile 5j |
| | 18 | dist_high_20p_norm + dist_low_20p_norm | OHLC |
| **D — Multi-TF** (4) | 19 | trend_1h_slope (regression) | C 1h |
| | 20 | trend_4h_slope | C 4h |
| | 21 | vol_1h_zscore | V 1h |
| | 22 | dist_vwap_daily_norm | OHLC + V daily |

Groupe E (entropie / Hurst / PACF) reporté à itération 2 selon résultats.

### Triple Barrier

| Paramètre | Valeur | Justification |
|---|---|---|
| TP (long) | Camarilla H1 | Niveau classique scalping, ~0.3-0.7% en 5min crypto |
| TP (short) | Camarilla L1 | Symétrique |
| SL (long) | `bas_signal − 0.5×ATR` | Stop sous swing low + buffer ATR |
| SL (short) | `haut_signal + 0.5×ATR` | Symétrique |
| Time barrier | 24 bougies (2h) | Horizon scalping classique |
| Label | binaire | 1 si TP avant SL avant timeout, sinon 0 |

### Trigger (event detector)

Combinaison logique :
1. **Pattern bougie reversal** détecté (Hammer / Engulfing / Pin Bar / Star)
2. **Proximité** d'un niveau Camarilla (H1-H4 / L1-L4) ou VWAP (distance < 0.3×ATR)
3. **volume_zscore_20p > 1.5** (confirmation activité)

Volume estimé d'events : ~500-3000 sur période test (à mesurer en step exploratoire).

---

## Module map

| Fichier | Rôle | Dépendances | Statut |
|---|---|---|---|
| `__init__.py` | Module Python | — | ✅ |
| `README.md` | Documentation projet (ce fichier) | — | ✅ |
| `feature_builder.py` | Calcul des 22 channels depuis CSV BTCUSD | TA-Lib, pandas | ✅ |
| `event_detector.py` | Scan historique → liste des triggers | feature_builder | ✅ |
| `pivot_labeler.py` | Camarilla pivots + Triple Barrier par event | OHLC future | ⏳ |
| `dataset_builder.py` | Extraction fenêtres 96×22 + labels → NPZ | feature_builder + pivot_labeler | ⏳ |
| `model.py` | Architecture PatchTST channel-independent | torch, transformers | ⏳ |
| `train.py` | Training + early stopping + class weights | model + NPZ | ⏳ |
| `evaluate.py` | Métriques (precision/recall/AUC top-N%) | trained model + test NPZ | ⏳ |
| `backtest_realistic.py` | Backtest event-driven avec frais/slippage | predictions + OHLC future | ⏳ |

---

## Pipeline de données

```
data_trad/BTCUSD_all_5m.csv  (879,710 bougies, 2017-08 → 2026-01)
   │
   ├─→ feature_builder.py
   │     22 channels per bar (groupes A+B+C+D)
   │     Output: features_btc_v5.parquet (~500 MB)
   │
   ├─→ event_detector.py
   │     Scan: pattern + level + volume → liste timestamps
   │     Output: events_btc_v5.parquet (~500-3000 lignes)
   │
   ├─→ pivot_labeler.py
   │     Pour chaque event: Camarilla H1/L1 + Triple Barrier
   │     Output: labels_btc_v5.parquet (events + tp_price + sl_price + label + meta)
   │
   ├─→ dataset_builder.py
   │     Pour chaque event: extraction fenêtre [t-96:t] sur 22 channels
   │     Split: 70% train / 15% val / 15% test (chronologique strict)
   │     Output: data/patchtst_v5/{train,val,test}.npz
   │
   ├─→ train.py
   │     PatchTST CI + classifieur binaire + class weights
   │     Output: models/patchtst_v5/best_model.pth
   │
   └─→ evaluate.py + backtest_realistic.py
         Métriques + PnL sur test out-of-sample
         Output: results/patchtst_v5/{metrics.json, equity_curve.csv}
```

---

## Critères de validation

Voir [`STATUS_v5.0.md`](../../STATUS_v5.0.md) section "Critères de succès".

Synthèse :
- ✅ **Succès** si Precision @ top 10% > 55% ET PnL net > +20% / an
- ⚠️ **Mitigé** si Precision 45-55% ET PnL net ~0% → tester groupe E
- ❌ **Échec** si Precision < 45% OU PnL net négatif → pivot v6 (données externes)

---

## Commandes

```bash
# (À implémenter aux étapes suivantes — placeholders)

# Étape 2 : calcul des 22 channels
python -m experiments.patchtst_v5.feature_builder \
    --csv data_trad/BTCUSD_all_5m.csv \
    --output data/patchtst_v5/features_btc.parquet

# Étape 3-4 : détection events + labeling Triple Barrier
python -m experiments.patchtst_v5.event_detector \
    --features data/patchtst_v5/features_btc.parquet \
    --output data/patchtst_v5/events_btc.parquet

python -m experiments.patchtst_v5.pivot_labeler \
    --events data/patchtst_v5/events_btc.parquet \
    --csv data_trad/BTCUSD_all_5m.csv \
    --output data/patchtst_v5/labels_btc.parquet

# Étape 5 : construction dataset NPZ
python -m experiments.patchtst_v5.dataset_builder \
    --features data/patchtst_v5/features_btc.parquet \
    --labels data/patchtst_v5/labels_btc.parquet \
    --output-dir data/patchtst_v5/

# Étape 6 : training
python -m experiments.patchtst_v5.train \
    --train data/patchtst_v5/train.npz \
    --val data/patchtst_v5/val.npz \
    --epochs 100

# Étape 7-8 : évaluation + backtest
python -m experiments.patchtst_v5.evaluate \
    --model models/patchtst_v5/best_model.pth \
    --test data/patchtst_v5/test.npz

python -m experiments.patchtst_v5.backtest_realistic \
    --predictions results/patchtst_v5/predictions_test.npz \
    --csv data_trad/BTCUSD_all_5m.csv
```

---

## Liens

- [STATUS_v5.0.md](../../STATUS_v5.0.md) — tracking complet du projet
- [foundation_finetune/](../foundation_finetune/) — version précédente clôturée Phase 14
- [slope_improvement/](../slope_improvement/) — calibration Kalman AQ-KF / MLE
- [CLAUDE.md](../../CLAUDE.md) — findings consolidés du projet
