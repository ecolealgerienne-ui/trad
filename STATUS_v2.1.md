# STATUS v2.1 — Pilot Model Net_macd_30m Results & Analysis

**Date**: 2026-04-14
**Phase**: Pilot model training + detailed KPI analysis

---

## Summary

The pilot CNN-LSTM model (Net_macd_30m) achieves **91% accuracy** on predicting oracle direction labels, but deeper KPI analysis reveals the model knows **WHAT direction** but not **WHEN direction changes**. It produces 2.8× more switches than the oracle, with 22% spurious.

---

## Pipeline Architecture

### Data Flow

```
5min CSV (BTCUSD, ETHUSD, BNBUSD, ADAUSD, LTCUSD)
    ↓
prepare_multitf_csv.py
    ↓  Live OHLCV (cummax/cummin within bucket)
    ↓  Live indicators (incremental EMA, freeze at bucket closure)
    ↓  Live Kalman (filter_update, freeze at bucket closure) — CAUSAL
    ↓  Oracle labels (kf.smooth on resampled candles) — NON-CAUSAL
    ↓
{ASSET}USD_multitf_{indicators}.csv
    ↓
train_multitf.py
    ↓  Features: macd_30m_live + macd_30m_filtered (2 features)
    ↓  Target: oracle_label_macd_30m (binary)
    ↓  Split: chronological 70/15/15, gap=25, per asset
    ↓  Normalization: z-score per asset (stats from train only)
    ↓  Sequences: window=25 sliding window
    ↓
Net_macd_30m (CNN-LSTM, BCEWithLogitsLoss)
    ↓
Predictions: P(UP) ∈ [0, 1]
```

### Features vs Labels

| Component | Type | Method | Purpose |
|-----------|------|--------|---------|
| `macd_30m_live` | Feature (causal) | Incremental EMA, freeze at closure | What the model sees |
| `macd_30m_filtered` | Feature (causal) | Kalman filter_update, freeze at closure | Smoothed view for the model |
| `oracle_label_macd_30m` | Label (non-causal) | kf.smooth() on full series | What the model predicts |

### Model Architecture

```
Input: (batch, 25, 2)
  → Conv1d(2, 128, kernel=3) + ReLU + Dropout(0.1)
  → LayerNorm(128)
  → LSTM(128, 2 layers, dropout=0.2)
  → Linear(128, 64) + ReLU + Dropout(0.3)
  → Linear(64, 1)  — raw logit, no sigmoid
Loss: BCEWithLogitsLoss
Optimizer: Adam(lr=0.0001), grad_clip=1.0
Parameters: ~300k
```

---

## Oracle Results (reference ceiling)

### Oracle MACD 30m — 5 assets, test set (~15 months)

| Asset | Trades | Win Rate | PnL Net | Duration |
|-------|--------|----------|---------|----------|
| BTC | 2,285 | 59.5% | +891% | 57.7p |
| ETH | 2,288 | 65.3% | +1,663% | 57.7p |
| BNB | 2,258 | 62.3% | +1,198% | 56.9p |
| ADA | 2,120 | 67.6% | +2,456% | 57.3p |
| LTC | 2,207 | 66.6% | +2,108% | 57.5p |
| **TOTAL** | **11,158** | **64.2%** | **+8,316%** | **57.4p** |

### Oracle MACD 1h — 5 assets, test set

| Asset | Trades | Win Rate | PnL Net | Duration |
|-------|--------|----------|---------|----------|
| BTC | 1,125 | 66.0% | +826% | 117.3p |
| ETH | 1,110 | 70.6% | +1,447% | 118.9p |
| BNB | 1,094 | 71.6% | +1,077% | 117.4p |
| ADA | 1,010 | 73.9% | +2,024% | 120.3p |
| LTC | 1,083 | 72.2% | +1,709% | 117.2p |
| **TOTAL** | **5,422** | **70.8%** | **+7,083%** | **118.2p** |

### All Oracle Approaches Compared

| Approach | Kalman | Trades | WR | PnL Net | PF | Sharpe |
|----------|--------|--------|-----|---------|-----|--------|
| 5min Oracle (Phase 2.15) | smooth | 68,924 | 53.4% | +14,359% | 2.79 | 85.44 |
| **30min pur** | **smooth** | **11,158** | **64.2%** | **+8,316%** | **4.76** | **133.62** |
| **1h pur** | **smooth** | **5,422** | **70.8%** | **+7,083%** | **6.72** | **161.38** |
| 30min escalier (Kalman 5min) | filter_update | 30,528 | 33.0% | +421% | 1.05 | 4.02 |
| 30min live causal | filter_update | 64,810 | 8.7% | -12,880% | 0.09 | -323 |

**Key insight**: Higher timeframe = better quality per trade (WR 53% → 64% → 71%), but fewer trades. PnL net converges because 1h trades capture bigger moves.

---

## Model Training Results

### Net_macd_30m — BTC only, 128/128/64 architecture

| Metric | Value |
|--------|-------|
| Best epoch | 21 / 31 (early stop at 31) |
| Train accuracy | 91.5% |
| Val accuracy | **90.8%** |
| Val loss | 0.2197 |
| Train loss | 0.2061 |
| Overfitting | 0.7% (minimal) |
| Parameters | ~300k |

### Net_macd_1h — BTC only, 128/128/64 architecture

| Metric | Value |
|--------|-------|
| Best epoch | 30 / 40 (early stop at 40) |
| Train accuracy | 89.8% |
| Val accuracy | **89.1%** |
| Val loss | 0.2616 |
| Overfitting | 1.0% (minimal) |

### Comparison

| Timeframe | Val Acc | Val Loss | Convergence |
|-----------|---------|----------|-------------|
| **30m** | **90.8%** | **0.2197** | Epoch 21 |
| 1h | 89.1% | 0.2616 | Epoch 30 |

---

## Deep Evaluation — Baselines & Transition Analysis

### Why 91% Accuracy is Misleading

| Metric | 30m | 1h |
|--------|-----|-----|
| **Model accuracy** | 90.97% | 88.86% |
| **Persistence baseline** (label[t]=label[t-1]) | **98.26%** | **99.15%** |
| Majority class baseline | 50.24% | 50.09% |
| Model vs persistence | **-7.29%** | **-10.29%** |

The oracle label is **constant for ~57 steps** (30m) or **~117 steps** (1h) then changes. Simply predicting "same as last step" gives 98-99% accuracy. The model's 91% is **worse than this trivial baseline**.

### Transition Accuracy (the metric that matters)

| Metric | 30m | 1h |
|--------|-----|-----|
| Transitions in data | 11,151 (1.7%) | 1,124 (0.9%) |
| Continuations in data | 629,528 (98.3%) | 130,807 (99.1%) |
| **Model accuracy on transitions** | **53.2%** | **44.0%** |
| Model accuracy on continuations | 91.6% | 89.2% |
| Persistence on transitions | 0% (always wrong) | 0% |

The model predicts continuations well (trivial) but transitions at barely above random (53% for 30m, 44% for 1h).

### AUC ROC

| Timeframe | AUC |
|-----------|-----|
| 30m | **0.9718** |
| 1h | 0.9590 |

High AUC means the model separates UP from DOWN well — it knows the **current direction** with high confidence. But AUC doesn't measure transition detection.

### Confusion Matrix (30m test set)

```
                  Predicted DOWN  Predicted UP
  Actual DOWN          294,032          27,821
  Actual UP             30,033         288,793

  Precision UP:  91.2%  |  Recall UP:  90.6%
  Precision DOWN: 90.7% |  Recall DOWN: 91.4%
```

Balanced precision/recall — no directional bias. The issue is purely temporal (when), not directional (what).

---

## KPI Analysis — Trading Decision Quality

### KPI 1 — Detection Latency

How fast does the model detect real transitions?

| Metric | 30m | 1h |
|--------|-----|-----|
| Transitions detected | 99.9% | 99.8% |
| Instant detection (latency=0) | **53.3%** | 44.1% |
| Within 6 steps (<30min) | **93.7%** | 63.1% |
| Mean latency | 2.1 steps (10min) | 5.4 steps (27min) |
| Median latency | 0 steps | 1 step |
| Never detected | 0.1% | 0.2% |

**30m latency histogram:**
```
0 (instant):  5,936 (53.3%)
1-3:          2,145 (19.3%)
4-6:          2,362 (21.2%)
7-12:           644 ( 5.8%)
13-30:           53 ( 0.5%)
30+:              0 ( 0.0%)
```

**Verdict**: Detection is fast. 94% of transitions caught within 6 steps for 30m. This is the model's strength.

### KPI 2 — Plateau Oscillations (False Switches)

How noisy is the model between real transitions?

| Metric | 30m | 1h |
|--------|-----|-----|
| Model switches / Oracle switches | **2.8×** | **3.6×** |
| Plateaus with 0 false switches | 22.3% | 15.8% |
| Avg interval between model switches | 20.6 steps (103min) | 32.7 steps (164min) |
| Avg interval between oracle switches | 57.5 steps (287min) | 117.4 steps (587min) |

**30m switches per plateau:**
```
0 switches:  2,489 (22.3%) — clean
1 switch:    2,999 (26.9%)
2 switches:  1,650 (14.8%)
3 switches:  1,371 (12.3%)
4+ switches: 2,643 (23.7%) — very noisy
```

**Verdict**: The model produces ~3× more switches than needed. Only 22% of plateaus are clean. This would generate ~3× more trades and fees than the oracle.

### KPI 3 — Switch Precision

Are model switches near real transitions or spurious?

| Metric | 30m | 1h |
|--------|-----|-----|
| Justified (within ±6 steps) | **58.3%** | 25.2% |
| Spurious (>20 steps from any transition) | 21.8% | **42.3%** |
| Mean distance to nearest transition | 12.4 steps | 28.9 steps |

**30m distance distribution:**
```
0 (exact):   4,577 (14.7%)
1-3:         6,419 (20.6%)
4-6:         7,176 (23.0%)
7-12:        3,836 (12.3%)
13-20:       2,362 ( 7.6%)
20+ (spurious): 6,803 (21.8%)
```

**Verdict**: 58% of 30m switches are near real transitions (useful), but 22% are spurious (loss-generating). The 30m model is much more precise than 1h (58% vs 25% justified).

### KPI 4 — Probability Distribution

| Metric | 30m | 1h |
|--------|-----|-----|
| Predictions <0.1 or >0.9 | 74.0% | 72.4% |
| Grey zone [0.4, 0.6] | 4.9% | 5.6% |
| Mean prob before transition | 0.4957 | 0.5142 |
| Mean prob mid-plateau | 0.4955 | 0.4903 |
| **Difference** | **0.0002** | **0.0239** |

```
30m probability histogram:
[0.0, 0.1):  239,497 (37.4%) ████████████████████████████████████
[0.1, 0.2):   30,966 ( 4.8%) ████
[0.2, 0.3):   20,610 ( 3.2%) ███
[0.3, 0.4):   17,119 ( 2.7%) ██
[0.4, 0.5):   15,873 ( 2.5%) ██
[0.5, 0.6):   15,702 ( 2.5%) ██
[0.6, 0.7):   16,496 ( 2.6%) ██
[0.7, 0.8):   19,971 ( 3.1%) ███
[0.8, 0.9):   30,257 ( 4.7%) ████
[0.9, 1.0):  234,188 (36.6%) ████████████████████████████████████
```

**Verdict**: Model is very confident (bimodal, 74% near 0 or 1). But it does NOT see transitions coming — probability before transition ≈ probability mid-plateau (diff = 0.0002). The model reacts AFTER the transition, it doesn't anticipate.

---

## Core Diagnosis

### What the model does well
1. **Knows the current direction** — 91% accuracy, AUC 0.97, bimodal probabilities
2. **Detects transitions quickly after they happen** — 94% within 6 steps, median latency 0
3. **No directional bias** — balanced precision/recall UP/DOWN

### What the model does poorly
1. **Cannot anticipate transitions** — proba before transition = proba mid-plateau
2. **Too many false switches** — 2.8× more than oracle, 22% spurious
3. **Worse than persistence** — 91% accuracy vs 98% for "same as last step"

### Root cause
The label is **constant for ~57 steps** then changes. The model sees 2 features that update every 5min. It learns the direction well but has no signal for WHEN the direction will change. The transitions are 1.7% of the data — not enough examples to learn "about to transition" patterns.

### 30m vs 1h

30m is better on every KPI:
- Faster detection (median 0 vs 1)
- Less spurious switches (22% vs 42%)
- More justified switches (58% vs 25%)
- More clean plateaus (22% vs 16%)
- Higher accuracy (91% vs 89%)

---

## Files Produced

```
models/
├── best_model_macd_30m.pth         — trained model weights (128/128/64)
├── best_model_macd_1h.pth          — trained model weights (128/128/64)
├── training_history_macd_30m.json  — loss/accuracy per epoch
├── training_history_macd_1h.json
├── kpi_macd_30m.json               — KPI analysis results
└── kpi_macd_1h.json

data/prepared/
├── BTCUSD_multitf_macd.csv         — BTC only, MACD only (205 MB)
├── {ASSET}USD_multitf_macd_rsi_cci.csv — all indicators (in progress)
├── macd_30m_dataset.npz            — train/val/test sequences + predictions
├── macd_1h_dataset.npz
├── norm_stats_macd_30m.json        — per-asset normalization stats
└── norm_stats_macd_1h.json

scripts/
├── src/prepare_multitf_csv.py      — CSV generation (live + oracle labels)
├── src/train_multitf.py            — training script
├── src/analyze_predictions.py      — KPI analysis
├── tests/test_oracle_30min_pure.py — Oracle backtest (resample approach)
├── tests/test_oracle_multitf_live.py — Oracle backtest (CSV labels)
└── tests/eval_multitf_model.py     — baseline comparisons
```

---

## Next Steps (candidates)

1. **Add more features** — volume, returns, additional timeframes to help detect transitions
2. **Weighted loss on transitions** — upweight the 1.7% transition samples in the loss function
3. **Hysteresis / filtering** — post-process predictions to reduce false switches (e.g., require N consecutive steps before switching)
4. **Reformulate the problem** — predict "time until next transition" instead of direction
5. **Train on 5 assets** — current results are BTC only
6. **Backtest with ML predictions** — measure actual PnL with the noisy predictions vs oracle

---

## All 6 Models — Training Results (BTC only, 128/128/64 architecture)

### Accuracy Summary

| Model | Val Acc | Val Loss | Best Epoch | Train Acc | Overfitting |
|-------|---------|----------|------------|-----------|-------------|
| **MACD 30m** | **90.8%** | **0.2197** | 21 | 91.5% | 0.7% |
| CCI 30m | 87.8% | 0.2837 | 9 | 88.0% | 0.2% |
| RSI 30m | 84.1% | 0.3484 | 30 | 85.4% | 1.3% |
| **MACD 1h** | **89.1%** | **0.2616** | 30 | 89.8% | 0.7% |
| CCI 1h | 86.6% | 0.3075 | 21 | 87.6% | 1.0% |
| RSI 1h | 82.7% | 0.3766 | 28 | 83.5% | 0.8% |

### Observations

1. **Hierarchy**: MACD > CCI > RSI — consistent across both timeframes, and consistent with the original project (Phase 2.8). MACD (trend indicator) is the most predictable, RSI (speed oscillator) the least.

2. **30m > 1h**: 30m outperforms 1h by 1.2-1.7% on every indicator. 30m has more transitions to learn from (15k vs 7.6k direction changes), giving the model more signal.

3. **Zero overfitting**: All models show <1.5% gap between train and val accuracy. The z-score normalization + architecture size + early stopping keep the model well-regularized.

4. **Convergence speed**: CCI 30m converges fastest (epoch 9), RSI 30m slowest (epoch 30). MACD and CCI are "easier" targets.

5. **All models are confident**: Prediction std ~0.39-0.43 (bimodal distribution near 0 and 1), only ~5% in grey zone [0.4, 0.6].

### Models Saved

```
models/
├── best_model_macd_30m.pth    (val_acc=90.8%)
├── best_model_cci_30m.pth     (val_acc=87.8%)
├── best_model_rsi_30m.pth     (val_acc=84.1%)
├── best_model_macd_1h.pth     (val_acc=89.1%)
├── best_model_cci_1h.pth      (val_acc=86.6%)
├── best_model_rsi_1h.pth      (val_acc=82.7%)
├── training_history_*.json    (6 files)
└── kpi_macd_30m.json, kpi_macd_1h.json

data/prepared/
├── macd_30m_dataset.npz       (train/val/test + predictions)
├── cci_30m_dataset.npz
├── rsi_30m_dataset.npz
├── macd_1h_dataset.npz
├── cci_1h_dataset.npz
├── rsi_1h_dataset.npz
└── norm_stats_*.json          (6 files)
```

### Key Caveat (from KPI analysis on MACD 30m)

The 91% accuracy is misleading:
- **Persistence baseline** (label[t] = label[t-1]) gives **98.3%** accuracy
- **Transition accuracy** is only **53.2%** (barely above random)
- The model knows the **direction** but not **when direction changes**
- It produces **2.8× more switches** than the oracle

These caveats likely apply to all 6 models. The accuracy numbers measure "knows the current direction" — the model's actual utility for trading depends on transition detection quality, which requires dedicated KPI analysis for each model.

---

## All 6 Models — Signal Quality KPIs (BTC test set)

### Full Comparison Table

| Model | Acc% | Pers% | Trans% | AUC | N_trans | Lat_med | Lat_p90 | <6stp% | Sw_ratio | Clean% | Spur% | Grey% |
|-------|------|-------|--------|-----|---------|---------|---------|--------|----------|--------|-------|-------|
| **macd_30m** | **91.0** | 98.3 | **53.2** | **0.9718** | 11,151 | **0.0** | 6.0 | **93.7** | 2.8x | 22.3 | 21.8 | 4.9 |
| cci_30m | 88.4 | 97.9 | 50.3 | 0.9535 | 2,766 | 0.0 | 7.0 | 88.9 | **2.5x** | **25.9** | 19.7 | 7.1 |
| rsi_30m | 84.4 | 97.5 | 41.1 | 0.9269 | 3,269 | 1.0 | 7.0 | 88.1 | 2.9x | 19.0 | **19.1** | 10.3 |
| macd_1h | 88.9 | 99.1 | 44.0 | 0.9590 | 1,124 | 1.0 | 13.0 | 63.1 | 3.6x | 15.8 | 42.3 | 5.6 |
| cci_1h | 87.4 | 99.0 | 52.5 | 0.9482 | 1,324 | 0.0 | 12.0 | 68.9 | 3.4x | 22.5 | 40.9 | 7.4 |
| rsi_1h | 83.2 | 98.7 | 38.6 | 0.9163 | 1,650 | 2.0 | 12.0 | 67.0 | 4.2x | 14.8 | 39.6 | 10.8 |

### Signal Quality Ranking (not accuracy!)

| Rank | Model | Score | Trans% | <6stp% | Ratio | Spur% |
|------|-------|-------|--------|--------|-------|-------|
| 1 ★ | **cci_30m** | **1,507** | 50.3% | 88.9% | 2.5x | 19.7% |
| 2 | macd_30m | 1,464 | 53.2% | 93.7% | 2.8x | 21.8% |
| 3 | rsi_30m | 1,059 | 41.1% | 88.1% | 2.9x | 19.1% |
| 4 | cci_1h | 749 | 52.5% | 68.9% | 3.4x | 40.9% |
| 5 | macd_1h | 544 | 44.0% | 63.1% | 3.6x | 42.3% |
| 6 | rsi_1h | 441 | 38.6% | 67.0% | 4.2x | 39.6% |

### Rankings Comparison

```
Accuracy ranking:        macd_30m > macd_1h > cci_30m > cci_1h > rsi_30m > rsi_1h
Signal quality ranking:  cci_30m  > macd_30m > rsi_30m > cci_1h > macd_1h > rsi_1h
→ Rankings DIFFER — accuracy ≠ signal quality
```

### Key Insights

**1. CCI 30m is the best signal model, not MACD 30m**

Despite lower accuracy (88.4% vs 91.0%), CCI 30m wins on signal quality because it is the **quietest** model:
- Lowest switch ratio: 2.5× (vs 2.8× MACD)
- Most clean plateaus: 25.9% (vs 22.3% MACD)
- Decent transition accuracy: 50.3% (vs 53.2% MACD)

For trading, fewer false switches = fewer bad trades = better net performance.

**2. 30m systematically better than 1h on signal quality**

All 3 indicators in 30m outrank their 1h counterparts:
- 30m spurious: 19-22% vs 1h: 40-42% (2× more noise in 1h)
- 30m detection within 6 steps: 88-94% vs 1h: 63-69%
- 30m switch ratio: 2.5-2.9× vs 1h: 3.4-4.2×

**3. MACD 30m is the best at raw detection**

- Highest transition accuracy: 53.2%
- Fastest detection: median 0 steps (instant)
- Best AUC: 0.9718
- Best within-6-steps: 93.7%

But it generates more noise than CCI 30m (2.8× vs 2.5× ratio).

**4. RSI is consistently the worst signal**

Across both timeframes:
- Lowest transition accuracy (38-41%)
- Most grey zone probabilities (10%)
- Slowest detection (median 1-2 steps)

**5. All persistence baselines beat all models on raw accuracy**

| Timeframe | Persistence | Best model |
|-----------|-------------|------------|
| 30m | 97.5-98.3% | 91.0% (MACD) |
| 1h | 98.7-99.1% | 88.9% (MACD) |

This confirms that accuracy is the wrong metric. Signal quality (transition detection + noise level) is what matters.

**6. The real differentiator is noise, not detection**

All models detect 88-94% of 30m transitions within 6 steps (KPI 1 is good everywhere). The models differ most in **how noisy they are between transitions** (KPI 2 and 3). CCI 30m's discipline wins over MACD 30m's raw speed.

---

## Cross-Model Switch Discrimination Analysis

### Objective

Test whether the predictions of the OTHER two 30m models can help distinguish false switches from true switches in a given model. If cross-model signals discriminate well, we can build a filtering rule that eliminates false switches without losing true ones.

### Methodology

For each model X (macd_30m, cci_30m, rsi_30m):
1. Label each switch as TRUE (oracle transition within ±3 steps) or FALSE
2. At each switch, compute features from the other two models Y and Z
3. Compare feature distributions between true and false switches
4. Test 5 filtering rules and measure benefit/cost ratio

### Results: Cross-Model Signals Have No Discriminative Power

| Model | Best Rule | False Filtered | True Lost | **Ratio** | Verdict |
|-------|-----------|---------------|-----------|-----------|---------|
| macd_30m | R1: Both contradict | 31.4% | 22.9% | **1.4×** | Unusable |
| cci_30m | R3: Low confidence | 24.4% | 16.8% | **1.5×** | Unusable |
| rsi_30m | R1: Both contradict | 23.4% | 23.2% | **1.0×** | Useless |

A ratio of 1.0-1.5× means the rule filters nearly as many true switches as false ones. A useful rule would need ratio >3×. None of the 5 tested rules achieve this.

### Feature Distribution Gaps (False vs True Switches)

| Feature | MACD Gap | CCI Gap | RSI Gap | Useful? |
|---------|----------|---------|---------|---------|
| agree_before (other models) | 10-12pp | 8-11pp | 0-9pp | Weak |
| prob_mean (other models) | 0.011-0.016 | 0.002-0.004 | 0.002-0.013 | None |
| stability (other models) | 0.36-0.43 | 0.05-0.42 | 0.06-0.16 | None |

All gaps are too small to build a reliable discriminator.

### Root Cause

MACD, RSI, and CCI are projections of the same latent signal (correlation 1.000, proven in Phase 2.13 of the original project). When one model makes a false switch, the other two are equally confused — they provide no independent information.

> *"On ne peut pas voter entre trois miroirs du même objet."* — Phase 2.13 conclusion

### Conclusion

Cross-model filtering is NOT a viable path. To filter false switches, we would need:
- **Independent signals**: volume, volatility (ATR), order flow, funding rates
- **Post-processing**: hysteresis, holding minimum, confidence thresholds on the SAME model
- **Architecture changes**: predict transition probability directly instead of direction

---

## Cross-Timeframe Switch Discrimination (30m ↔ 1h)

### Results

| Filtering | Best Ratio | Best Rule |
|-----------|-----------|-----------|
| macd_30m filtered by macd_1h | 1.0× | 1h no switch before [t-3,t] |
| macd_1h filtered by macd_30m | 1.7× | 30m direction disagrees at t+3 |
| cci_30m filtered by cci_1h | 1.2× | 1h no switch in [t-3,t+3] |
| cci_1h filtered by cci_30m | 1.1× | 30m direction disagrees at t+3 |
| rsi_30m filtered by rsi_1h | 1.3× | 1h no switch in [t-3,t+3] |
| rsi_1h filtered by rsi_30m | 1.0× | 30m no switch after [t,t+3] |

All ratios 0.8-1.7×. No viable rule (need >3× to be useful).

### Conclusion

Cross-timeframe signals are as useless as cross-model signals. The 30m and 1h of the same indicator are highly correlated — when 30m makes a false switch, 1h is in the same state of confusion.

---

## Velocity Feature Experiment

### Hypothesis

The Kalman state contains [position, velocity]. Position was used (filtered column), but velocity (slope estimate) was never given to the model. The velocity approaching zero could signal imminent transitions — the missing information identified in KPI 4.

### Background

In the original project, velocity was used as a MODEL OUTPUT (Force STRONG/WEAK label), never as an INPUT feature. It was abandoned in Phase 2.8 because Force didn't help trading. But it was never tested as a feature for ML prediction.

### Implementation

Added `{ind}_{tf}_velocity` column to CSV (Kalman state[1], same freeze/provisional logic as position). Training with 3 features: `_live`, `_filtered`, `_velocity`.

### Results (BTC, MACD 30m, 128/128/64 architecture)

| Metric | 2-features | 3-features (+velocity) | Delta |
|--------|-----------|------------------------|-------|
| Val accuracy | 90.8% | 91.1% | +0.3% |
| **Switch ratio** | **2.8×** | **2.7×** | **-0.1×** |
| Clean plateaus | 22.3% | 26.0% | +3.7% |
| Spurious switches | 21.8% | 21.1% | -0.7% |
| Justified switches | 58.3% | 59.6% | +1.3% |
| Prob before transition | 0.4957 | 0.5119 | +0.016 |
| Grey zone | 4.9% | 4.0% | -0.9% |

### Verdict

**Marginal improvement across all KPIs, no significant change.** Switch ratio 2.7× >> 2.0× threshold. The velocity feature helps very slightly with plateau stability (+3.7% clean plateaus) but does not solve the transition detection problem.

The causal velocity is just the smoothed diff of the filtered position — information the model could already infer implicitly from the 25-step sequence of `macd_30m_filtered`.

---

## Final Diagnosis — Structural Limitation

### Three filtering approaches tested, all failed

| Approach | Best Ratio | Verdict |
|----------|-----------|---------|
| Cross-model (MACD↔RSI↔CCI, same TF) | 1.0-1.5× | Same latent signal |
| Cross-timeframe (30m↔1h, same indicator) | 1.0-1.7× | Too correlated |
| Additional feature (Kalman velocity) | 2.7× ratio (was 2.8×) | Marginal |

### Root cause confirmed

The model knows the **current direction** with 91% accuracy and AUC 0.97. But it **cannot predict when the direction will change** because:

1. **Transitions are inherently unpredictable** from past price data alone in crypto markets (sudden, news-driven, liquidation-driven)
2. **All features are derived from the same price** — MACD, RSI, CCI, Kalman position, Kalman velocity are all projections of close/high/low. No independent information.
3. **The non-causal smooth (Oracle) requires future data** — the quality of the Oracle labels comes precisely from seeing what happens AFTER the transition. No causal feature can replicate this.
4. **The persistence baseline (98.3%) beats all models (91%)** — confirming that for this label structure (constant blocks with rare transitions), prediction is trivial except at the hard points.

### What would be needed

To break through this ceiling, the model needs **information that is NOT derived from price**:
- **Volume spikes** (precede some transitions)
- **Order book imbalance** (bid/ask pressure)
- **Funding rates** (leverage positioning)
- **Liquidation data** (cascade events)
- **Sentiment / news** (external catalysts)
- **On-chain data** (whale movements)

Or a fundamentally different approach:
- **LLM-based analysis** of market context
- **Regime detection** from external signals
- **Event-driven** trading instead of continuous prediction

---

## Cross-Feature Experiment (6 features)

### Hypothesis

The single-indicator model (2 features: macd_live + macd_filtered) knows direction but can't anticipate transitions. Adding RSI and CCI as INPUT features (instead of separate models) might help the model detect regime changes — when indicators disagree, a transition may be imminent.

Previous cross-model FILTERING failed (ratio 1.0-1.5×), but giving raw features to the model lets it learn non-linear combinations that simple rules can't capture.

### Configuration

| Parameter | Single | Crossfeat |
|-----------|--------|-----------|
| Features | macd_30m_live, macd_30m_filtered (2) | + rsi_30m_live/filtered + cci_30m_live/filtered (6) |
| Target | oracle_label_macd_30m | oracle_label_macd_30m (same) |
| Architecture | CNN-LSTM 128/128/64 | CNN-LSTM 128/128/64 (same) |
| Input shape | (25, 2) | (25, 6) |

### Results (BTC, MACD 30m target)

| KPI | 2-feat (single) | 3-feat (+velocity) | **6-feat (crossfeat)** |
|-----|----------------|-------------------|----------------------|
| Val accuracy | **90.8%** | 91.1% | 89.8% |
| **Switch ratio** | **2.8×** | 2.7× | **2.2×** |
| Total switches | 6,432 | 6,065 | **5,020 (-22%)** |
| Plateaus 1 switch (ideal) | 26.9% | 25.3% | **33.3%** |
| Plateaus 4+ (noisy) | 23.7% | 21.1% | **17.6%** |
| Spurious switches | 21.8% | 21.1% | 20.1% |
| Detection <6 steps | **93.7%** | 93.3% | 90.8% |
| Latency median | **0** | 0 | 1 |
| Best epoch | 21 | 17 | 8 |

### Analysis

**Switch ratio drops from 2.8× to 2.2× (−21%)** — meets the ≥20% relative reduction criterion.

The crossfeat model trades accuracy (−1%) for **discipline**:
- 22% fewer total switches (5,020 vs 6,432)
- 33% of plateaus have exactly 1 switch (ideal: transition in + transition out)
- Only 17.6% of plateaus are very noisy (4+ switches) vs 23.7%

The cost is slightly slower detection (+1 step median latency) and faster overfitting (best epoch 8 vs 21).

### Why Cross-Features Work As Input But Failed As Filter

Cross-model **filtering** (Phase 2.13 analysis) tested simple binary rules ("do RSI and CCI agree?"). The ratios were 1.0-1.5× — useless.

Cross-model **features** work better because:
1. The CNN-LSTM learns **non-linear** combinations (not just agreement/disagreement)
2. It sees the **temporal evolution** of all 3 indicators over 25 steps
3. It can detect **divergence patterns** (MACD changing while RSI stays flat = suspicious)
4. The model learns to weight each indicator's contribution dynamically

### Verdict

The crossfeat approach is the first modification that meaningfully reduces false switches. However:
- **2.2× is still far from 1.0×** (oracle)
- **Val accuracy dropped 1%** (overfitting on extra features)
- **The model still can't anticipate transitions** (prob before = prob mid-plateau, diff = 0.006)

The fundamental limitation remains: transitions in crypto are unpredictable from price-derived features alone. But crossfeat provides the best noise reduction so far.

### 9-Feature Variant (crossfeat + velocity)

Added velocity (Kalman state[1]) for each indicator: 3 × (live + filtered + velocity) = 9 features.

| KPI | 6-feat (cross) | 9-feat (cross+vel) |
|-----|----------------|-------------------|
| Val accuracy | 89.8% | **90.7%** |
| Switch ratio | **2.2×** | 2.4× |
| Justified switches | 57.4% | **62.7%** |
| Spurious switches | 20.1% | **18.0%** |
| Plateaus 1 switch | **33.3%** | 27.6% |
| Best epoch | 8 | **3** (overfits faster) |

The 9-feat variant improves accuracy and switch precision but loses the discipline advantage (ratio 2.4× vs 2.2×). The velocity helps accuracy but doesn't reduce the overall switch count.

**Best configurations:**
- **For discipline (fewest switches):** 6-feat crossfeat (ratio 2.2×)
- **For precision (best switch quality):** 9-feat crossfeat+vel (62.7% justified, 18% spurious)

---

## Complete Experiment Summary

### All Configurations Tested (BTC, MACD 30m target)

| # | Config | Features | Val Acc | Ratio | Justified | Spurious | Best Epoch |
|---|--------|----------|---------|-------|-----------|----------|------------|
| 1 | Single | live + filtered (2) | **90.8%** | 2.8× | 58.3% | 21.8% | 21 |
| 2 | + velocity | + velocity (3) | 91.1% | 2.7× | 59.6% | 21.1% | 17 |
| 3 | **Crossfeat** | 3 ind × (live+filt) (6) | 89.8% | **2.2×** | 57.4% | 20.1% | 8 |
| 4 | Cross + vel | 3 ind × (live+filt+vel) (9) | 90.7% | 2.4× | **62.7%** | **18.0%** | 3 |

### What Worked

1. **Cross-indicator features (6-feat):** Only approach to meaningfully reduce switch ratio (2.8× → 2.2×, −21%)
2. **Velocity helps accuracy** when combined with cross-features (+0.9%)
3. **CNN-LSTM learns non-linear cross-indicator patterns** that simple rules couldn't capture

### What Didn't Work

1. **Velocity alone:** Marginal improvement (2.8× → 2.7×)
2. **Cross-model filtering rules:** Ratios 1.0-1.5× (same signal)
3. **Cross-timeframe filtering:** Ratios 1.0-1.7× (too correlated)
4. **All models still can't anticipate transitions:** prob before ≈ prob mid-plateau

### Structural Limitation Confirmed

Despite testing 4 feature configurations, 5 filtering approaches, and 6 models:
- **Best switch ratio: 2.2×** (still 2.2× more switches than Oracle)
- **Persistence baseline (98%) still beats all models (89-91%)**
- **Transition accuracy: 44-53%** (barely above random)
- **No model anticipates transitions** (probability gap < 0.01)

The signal exists (Oracle +8,316% PnL net) but requires future information (kf.smooth). Causal approximation from price-derived features alone hits a fundamental ceiling.

---

## Full Crossfeat Experiment — All 6 Models

### Configuration

| Timeframe | Features | Count |
|-----------|----------|-------|
| 30m targets | macd/rsi/cci × (live + filtered) at 30m | 6 |
| 1h targets | 6 × 30m + macd/rsi/cci × (live + filtered) at 1h | 12 |

### Training Results

| Model | Val Acc (single) | Val Acc (crossfeat) | Best Epoch (single) | Best Epoch (cross) |
|-------|-----------------|--------------------|--------------------|-------------------|
| macd_30m | 90.8% | 89.8% | 21 | 8 |
| cci_30m | 87.8% | 87.4% | 9 | 10 |
| rsi_30m | 84.1% | 83.8% | 30 | 13 |
| macd_1h | 89.1% | 89.9% | 30 | 3 |
| cci_1h | 86.6% | 85.6% | 21 | 5 |
| rsi_1h | 82.7% | 82.5% | 28 | 3 |

### KPI Comparison: Baseline vs Crossfeat

| Model | Ratio (base→cross) | Delta | Spurious (base→cross) | Trans% (base→cross) |
|-------|-------------------|-------|----------------------|---------------------|
| **macd_30m** | 2.8× → **2.4×** | **−14%** | 21.8% → **18.0%** | 53.2% → 50.0% |
| cci_30m | 2.5× → 2.5× | 0% | 19.7% → 21.0% | 50.3% → 42.1% |
| rsi_30m | 2.9× → **2.6×** | **−10%** | 19.1% → **18.0%** | 41.1% → 34.8% |
| **macd_1h** | 3.6× → **2.4×** | **−33%** | 42.3% → **32.5%** | 44.0% → 33.3% |
| cci_1h | 3.4× → **3.1×** | **−9%** | 40.9% → 42.0% | 52.5% → 39.6% |
| rsi_1h | 4.2× → **3.0×** | **−29%** | 39.6% → **37.2%** | 38.6% → 30.2% |

### Key Findings

1. **Crossfeat reduces switch ratio on 5/6 models** (CCI 30m unchanged)
2. **Biggest gains on 1h models**: MACD 1h −33% (3.6×→2.4×), RSI 1h −29% (4.2×→3.0×)
3. **Transition accuracy drops everywhere** (−3pp to −13pp) — model switches less but also detects fewer real transitions
4. **MACD 30m crossfeat remains overall best** (signal quality score 1594)
5. **1h models benefit most** because they were the noisiest (4.2× down to 3.0×)

### The Trade-off

```
Crossfeat = LESS noise + LESS detection

More features → model becomes more conservative → fewer switches overall
This reduces false switches (good) but also reduces true switch detection (bad)
Net effect: switch ratio improves but transition accuracy degrades
```

### Signal Quality Rankings

```
Baseline:   macd_30m > cci_30m > rsi_30m > cci_1h > macd_1h > rsi_1h
Crossfeat:  macd_30m > cci_30m > rsi_30m > macd_1h > cci_1h > rsi_1h
```

Hierarchy preserved. MACD 30m is the best model in both configurations.

---

## Recommended Next Directions

1. **Volume features** — only price-derived signal NOT yet tested, exists in CSV but never used as ML feature
2. **Non-price data** — funding rates, order book depth, liquidation data
3. **Post-processing** — hysteresis/holding minimum on the crossfeat predictions to reduce remaining false switches from 2.2-2.4× toward 1.5×
4. **LLM approach** — fundamentally different paradigm, already being explored in parallel
5. **Regularization** — crossfeat overfits fast (epoch 3-8); dropout increase, weight decay, or smaller model might help maintain accuracy while keeping the noise reduction benefit

---

## Regression Experiment — Continuous Slope Prediction

### Objective

Test if predicting the continuous slope (smoothed[t-1] - smoothed[t-2]) instead of binary direction captures more information. R² > 0.3 = substantial signal, R² < 0.1 = structural ceiling confirmed.

### Configuration

- **Features**: 9 crossfeat (live + filtered + velocity × 3 indicators at 30m)
- **Target**: `oracle_slope_macd_30m` (continuous, z-scored per asset)
- **Loss**: MSELoss
- **Architecture**: Same CNN-LSTM 128/128/64

### Training Results (BTC, MACD 30m)

| Config | Val MSE | Best Epoch |
|--------|---------|------------|
| 6-feat (no velocity) | 0.1803 | 13 |
| **9-feat (with velocity)** | **0.1570** | **8** |

Velocity improves regression MSE by −13% (relevant: velocity IS the slope estimate).

### Evaluation — Surface Metrics

| Metric | Value |
|--------|-------|
| R² (z-scored) | **0.9110** |
| Correlation | **0.9546** |
| MAE (real) | 3.80 |
| Sign accuracy | 91.30% |
| Persistence baseline | 98.27% |
| Test/Train std ratio | 2.21× ⚠️ |

### Deep Analysis — R² is Misleading

| Zone | R² | Correlation | % of data |
|------|-----|------------|-----------|
| **Plateau** (>3 steps from transition) | **0.9203** | 0.9597 | 87.9% |
| **Transition** (±3 steps) | **−0.1389** | 0.6036 | 12.1% |
| Global | 0.9110 | 0.9546 | 100% |

**R² is negative at transitions** — the model is WORSE than predicting the mean at the critical moments. The 0.91 global R² comes entirely from the 88% of easy plateau samples.

### Magnitude Threshold Analysis

Tested: does |predicted slope| > threshold help filter false switches?

| Threshold | False filtered | True filtered | **Ratio** |
|-----------|---------------|-------------|-----------|
| 0.05 | 33.7% | 25.2% | 1.3× |
| 0.10 | 53.0% | 42.4% | 1.3× |
| 0.20 | 74.4% | 61.1% | 1.2× |
| 0.50 | 93.4% | 83.9% | 1.1× |

**Ratio 1.2-1.3× everywhere** — the magnitude does NOT discriminate. True and false switches have very similar magnitudes (median 0.13 vs 0.09).

Mann-Whitney U test: p=3.57e-16 (statistically significant) but practically useless (gap too small).

### Regression Verdict

The regression gives an impressive global R²=0.91 but it is **plateau prediction** (R²=0.92 on plateaus, R²=−0.14 on transitions). At the moments that matter for trading (transitions and switches), the regression provides **no exploitable advantage** over binary classification.

---

## FINAL STRUCTURAL DIAGNOSIS

### All Approaches Tested

| # | Approach | Key metric | Verdict |
|---|----------|-----------|---------|
| 1 | Binary single (2 feat) | Switch ratio 2.8× | Baseline |
| 2 | + Velocity (3 feat) | 2.7× | Marginal |
| 3 | Binary crossfeat (6 feat) | **2.2×** | Best switch ratio |
| 4 | + Velocity crossfeat (9 feat) | 2.4× | Better precision, worse ratio |
| 5 | Cross-model filtering | 1.0-1.5× ratio | Failed (same signal) |
| 6 | Cross-timeframe filtering | 1.0-1.7× ratio | Failed (too correlated) |
| 7 | Regression (9 feat) | R²=0.91 (plateau), −0.14 (transition) | Plateau prediction only |
| 8 | Magnitude filter | 1.2-1.3× ratio | Cannot discriminate |

### Root Cause (Confirmed)

The model predicts **what** (direction: 91% accuracy, R²=0.91 on plateaus) but not **when** (transitions: R²=−0.14, switch ratio 2.2-2.8×).

This is structural:
1. **Transitions in crypto are unpredictable** from past price alone (sudden, news/liquidation driven)
2. **All features derive from the same price** — no independent information source
3. **The Oracle requires future data** (kf.smooth) — no causal feature can replicate it at transitions
4. **Persistence baseline (98%) beats all models (89-91%)** for this label structure

### What Would Be Needed to Break Through

**Non-price signals** (independent information):
- Volume, order book depth, bid/ask imbalance
- Funding rates, open interest, liquidation data
- Sentiment, news events, on-chain data

**Different paradigm**:
- LLM-based market context analysis
- Event-driven trading (react to events, don't predict)
- Regime detection from external signals

**Post-processing** (incremental improvement):
- Hysteresis on crossfeat 6-feat predictions (reduce 2.2× → ~1.5×)
- But cannot eliminate the fundamental 2× floor

---

## Magnitude Dynamics Analysis (Final)

### Hypothesis Tested

True switches might be preceded by a progressive "crescendo" in |predicted slope| while false switches are isolated spikes. If so, the temporal shape of magnitude could discriminate.

### Result: Hypothesis REJECTED

The trajectory is **descending** toward the switch (not ascending):

```
Offset   True mean   False mean   Gap
t-10     1.3552      0.8411       +0.51
t-5      0.7675      0.5050       +0.26
t-3      0.5508      0.3778       +0.17
t-1      0.2946      0.1812       +0.11
t (switch) 0.2567   0.1620       +0.09  ← minimum
t+1      0.3536      0.2697       +0.08
t+5      0.6864      0.5522       +0.13
```

Both true and false switches show the **same V-shape** pattern: magnitude decreases toward the switch (signal crosses zero) then increases after. The gap between true and false is a **level difference** (~0.1 at switch), not a shape difference. No dynamic rule can exploit this.

### Filtering Rules Tested

| Rule | True kept | False filtered | **Ratio** |
|------|-----------|---------------|-----------|
| R_avg_window > 0.05 | 94.1% | 11.0% | **1.8×** |
| R_max_recent > 0.1 | 93.2% | 11.5% | 1.7× |
| R_avg_window > 0.1 | 82.6% | 25.7% | 1.5× |
| R_max_recent > 0.2 | 79.9% | 29.2% | 1.5× |
| max>0.2 AND slope>0 | 10.3% | 90.2% | 1.0× |
| R_ascending [t-3,t] | 2.0% | 98.5% | 1.0× |

**Best ratio: 1.8×** — all rules below the 5× threshold needed for practical use.

### Complete Experiment Log (9 approaches, all failed to break 2×)

| # | Approach | Best ratio | Verdict |
|---|----------|-----------|---------|
| 1 | Binary single (2 feat) | 2.8× switch ratio | Baseline |
| 2 | + Velocity (3 feat) | 2.7× | Marginal |
| 3 | Binary crossfeat (6 feat) | 2.2× | Best binary |
| 4 | + Velocity crossfeat (9 feat) | 2.4× | Worse than 6-feat |
| 5 | Cross-model filtering | 1.0-1.5× filter ratio | Same signal |
| 6 | Cross-timeframe filtering | 1.0-1.7× | Too correlated |
| 7 | Regression R² | 0.91 global, −0.14 transitions | Plateau prediction |
| 8 | Magnitude threshold | 1.2-1.3× | Cannot discriminate |
| 9 | Magnitude dynamics | 1.8× max | Same shape, level diff only |

---

## All 6 Regression Models — Complete Results

### Training Summary

| Model | Val MSE | Best Epoch | Test/Train std |
|-------|---------|------------|----------------|
| macd_30m | **0.1570** | 8 | 2.21× ⚠️ |
| cci_30m | 0.1729 | 6 | 0.99× ✅ |
| rsi_30m | 0.2248 | 7 | 1.02× ✅ |
| macd_1h | 0.1827 | 6 | 2.21× ⚠️ |
| cci_1h | 0.1953 | 2 | 0.99× ✅ |
| rsi_1h | 0.2384 | 4 | 0.98× ✅ |

### Evaluation (test set, BTC only)

| Model | R² | Correlation | Sign Acc | Persistence | Kurt |
|-------|-----|------------|----------|-------------|------|
| **macd_30m** | **0.9110** | **0.9546** | 91.3% | 98.3% | 12.96 |
| cci_30m | 0.8297 | 0.9110 | 88.5% | 97.9% | 3.59 |
| rsi_30m | 0.7774 | 0.8817 | 84.6% | 97.5% | 0.76 |
| **macd_1h** | **0.8911** | **0.9449** | 91.0% | 99.2% | 12.33 |
| cci_1h | 0.8106 | 0.9004 | 87.8% | 99.0% | 4.18 |
| rsi_1h | 0.7584 | 0.8710 | 84.3% | 98.8% | 1.01 |

### Key Observations

1. **Hierarchy preserved**: MACD > CCI > RSI across all metrics, 30m ≈ 1h
2. **All R² > 0.75**: substantial continuous signal for all 6 models
3. **Sign accuracy ≈ classification accuracy**: regression doesn't improve binary prediction (91.3% vs 90.8% for MACD 30m)
4. **MACD test divergent** (std ×2.2): BTC MACD dynamics differ between train period ($4k-$60k) and test period ($60k-$100k). CCI and RSI are perfectly stable.
5. **Heavy tails on MACD** (kurtosis 12-13): rare but extreme prediction errors. CCI/RSI have lighter tails.
6. **Persistence still beats all models** on sign accuracy (97-99% vs 84-91%)

### Deep Analysis Reminder (from MACD 30m)

The global R²=0.91 is misleading:
- **R² plateau = 0.92** (87.9% of data, easy)
- **R² transition = −0.14** (12.1% of data, the part that matters)
- **Magnitude filter ratio: 1.2-1.3×** (cannot discriminate true/false switches)
- **Dynamics ratio: 1.8×** (same V-shape for true and false)

These patterns likely apply to all 6 models: high global R² driven by plateau prediction.

### Deep Analysis — All 6 Models Confirmed

| Model | R² global | **R² transitions** | R² plateaux | Best mag ratio |
|-------|----------|-------------------|-------------|---------------|
| macd_30m | 0.9110 | **−0.1389** | 0.9203 | 1.3× |
| cci_30m | 0.8297 | **−0.2790** | 0.8551 | — |
| rsi_30m | 0.7774 | **−0.6235** | 0.8123 | — |
| macd_1h | 0.8911 | **−0.4690** | 0.8970 | — |
| cci_1h | 0.8106 | **−0.4870** | 0.8252 | — |
| rsi_1h | 0.7584 | **−0.8894** | 0.7781 | — |

**All 6 models have NEGATIVE R² at transitions.** No model achieves R² > 0 at transitions, let alone the 0.2 threshold. The regression is definitively plateau prediction only.

Hierarchy at transitions: MACD (−0.14, −0.47) > CCI (−0.28, −0.49) > RSI (−0.62, −0.89). RSI is the worst, consistent with all previous experiments.

### Conclusion: Regression Closed

The continuous slope prediction provides no advantage over binary classification:
- Sign accuracy ≈ classification accuracy (91% vs 91%)
- R² at transitions is negative for all 6 models
- Magnitude cannot discriminate true vs false switches (ratio 1.2-1.3×)
- Magnitude dynamics show same V-shape for true and false (ratio 1.8×)

**The structural ceiling is confirmed from every possible angle.** Price-derived features cannot predict transition timing.

---

## Binary vs Regression — Final Comparison

### Phase 1: Binary wins on switch ratio (6/6 models)

| Model | Binary ratio | Regression ratio |
|-------|-------------|-----------------|
| macd_30m | **2.4×** | 2.5× |
| cci_30m | **2.5×** | 2.7× |
| rsi_30m | **2.6×** | 2.8× |
| macd_1h | **2.4×** | 3.0× |
| cci_1h | **3.1×** | 3.4× |
| rsi_1h | **3.0×** | 3.6× |

Binary classification produces fewer switches on every model. Regression is noisier.

### Phase 2: R_strong_agree achieves ratio 1.2× but loses detection

Combined rule: binary and regression agree AND |regression slope| > median true magnitude.

| Model | Baseline | R_strong_agree | Spurious% | Det<6% |
|-------|---------|---------------|-----------|--------|
| macd_30m | 2.4× | **1.2×** | **11.9%** | 73.2% (was 91.7%) |
| cci_30m | 2.5× | **1.2×** | **11.9%** | 72.9% (was 87.5%) |
| rsi_30m | 2.6× | **1.2×** | **12.5%** | 65.3% (was 84.2%) |
| macd_1h | 2.4× | **1.2×** | 26.6% | 44.0% (was 63.7%) |

R_strong_agree nearly matches Oracle switch count (1.2×) with low spurious (12%), but loses 20-30% of transition detections. Trade-off: much cleaner signals but misses more real transitions.

### Phase 3: Regression better at transitions (especially 1h)

| Model | Binary trans acc | Regression trans acc | Delta |
|-------|-----------------|---------------------|-------|
| macd_1h | 57.7% | **66.0%** | **+8.3%** |
| rsi_1h | 55.2% | **62.1%** | **+6.9%** |
| rsi_30m | 62.2% | **65.9%** | +3.7% |
| cci_30m | 65.1% | **66.5%** | +1.4% |
| macd_30m | 71.1% | 70.7% | −0.4% |

Regression improves transition accuracy on 5/6 models, with biggest gains on 1h (+6-8%).

### Final Verdict

- **Binary classification is the better approach for trading** (fewer switches, 6/6 models)
- **R_strong_agree** (combined) achieves best ratio (1.2×) but loses too much detection
- **Regression helps at transitions** (+3-8% accuracy) but generates more overall noise
- **No combination achieves ratio ≥ 5×** — structural ceiling confirmed

### Recommended Configuration

For production: **Binary crossfeat 6-feat (30m)** with optional R_strong_agree filter when lower trade frequency is acceptable. This gives:
- Switch ratio: 1.2× (with R_strong_agree) or 2.2× (without)
- Spurious: 12% (with) or 20% (without)
- Detection: 73% (with) or 91% (without)

---

## Architecture Comparison (CNN-LSTM vs CNN-GRU vs TCN)

### Hypothesis

The CNN-LSTM might be a bottleneck. Test CNN-GRU (KalmanNet-inspired) and TCN causal (dilated convolutions, no recurrence) to see if architecture matters.

### Results (MACD 30m, 9-feat crossfeat+velocity, BTC)

| Architecture | Val Acc | Val Loss | Switchs | **Ratio** | Justified | Spurious | Det<6% | Epoch |
|-------------|---------|----------|---------|----------|-----------|----------|--------|-------|
| **CNN-LSTM** | 91.1% | 0.2193 | **5,574** | **2.4×** | **62.7%** | **18.0%** | 91.7% | 17 |
| CNN-GRU | **91.1%** | **0.2174** | 6,166 | 2.7× | 61.7% | 19.3% | **93.3%** | 10 |
| TCN causal | 91.0% | 0.2189 | 6,026 | 2.6× | 60.6% | 19.4% | 92.9% | 15 |

### Conclusion

- **CNN-GRU** has marginally better val loss (0.2174 vs 0.2193) and detection (93.3% vs 91.7%)
- **TCN** converges slower (epoch 15) but similar accuracy
- **CNN-LSTM has the fewest switches** (5,574 vs 6k+) and best ratio (2.4× vs 2.6-2.7×)
- Spurious rate nearly identical (~18-19%) across all three

**The ceiling is in the data, not the architecture.** Switching from LSTM to GRU or TCN changes accuracy by ±0.1% and switch ratio by ±0.3×. The fundamental limitation (cannot predict transition timing from price features) persists regardless of architecture.

Criterion: ≥15% ratio improvement needed → NOT met (ratio worsened for GRU/TCN).
