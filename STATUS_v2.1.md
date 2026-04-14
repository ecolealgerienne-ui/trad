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
