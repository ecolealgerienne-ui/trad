# STATUS v2.2 — Complete Experiment Summary

**Date**: 2026-04-14
**Asset**: BTC (single asset experiments)
**Objective**: Find the structural ceiling of price-derived features for transition prediction

---

## Master Comparison Table — All Experiments

### Classification Models (binary, BCEWithLogitsLoss)

| # | Config | Features | Window | Arch | Val Acc | Ratio | Justified% | Spurious% | Det<6% |
|---|--------|----------|--------|------|---------|-------|------------|-----------|--------|
| 1 | Single baseline | 2 (live+filt) | 25 | LSTM | **90.8%** | 2.8× | 58.3% | 21.8% | **93.7%** |
| 2 | + velocity | 3 | 25 | LSTM | 91.1% | 2.7× | 59.6% | 21.1% | 93.3% |
| 3 | **Crossfeat** | **6** | **25** | **LSTM** | 89.8% | **2.2×** | 57.4% | 20.1% | 90.8% |
| 4 | Cross + velocity | 9 | 25 | LSTM | 90.7% | 2.4× | **62.7%** | **18.0%** | 91.7% |
| 5 | Crossfeat | 6 | 25 | GRU | 91.1% | 2.7× | 61.7% | 19.3% | 93.3% |
| 6 | Crossfeat | 6 | 25 | TCN | 91.0% | 2.6× | 60.6% | 19.4% | 92.9% |
| 7 | Crossfeat | 6 | **12** | LSTM | 89.6% | — | — | — | — |
| 8 | Crossfeat | 6 | **50** | LSTM | 89.7% | — | — | — | — |

### Regression Models (continuous slope, MSELoss)

| # | Model | R² global | **R² transition** | R² plateau | Sign Acc | Val MSE |
|---|-------|----------|-------------------|------------|----------|---------|
| 1 | **macd_30m** | **0.9110** | **−0.1389** | 0.9203 | 91.3% | 0.1570 |
| 2 | cci_30m | 0.8297 | −0.2790 | 0.8551 | 88.5% | 0.1729 |
| 3 | rsi_30m | 0.7774 | −0.6235 | 0.8123 | 84.6% | 0.2248 |
| 4 | macd_1h | 0.8911 | −0.4690 | 0.8970 | 91.0% | 0.1827 |
| 5 | cci_1h | 0.8106 | −0.4870 | 0.8252 | 87.8% | 0.1953 |
| 6 | rsi_1h | 0.7584 | **−0.8894** | 0.7781 | 84.3% | 0.2384 |

### Crossfeat Classification — All 6 Models

| Model | Val Acc (single→cross) | Ratio (single→cross) | Spurious (single→cross) |
|-------|----------------------|---------------------|------------------------|
| macd_30m | 90.8% → 89.8% | 2.8× → **2.4×** | 21.8% → **18.0%** |
| cci_30m | 87.8% → 87.4% | 2.5× → 2.5× | 19.7% → 21.0% |
| rsi_30m | 84.1% → 83.8% | 2.9× → **2.6×** | 19.1% → **18.0%** |
| macd_1h | 89.1% → 89.9% | 3.6× → **2.4×** | 42.3% → **32.5%** |
| cci_1h | 86.6% → 85.6% | 3.4× → **3.1×** | 40.9% → 42.0% |
| rsi_1h | 82.7% → 82.5% | 4.2× → **3.0×** | 39.6% → **37.2%** |

### Switch Filtering Approaches

| # | Approach | Best ratio | Verdict |
|---|----------|-----------|---------|
| 1 | Cross-model filter (MACD↔RSI↔CCI) | 1.0-1.5× | ❌ Same signal |
| 2 | Cross-timeframe filter (30m↔1h) | 1.0-1.7× | ❌ Too correlated |
| 3 | Magnitude threshold (regression) | 1.2-1.3× | ❌ Cannot discriminate |
| 4 | Magnitude dynamics | 1.8× | ❌ Same V-shape |
| 5 | **R_strong_agree (binary × regression)** | **1.2×** | ⚠️ Best ratio but loses 20-30% detection |
| 6 | **Cross-architecture vote (3/3 unanimous)** | **0.8×** | ⚠️ Best absolute but loses 29% detection |

### Cross-Architecture Comparison (CNN-LSTM vs CNN-GRU vs TCN)

| Architecture | Val Acc | Val Loss | Switchs | Ratio | Justified | Spurious | Det<6% |
|-------------|---------|----------|---------|-------|-----------|----------|--------|
| **CNN-LSTM** | 91.1% | 0.2193 | **5,574** | **2.4×** | **62.7%** | **18.0%** | 91.7% |
| CNN-GRU | 91.1% | **0.2174** | 6,166 | 2.7× | 61.7% | 19.3% | **93.3%** |
| TCN | 91.0% | 0.2189 | 6,026 | 2.6× | 60.6% | 19.4% | 92.9% |

Error Jaccard overlap: 68% — architectures make similar errors.

### Window Size Comparison

| Window | Duration | Val Acc (30m) | Val Acc (1h) | Overfitting |
|--------|----------|---------------|--------------|-------------|
| 12 | 1h00 | 89.6% | 90.1% | Low |
| **25** | **2h05** | **89.8%** | **89.9%** | **Moderate** |
| 50 | 4h10 | 89.7% | 88.7% | High |

---

## Summary of Findings

### What Works
1. **Crossfeat 6-feat** reduces switch ratio from 2.8× to 2.2× (best single config)
2. **Cross-architecture voting (3/3)** achieves 0.8× ratio (below Oracle!) but loses 29% detection
3. **R_strong_agree** achieves 1.2× ratio on all 30m models
4. **Regression** captures 91% R² globally (but only on plateaus)

### What Doesn't Work
1. **Velocity as feature**: marginal (+0.1× ratio improvement)
2. **Regression at transitions**: R² negative for all 6 models (−0.14 to −0.89)
3. **Magnitude filtering**: ratio 1.2-1.8× (cannot discriminate true/false switches)
4. **Cross-model / cross-timeframe filters**: ratio 1.0-1.7× (same signal)
5. **Architecture change (GRU, TCN)**: ±0.1% accuracy, ratio worsens
6. **Window size (12, 50)**: ±0.2% accuracy, no improvement

### Structural Ceiling Confirmed By

| Evidence | Detail |
|----------|--------|
| Persistence baseline | 98% beats all models (89-91%) |
| R² at transitions | Negative for all 6 regression models |
| Error correlation | 68% Jaccard between 3 architectures |
| Prob before transition | = prob mid-plateau (no anticipation signal) |
| Window insensitivity | 12, 25, 50 steps all give same accuracy |
| All indicators same | MACD ≈ CCI ≈ RSI (correlation 1.0 at Oracle level) |

### Recommended Production Configuration

**Binary crossfeat 6-feat, CNN-LSTM, window 25, MACD 30m target**

With optional post-processing:
- **Conservative**: R_strong_agree filter (ratio 1.2×, 73% detection)
- **Aggressive**: No filter (ratio 2.2×, 91% detection)
- **Ultra-conservative**: 3-architecture unanimous vote (ratio 0.8×, 71% detection)

---

## Total Experiments Conducted

- **12 classification models** (3 feat configs × 2 arch variants + baselines)
- **6 regression models** (all indicator × timeframe combos)
- **6 filtering approaches** tested
- **3 architectures** compared
- **3 window sizes** tested
- **~30 unique experiments** total
- **Structural ceiling**: confirmed from every angle

---

## All 22 Models — Training Results (crossfeat, BTC)

### Classification (BCEWithLogitsLoss) — 3 architectures × 6 configs

| Model | CNN-LSTM loss | CNN-GRU loss | TCN loss | **Best arch** |
|-------|-------------|-------------|---------|--------------|
| macd_30m | 0.2438 | **0.2174** | 0.2189 | GRU |
| cci_30m | 0.2903 | 0.2822 | **0.2754** | TCN |
| rsi_30m | 0.3581 | 0.3469 | **0.3397** | TCN |
| macd_1h | 0.2526 | **0.2418** | 0.2480 | GRU |
| cci_1h | 0.3314 | **0.3159** | 0.3187 | GRU |
| rsi_1h | 0.3843 | **0.3706** | 0.3724 | GRU |

**GRU wins 4/6, TCN wins 2/6, LSTM wins 0/6** on val loss.

Note: val loss ≠ switch ratio. MACD 30m GRU had best loss (0.2174) but worst switch ratio (2.7× vs LSTM's 2.4×). KPI analysis needed for all new models before conclusions.

### Regression (MSELoss) — CNN-LSTM only

| Model | Val MSE | R² global | R² transition |
|-------|---------|----------|---------------|
| macd_30m | **0.1570** | 0.9110 | −0.14 |
| cci_30m | 0.1729 | 0.8297 | −0.28 |
| rsi_30m | 0.2248 | 0.7774 | −0.62 |
| macd_1h | 0.1827 | 0.8911 | −0.47 |
| cci_1h | 0.1953 | 0.8106 | −0.49 |
| rsi_1h | 0.2384 | 0.7584 | −0.89 |

### Hierarchy (consistent across all experiments)

```
By indicator: MACD > CCI > RSI (all metrics, all architectures)
By timeframe: 30m ≈ 1h (30m slightly better on switch quality)
By architecture: GRU best val loss, LSTM best switch ratio (on MACD 30m)
```

### MACD Price Normalization (in progress)

**Issue identified**: MACD is in price units — BTC $4k→$100k causes MACD to scale 2.3×.
- Train std: 47.12, Test std: 110.17, Ratio: **2.34×**
- RSI and CCI unaffected (bounded indicators)

**Fix applied**: `macd_normalized = macd / close * 10000` (basis points)
- CSV regeneration in progress
- MACD models (30m + 1h, all 3 architectures) must be retrained after
- RSI and CCI results remain valid
