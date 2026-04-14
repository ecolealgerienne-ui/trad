# STATUS v3.0 — Post-MACD Normalization Complete Rerun

**Date**: 2026-04-15
**Asset**: BTC (single asset)
**Fix applied**: MACD normalized by price (`macd / close * 10000`, basis points)

---

## MACD Normalization Fix

### Problem
MACD is in price units — BTC $4k→$100k caused MACD to scale 2.34×.
All crossfeat models used MACD as one of 6 features, so ALL results were affected.

### Fix
```python
macd_normalized = macd / close * 10000  # stable basis points
```

### Verification
```
BEFORE: MACD 30m - Train std: 47.12, Test std: 110.17, Ratio: 2.34×
AFTER:  MACD 30m - Train std: 19.22, Test std: 11.52, Ratio: 0.60×
RSI 30m (unchanged): Train std: 14.91, Test std: 15.17, Ratio: 1.02×
```

MACD ratio improved from 2.34× to 0.60× (slight over-correction but acceptable).

---

## ÉTAPE 2 — 6 LSTM Crossfeat Binaire (post-normalization)

### Results vs Before Normalization

| Model | Val Loss (before) | Val Loss (after) | Val Acc (before) | Val Acc (after) |
|-------|------------------|-----------------|-----------------|----------------|
| macd_30m | 0.2438 | **0.2382** | 89.8% | 89.6% |
| cci_30m | 0.2903 | **0.2855** | 87.4% | 87.6% |
| rsi_30m | 0.3581 | **0.3531** | 83.8% | 83.8% |
| macd_1h | 0.2526 | **0.2466** | 89.9% | 89.8% |
| cci_1h | 0.3314 | **0.3256** | 85.6% | 85.8% |
| rsi_1h | 0.3843 | **0.3826** | 82.5% | 82.3% |

**Loss improved everywhere** (−2 to −5%), accuracy unchanged (±0.2%).
Prediction std now stable: test/train ratio ≈ 1.0× (was 2.21× for MACD).

### Best Epochs

| Model | Best Epoch | Val Loss |
|-------|-----------|----------|
| macd_30m | 12 | 0.2382 |
| cci_30m | 10 | 0.2855 |
| rsi_30m | 13 | 0.3531 |
| macd_1h | 2 | 0.2466 |
| cci_1h | — | 0.3256 |
| rsi_1h | 2 | 0.3826 |

---

## ÉTAPE 3 — KPIs LSTM Crossfeat (post-normalization)

### Full KPI Table

| Model | Acc% | Trans% | AUC | Ratio | Justified% | Spurious% | Clean% | Det<6% |
|-------|------|--------|-----|-------|------------|-----------|--------|--------|
| macd_30m | 90.4 | 49.2 | 0.9686 | **2.5×** | — | 19.4% | 23.5% | 90.6% |
| cci_30m | 88.1 | 44.2 | 0.9528 | **2.5×** | — | 19.5% | 22.1% | 88.4% |
| rsi_30m | 84.0 | 39.3 | 0.9240 | **2.8×** | — | 17.6% | 18.7% | 86.6% |
| macd_1h | 90.5 | 38.5 | 0.9670 | 3.1× | — | 34.0% | 15.5% | 70.0% |
| cci_1h | 86.2 | 35.7 | 0.9402 | 3.4× | — | 41.8% | 15.0% | 64.7% |
| rsi_1h | 83.3 | 33.7 | 0.9160 | 3.4× | — | 38.4% | 13.9% | 64.1% |

### Before vs After Normalization — KPIs

| Model | Ratio (before) | Ratio (after) | Spurious (before) | Spurious (after) | Det<6 (before) | Det<6 (after) |
|-------|---------------|--------------|------------------|-----------------|---------------|--------------|
| macd_30m | 2.4× | **2.5×** | 18.0% | 19.4% | 91.7% | 90.6% |
| cci_30m | 2.5× | **2.5×** | 21.0% | **19.5%** | 87.5% | **88.4%** |
| rsi_30m | 2.6× | **2.8×** | 18.0% | **17.6%** | 84.2% | **86.6%** |
| macd_1h | 2.4× | **3.1×** | 32.5% | 34.0% | 63.7% | **70.0%** |
| cci_1h | 3.1× | **3.4×** | 42.0% | 41.8% | 68.3% | 64.7% |
| rsi_1h | 3.0× | **3.4×** | 37.2% | 38.4% | 60.8% | **64.1%** |

### Conclusion Étape 3

**Normalization did NOT improve KPIs.** Switch ratios slightly worse on most models.
The MACD scaling issue was NOT the cause of the structural ceiling.
Hierarchy preserved: MACD > CCI > RSI, 30m > 1h.

Signal quality ranking (post-norm):
```
macd_30m (1516) > cci_30m (1302) > rsi_30m (1036) > macd_1h (640) > cci_1h (486) > rsi_1h (454)
```

---

## ÉTAPE 4 — 12 GRU + TCN Crossfeat (complete)

### Val Loss — 3 Architectures Post-Normalization

| Model | LSTM | GRU | TCN | **Best** |
|-------|------|-----|-----|----------|
| macd_30m | 0.2382 | 0.2317 | **0.2243** | TCN |
| cci_30m | 0.2855 | 0.2812 | **0.2743** | TCN |
| rsi_30m | 0.3531 | 0.3428 | **0.3382** | TCN |
| macd_1h | 0.2466 | **0.2411** | 0.2475 | GRU |
| cci_1h | 0.3256 | **0.3115** | 0.3135 | GRU |
| rsi_1h | 0.3826 | **0.3685** | 0.3669 | GRU≈TCN |

### Best Epochs

| Model | LSTM | GRU | TCN |
|-------|------|-----|-----|
| macd_30m | 12 | 12 | 29 |
| cci_30m | 10 | 16 | 19 |
| rsi_30m | 13 | 21 | 32 |
| macd_1h | 2 | 4 | 3 |
| cci_1h | — | 11 | 12 |
| rsi_1h | 2 | 6 | 8 |

### Observations

- **TCN wins all 3 30m models** on val loss (0.2243, 0.2743, 0.3382)
- **GRU wins all 3 1h models** (0.2411, 0.3115, 0.3685)
- **LSTM wins 0/6** on val loss (consistent with pre-normalization finding)
- **TCN converges slowest** (epochs 19-32 for 30m) but reaches lowest loss
- **1h models converge very fast** (epochs 2-6 for LSTM, 3-11 for GRU/TCN)

### Before vs After Normalization — Architecture Ranking

```
BEFORE normalization: GRU wins 4/6, TCN wins 2/6, LSTM wins 0/6
AFTER normalization:  TCN wins 3/6, GRU wins 3/6, LSTM wins 0/6
```

MACD normalization shifted the ranking: TCN now dominates on 30m.

**IMPORTANT**: Val loss ≠ switch ratio. Previous analysis showed LSTM had best switch ratio despite worst loss. KPI analysis needed (step 6) to confirm.

---

## ÉTAPE 5 — 6 Regression Crossfeat (complete)

### Val MSE — Before vs After Normalization

| Model | Val MSE (before) | Val MSE (after) | Target std | Pred std test/train |
|-------|-----------------|----------------|-----------|-------------------|
| macd_30m | 0.1570 | **0.0311** | 3.62 | 0.57× |
| cci_30m | 0.1729 | **0.1820** | 21.82 | 0.97× ✅ |
| rsi_30m | 0.2248 | **0.2328** | 2.66 | 1.00× ✅ |
| macd_1h | 0.1827 | **0.0371** | 4.84 | 0.61× |
| cci_1h | 0.1953 | **0.2255** | 21.80 | 0.98× ✅ |
| rsi_1h | 0.2384 | **0.2572** | 2.71 | 0.99× ✅ |

### Analysis

- **MACD MSE dropped 5× (0.157→0.031, 0.183→0.037)** — but this is an AMPLITUDE effect, not a model improvement. MACD targets are now in smaller basis points, so MSE is mechanically lower.
- **CCI and RSI MSE slightly WORSE** (0.173→0.182, 0.225→0.233) — the MACD feature normalization slightly degraded the crossfeat for non-MACD targets.
- **MACD pred std ratio = 0.57-0.61×** — model predicts less dispersed values on val/test because MACD basis points have lower variance in recent data (BTC high price → smaller percentage moves).
- **CCI and RSI pred std stable** (0.97-1.00×) — as expected for bounded indicators.

### Best Epochs

| Model | Best Epoch | Val MSE |
|-------|-----------|---------|
| macd_30m | 30 | 0.0311 |
| cci_30m | 18 | 0.1820 |
| rsi_30m | 17 | 0.2328 |
| macd_1h | 7 | 0.0371 |
| cci_1h | 6 | 0.2255 |
| rsi_1h | 5 | 0.2572 |

### Key Takeaway

MACD MSE improvement is an artifact of scale change, not model improvement. R² conditionnel (step 6d) will reveal the true picture.

---

## ÉTAPE 6-8 — Analyses (complete)

### 6a. KPIs LSTM Crossfeat (identical to step 3)

macd_30m remains best (score 1516). Hierarchy unchanged.

### 6b. Cross-Architecture Vote (post-normalization)

| Metric | Before norm | After norm |
|--------|------------|-----------|
| Error Jaccard (3-way) | 68% | **56%** |
| Unanimous ratio | 0.8× | **0.5×** |
| Unanimous det<6% | 71.1% | **63.2%** |
| Unanimous spurious | 17.5% | **16.8%** |
| Filter "both agree" ratio | 1.6× | **2.0×** |

Error correlation **dropped from 68% to 56%** — normalization made architectures more diverse.
Unanimous vote now 0.5× (fewer switches than before) but loses more detection (63% vs 71%).

### 6c. Regression Evaluation (post-normalization)

| Model | R² global | Correlation | Sign Acc | Pred std ratio |
|-------|----------|-------------|----------|---------------|
| macd_30m | **0.9114** | **0.9547** | 90.6% | 0.57× |
| cci_30m | 0.8231 | 0.9075 | 88.4% | 0.97× ✅ |
| rsi_30m | 0.7731 | 0.8794 | 84.4% | 1.00× ✅ |
| macd_1h | **0.9052** | **0.9516** | 90.3% | 0.61× |
| cci_1h | 0.7922 | 0.8912 | 86.9% | 0.98× ✅ |
| rsi_1h | 0.7459 | 0.8638 | 83.9% | 0.99× ✅ |

MACD pred std ratio 0.57-0.61× (over-correction effect). CCI/RSI stable.

### 6d. Deep Regression — R² Conditionnel (THE critical table)

| Model | R² global | **R² transition** | R² plateau | Before norm R² trans |
|-------|----------|-------------------|------------|---------------------|
| macd_30m | 0.9114 | **−0.2369** | 0.9216 | −0.1389 |
| cci_30m | 0.8231 | **−0.4145** | 0.8514 | −0.2790 |
| rsi_30m | 0.7731 | **−0.6550** | 0.8087 | −0.6235 |
| macd_1h | 0.9052 | **−0.5933** | 0.9118 | −0.4690 |
| cci_1h | 0.7922 | **−0.6910** | 0.8089 | −0.4870 |
| rsi_1h | 0.7459 | **−1.0698** | 0.7676 | −0.8894 |

**R² at transitions got WORSE after normalization** for all 6 models!
- MACD 30m: −0.14 → **−0.24** (worse)
- MACD 1h: −0.47 → **−0.59** (worse)
- RSI 1h: −0.89 → **−1.07** (much worse)

The MACD normalization **degraded** transition prediction. The model was actually using the raw MACD scale as implicit information about price regime.

### Magnitude Threshold (MACD 30m only interesting result)

| Threshold | Switches | Ratio | Justified% | Spurious% |
|-----------|---------|-------|------------|-----------|
| 0.0 | 6,102 | 2.7× | 57.6% | 19.7% |
| **0.1** | **2,674** | **1.2×** | **60.5%** | **11.7%** |
| 0.2 | 2,048 | 0.9× | 45.9% | 8.6% |

MACD 30m with threshold 0.1 achieves **1.2× ratio, 60.5% justified, 11.7% spurious** — the best single-model result.

---

## FINAL COMPARISON — Post-Normalization

### Best Configurations (all tested on BTC test set)

| Rank | Config | Ratio | Spurious% | Det<6% | Notes |
|------|--------|-------|-----------|--------|-------|
| 1 | **Unanimous 3-arch vote** | **0.5×** | **16.8%** | 63.2% | Needs 3 models, loses 37% detection |
| 2 | **Reg MACD 30m thr=0.1** | **1.2×** | **11.7%** | ~79% | Single model, best spurious rate |
| 3 | **Majority 2/3 vote** | **1.8×** | 17.7% | 84.1% | Good balance |
| 4 | Crossfeat LSTM | 2.5× | 19.4% | 90.6% | Simplest, best detection |
| 5 | TCN crossfeat | 2.4× | 18.2% | 91.4% | Slightly better than LSTM |

### Impact of MACD Normalization

| Metric | Before | After | Verdict |
|--------|--------|-------|---------|
| Val loss (classification) | Baseline | Improved 2-5% | ✅ Better |
| Accuracy | ~89-91% | ~83-91% | ≈ Same |
| Switch ratio | 2.2-3.4× | 2.5-3.4× | ❌ Slightly worse |
| R² at transitions | −0.14 to −0.89 | −0.24 to −1.07 | ❌ Worse |
| Error Jaccard | 68% | 56% | ✅ More diverse |
| Pred std stability | 2.21× divergent | 0.57× over-corrected | ⚠️ Different bias |

**Conclusion**: MACD normalization was a mixed bag. It improved loss and diversity but degraded transition prediction. The raw MACD scale contained useful regime information.

---

## ÉTAPE 9 — Final Decision (pending)
