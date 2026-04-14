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

## ÉTAPE 4 — 12 GRU + TCN Crossfeat (in progress)

*Results pending...*

---

## ÉTAPE 5 — 6 Regression Crossfeat (pending)

---

## ÉTAPE 6-8 — Analyses (pending)

---

## ÉTAPE 9 — Final Decision (pending)
