# STATUS v2.0 — Multi-Timeframe Live Pipeline

**Date**: 2026-04-13
**Phase**: Validation Oracle multi-timeframe avec indicateurs live-style

---

## Résumé de la session

### Objectif
Tester si les indicateurs techniques (MACD, RSI, CCI) calculés sur des timeframes supérieurs (30min, 1h) avec estimation live toutes les 5min produisent un signal de trading rentable.

### Approche "Live-Style" (Binance API)
À chaque pas 5min, on reproduit ce que renverrait l'API Binance pour une bougie 30min/1h en cours de formation :
- OHLCV progressif (cummax/cummin/cumsum intra-bougie)
- Indicateurs avec EMA incrémentale : état figé sur les bougies closes, valeur provisoire recalculée à chaque pas 5min
- Kalman avec même logique freeze/provisional (filter_update causal pour les features live, smooth non-causal pour les oracle labels)

### Données
- **Source** : CSV 5min bruts (BTC, ETH, BNB, ADA, LTC, ~8.5 ans)
- **Indicateurs** : MACD(12,26,9), RSI(14), CCI(20)
- **Timeframes** : 30min (6 steps/bougie) et 1h (12 steps/bougie)
- **Kalman live** : PROCESS_VAR=0.01, MEASURE_VAR=0.1, filter_update (causal)
- **Kalman labels** : Mêmes paramètres, kf.smooth() (non-causal, RTS smoother)

---

## Architecture du pipeline

### Séparation Features / Labels

```
FEATURES (causales, ce que le modèle voit) :
  - OHLCV live 30m/1h : cummax, cummin, cumsum intra-bougie
  - Indicateurs live : EMA frozen/provisional, freeze au dernier bar du bucket
  - Kalman filtered live : filter_update frozen/provisional
  - Causalité : close_live[i] = close_5min[i], connu à temps i

LABELS (non-causaux, ce que le modèle doit prédire) :
  - Oracle labels : kf.smooth() sur indicateurs 30min/1h resampleés
  - Formule : label[t] = smoothed[t-1] > smoothed[t-2]
  - Forward-fill vers 5min sans shift (non-causal par construction)
```

### Logique freeze/provisional (identique pour EMA, RSI, CCI, Kalman)

```
À chaque pas 5min i :
  1. Calculer valeur provisoire depuis état_figé + observation_courante
  2. Sortie[i] = valeur provisoire  (jetable, ne s'accumule pas)

Si is_close[i] == True (dernier bar du bucket) :
  3. état_figé ← valeur provisoire  (le "vrai" état avance d'un cran)
```

La détection de closure utilise le changement de bucket (`floor(tf)` change), PAS `step == max_step`, pour gérer correctement les gaps dans les données.

### Scripts créés

| Script | Rôle |
|--------|------|
| `src/prepare_multitf_csv.py` | Génère CSV avec features live + Kalman + oracle labels |
| `tests/test_oracle_30min_pure.py` | Oracle test avec Kalman smooth sur bougies 30min/1h natives |
| `tests/test_oracle_multitf.py` | Oracle test avec Kalman sur données en escalier (abandonné) |
| `tests/test_oracle_multitf_live.py` | Oracle backtest lisant les oracle labels du CSV |

---

## Résultats Oracle (test set, ~15 mois)

### 5 assets — 30min pur (Kalman smooth sur bougies 30min)

| Asset | Trades | Win Rate | PnL Net | Durée moy |
|-------|--------|----------|---------|-----------|
| BTC | 2,285 | 59.5% | +891% | 57.7p |
| ETH | 2,288 | 65.3% | +1,663% | 57.7p |
| BNB | 2,258 | 62.3% | +1,198% | 56.9p |
| ADA | 2,120 | 67.6% | +2,456% | 57.3p |
| LTC | 2,207 | 66.6% | +2,108% | 57.5p |
| **TOTAL** | **11,158** | **64.2%** | **+8,316%** | **57.4p** |

### 5 assets — 1h pur (Kalman smooth sur bougies 1h)

| Asset | Trades | Win Rate | PnL Net | Durée moy |
|-------|--------|----------|---------|-----------|
| BTC | 1,125 | 66.0% | +826% | 117.3p |
| ETH | 1,110 | 70.6% | +1,447% | 118.9p |
| BNB | 1,094 | 71.6% | +1,077% | 117.4p |
| ADA | 1,010 | 73.9% | +2,024% | 120.3p |
| LTC | 1,083 | 72.2% | +1,709% | 117.2p |
| **TOTAL** | **5,422** | **70.8%** | **+7,083%** | **118.2p** |

### Tableau comparatif toutes approches (MACD, 5 assets sauf mention)

| Approche | Kalman | Trades | WR | PnL Net | PF | Sharpe |
|----------|--------|--------|-----|---------|-----|--------|
| 5min Oracle (Phase 2.15) | smooth | 68,924 | 53.4% | +14,359% | 2.79 | 85.44 |
| **30min pur** | **smooth** | **11,158** | **64.2%** | **+8,316%** | **4.76** | **133.62** |
| **1h pur** | **smooth** | **5,422** | **70.8%** | **+7,083%** | **6.72** | **161.38** |
| 30min escalier (Kalman 5min) | filter_update | 30,528 | 33.0% | +421% | 1.05 | 4.02 |
| 30min live causal | filter_update | 64,810 | 8.7% | -12,880% | 0.09 | -323 |
| **30m live Oracle** (BTC seul) | smooth labels | 2,285 | 59.5% | +891% | 3.63 | 121.58 |
| **1h live Oracle** (BTC seul) | smooth labels | 1,125 | 66.0% | +826% | 5.28 | 162.91 |

### Observations clés

1. **Plus le timeframe monte, meilleure est la qualité** : WR 53% (5min) → 64% (30min) → 71% (1h)
2. **Le PnL net converge** : le 1h perd peu en absolu vs 30min (-15%) malgré ÷2 trades
3. **Le PnL par trade double** : +0.39%/trade (30m) → +0.73%/trade (1h) pour BTC
4. **16/16 mois positifs** (30m et 1h purs sur 5 assets)
5. **Le Kalman causal seul ne suffit pas** : filter_update trop bruité → 65k trades, -12,880%
6. **L'approche live Oracle = même signal que le pur** (vérifié identique bit-à-bit sur BTC)

---

## Bugs corrigés dans la session

| Bug | Cause | Fix |
|-----|-------|-----|
| Validation FAIL MACD max_diff=1350 | Alignement positionnel vs resample+dropna | Alignement par timestamp (floor to tf) |
| RSI/CCI calculés malgré --indicators macd | Flag non propagé | Filtrage correct dans generate_multitf_csv |
| Kalman FAIL aux indices 1,2,3 | filter_update init ≠ kf.filter init | kf.filter sur closures + filter_update pour provisoires |
| EMA drift sur gaps (2,146 bars manquants) | step==max_step rate les buckets incomplets | Bucket-change mask (floor change detection) |
| Oracle backtest lit macd_30m_label au lieu de oracle_label | Nom de colonne incorrect | Fix label_col |

---

## Approches testées et résultats

| # | Approche | Résultat | Verdict |
|---|----------|----------|---------|
| 1 | 30min pur (Kalman smooth sur 147k candles) | +8,316% net, 64.2% WR | ✅ Excellent |
| 2 | 1h pur (Kalman smooth sur 73k candles) | +7,083% net, 70.8% WR | ✅ Meilleure qualité |
| 3 | 30min escalier (Kalman filter_update sur paliers 5min) | +421% net, 33% WR | ❌ Trop de trades |
| 4 | 30min live (Kalman causal sur indicateurs live) | -12,880% net, 8.7% WR | ❌ Catastrophique |
| 5 | Live avec oracle labels (smooth) | Identique au pur | ✅ Validé |

**Leçon** : Le Kalman smoother (non-causal) est essentiel pour la qualité du signal Oracle. Le filtre causal seul ne suffit pas. Le ML devra apprendre à approximer le signal smooth à partir de features causales.

---

## Prochaines étapes

### En cours
- [ ] Génération CSV complète : 5 assets × 3 indicateurs × 2 timeframes (30min + 1h)
  - Commande : `python src/prepare_multitf_csv.py --assets BTC ETH BNB ADA LTC --indicators macd rsi cci`

### À faire
- [ ] Oracle backtest sur 5 assets × 3 indicateurs (30m et 1h)
- [ ] Entraîner CNN-LSTM : features live (causales) → oracle labels (smooth, non-causales)
- [ ] Comparer accuracy ML vs Oracle (plafond théorique)
- [ ] Évaluer si le ML peut approximer le signal smooth suffisamment bien pour être profitable

### Questions ouvertes
- Le 1h est-il meilleur que le 30m pour le ML ? (moins de bruit, mais moins de transitions à apprendre)
- Le volume (jamais utilisé comme feature ML) pourrait-il aider à détecter les transitions ?
- Faut-il combiner 30m et 1h comme features multi-résolution ?

---

## Fichiers générés

```
data/prepared/
├── BTCUSD_multitf_macd.csv          (205 MB, BTC seul, MACD only)
├── BTCUSD_multitf_macd_rsi_cci.csv  (en cours de génération)
├── ETHUSD_multitf_macd_rsi_cci.csv  (en cours)
├── BNBUSD_multitf_macd_rsi_cci.csv  (en cours)
├── ADAUSD_multitf_macd_rsi_cci.csv  (en cours)
└── LTCUSD_multitf_macd_rsi_cci.csv  (en cours)
```

### Structure CSV (25 colonnes pour 1 indicateur, ~41 pour 3)

```
5min brut :     open, high, low, close, volume
Par timeframe (30m, 1h) :
  OHLCV live :  open_{tf}_live, high_{tf}_live, low_{tf}_live, close_{tf}_live, volume_{tf}_live
  Step :        step_{tf}
  Par indicateur :
    Live :      {ind}_{tf}_live
    Kalman :    {ind}_{tf}_filtered
    Label causal : {ind}_{tf}_label
    Oracle :    oracle_label_{ind}_{tf}
```
