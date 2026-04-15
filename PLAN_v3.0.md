# PLAN DE REPRISE v3.0 — Post-normalisation MACD

**Date**: 2026-04-15
**Raison**: MACD non normalisé par le prix (ratio train/test std = 2.34×)
**Impact**: TOUS les résultats crossfeat sont invalidés (MACD utilisé comme feature dans les 6 modèles)

---

## Inventaire des résultats INVALIDÉS

| Catégorie | Modèles | Raison |
|-----------|---------|--------|
| Crossfeat binaire LSTM (6) | macd/cci/rsi × 30m/1h | MACD feature non normalisée |
| Crossfeat binaire GRU (6) | macd/cci/rsi × 30m/1h | Idem |
| Crossfeat binaire TCN (6) | macd/cci/rsi × 30m/1h | Idem |
| Crossfeat régression (6) | macd/cci/rsi × 30m/1h | Idem |
| Cross-arch vote | MACD 30m seul | Idem |
| Toutes les analyses KPI | 6 configs | Basées sur prédictions biaisées |
| R_strong_agree | 6 configs | Combinaison de modèles biaisés |
| Window 12/50 | MACD 30m/1h | Idem |
| **Total** | **~30 expériences** | |

## Résultats VALIDES (à conserver)

| Catégorie | Détail |
|-----------|--------|
| Single baseline 2-feat RSI | Non crossfeat, pas de MACD en feature |
| Single baseline 2-feat CCI | Non crossfeat, pas de MACD en feature |
| Oracle backtest (30m, 1h pur) | Labels Oracle indépendants du training |
| Pipeline CSV (sauf MACD values) | RSI/CCI live et filtered corrects |
| Architecture + code | Scripts prêts à relancer |
| Conclusions qualitatives | Plafond structurel reste valide (confirmé par RSI/CCI non affectés) |

---

## Plan séquentiel détaillé

### ÉTAPE 1 — Regénération CSV BTC (en cours ~50min)

```bash
python src/prepare_multitf_csv.py --assets BTC --indicators macd rsi cci
```

**Vérification obligatoire** :
```bash
python -c "
import pandas as pd
df = pd.read_csv('data/prepared/BTCUSD_multitf_macd_rsi_cci.csv')
n = len(df)
t_end = int(n * 0.70); t_start = int(n * 0.85)
for col in ['macd_30m_live', 'rsi_30m_live', 'cci_30m_live']:
    tr = df[col].iloc[:t_end].std()
    te = df[col].iloc[t_start:].std()
    print(f'{col}: train_std={tr:.2f}, test_std={te:.2f}, ratio={te/tr:.2f}x')
"
```

**Critère** : MACD ratio ≈ 1.0× (au lieu de 2.34×). RSI et CCI doivent rester ≈ 1.0×.

**Vérification validations (toutes PASS à atol=1e-10)** :
- MACD_30m, MACD_1h : PASS
- RSI_30m, RSI_1h : PASS
- CCI_30m, CCI_1h : PASS
- Kalman_MACD_pos, Kalman_MACD_vel : PASS

---

### ÉTAPE 2 — Entraîner 6 modèles binaires crossfeat LSTM (~30min)

**Script** : `train_multitf.py --crossfeat`
**Features** : 6 (macd/rsi/cci × live + filtered, 30m) ou 12 (+ 1h)
**Target** : `oracle_label_{ind}_{tf}`
**Loss** : BCEWithLogitsLoss

```bash
python src/train_multitf.py --indicator macd --timeframe 30m --crossfeat --epochs 100 --batch-size 512 --cnn-filters 128 --lstm-hidden 128 --dense-hidden 64 --assets BTC;
python src/train_multitf.py --indicator cci --timeframe 30m --crossfeat --epochs 100 --batch-size 512 --cnn-filters 128 --lstm-hidden 128 --dense-hidden 64 --assets BTC;
python src/train_multitf.py --indicator rsi --timeframe 30m --crossfeat --epochs 100 --batch-size 512 --cnn-filters 128 --lstm-hidden 128 --dense-hidden 64 --assets BTC;
python src/train_multitf.py --indicator macd --timeframe 1h --crossfeat --epochs 100 --batch-size 512 --cnn-filters 128 --lstm-hidden 128 --dense-hidden 64 --assets BTC;
python src/train_multitf.py --indicator cci --timeframe 1h --crossfeat --epochs 100 --batch-size 512 --cnn-filters 128 --lstm-hidden 128 --dense-hidden 64 --assets BTC;
python src/train_multitf.py --indicator rsi --timeframe 1h --crossfeat --epochs 100 --batch-size 512 --cnn-filters 128 --lstm-hidden 128 --dense-hidden 64 --assets BTC 2>&1 | tee models/step2_lstm.log
```

**Fichiers générés** :
- `models/best_model_{ind}_{tf}_crossfeat.pth` × 6
- `data/prepared/{ind}_{tf}_crossfeat_dataset.npz` × 6
- `models/training_history_{ind}_{tf}_crossfeat.json` × 6

---

### ÉTAPE 3 — KPIs de base sur les 6 LSTM (~5min)

**Script** : `compare_all_models.py --crossfeat`

```bash
python src/compare_all_models.py --crossfeat 2>&1 | tee models/step3_kpi_lstm.log
```

**Ce qu'on vérifie** :
- MACD ratio de switchs a changé vs avant normalisation ?
- RSI/CCI ratio stable (confirmation non-impact) ?
- Hiérarchie MACD > CCI > RSI préservée ?

**Point de décision** : si les résultats sont très différents, s'arrêter et analyser avant de continuer.

---

### ÉTAPE 4 — Entraîner 12 modèles GRU + TCN (~60min)

**Scripts** : `train_multitf.py --crossfeat --arch cnn-gru` et `--arch tcn`

```bash
# GRU × 6
python src/train_multitf.py --indicator macd --timeframe 30m --crossfeat --arch cnn-gru --epochs 100 --batch-size 512 --cnn-filters 128 --lstm-hidden 128 --dense-hidden 64 --assets BTC;
python src/train_multitf.py --indicator cci --timeframe 30m --crossfeat --arch cnn-gru --epochs 100 --batch-size 512 --cnn-filters 128 --lstm-hidden 128 --dense-hidden 64 --assets BTC;
python src/train_multitf.py --indicator rsi --timeframe 30m --crossfeat --arch cnn-gru --epochs 100 --batch-size 512 --cnn-filters 128 --lstm-hidden 128 --dense-hidden 64 --assets BTC;
python src/train_multitf.py --indicator macd --timeframe 1h --crossfeat --arch cnn-gru --epochs 100 --batch-size 512 --cnn-filters 128 --lstm-hidden 128 --dense-hidden 64 --assets BTC;
python src/train_multitf.py --indicator cci --timeframe 1h --crossfeat --arch cnn-gru --epochs 100 --batch-size 512 --cnn-filters 128 --lstm-hidden 128 --dense-hidden 64 --assets BTC;
python src/train_multitf.py --indicator rsi --timeframe 1h --crossfeat --arch cnn-gru --epochs 100 --batch-size 512 --cnn-filters 128 --lstm-hidden 128 --dense-hidden 64 --assets BTC;
# TCN × 6
python src/train_multitf.py --indicator macd --timeframe 30m --crossfeat --arch tcn --epochs 100 --batch-size 512 --cnn-filters 128 --lstm-hidden 128 --dense-hidden 64 --assets BTC;
python src/train_multitf.py --indicator cci --timeframe 30m --crossfeat --arch tcn --epochs 100 --batch-size 512 --cnn-filters 128 --lstm-hidden 128 --dense-hidden 64 --assets BTC;
python src/train_multitf.py --indicator rsi --timeframe 30m --crossfeat --arch tcn --epochs 100 --batch-size 512 --cnn-filters 128 --lstm-hidden 128 --dense-hidden 64 --assets BTC;
python src/train_multitf.py --indicator macd --timeframe 1h --crossfeat --arch tcn --epochs 100 --batch-size 512 --cnn-filters 128 --lstm-hidden 128 --dense-hidden 64 --assets BTC;
python src/train_multitf.py --indicator cci --timeframe 1h --crossfeat --arch tcn --epochs 100 --batch-size 512 --cnn-filters 128 --lstm-hidden 128 --dense-hidden 64 --assets BTC;
python src/train_multitf.py --indicator rsi --timeframe 1h --crossfeat --arch tcn --epochs 100 --batch-size 512 --cnn-filters 128 --lstm-hidden 128 --dense-hidden 64 --assets BTC 2>&1 | tee models/step4_gru_tcn.log
```

**Fichiers générés** :
- `models/best_model_{ind}_{tf}_crossfeat_cnngru.pth` × 6
- `models/best_model_{ind}_{tf}_crossfeat_tcn.pth` × 6
- NPZ et JSON correspondants × 12

---

### ÉTAPE 5 — Entraîner 6 modèles régression crossfeat (~30min)

**Script** : `train_multitf.py --crossfeat --target-type continuous`

```bash
python src/train_multitf.py --indicator macd --timeframe 30m --crossfeat --target-type continuous --epochs 100 --batch-size 512 --cnn-filters 128 --lstm-hidden 128 --dense-hidden 64 --assets BTC;
python src/train_multitf.py --indicator cci --timeframe 30m --crossfeat --target-type continuous --epochs 100 --batch-size 512 --cnn-filters 128 --lstm-hidden 128 --dense-hidden 64 --assets BTC;
python src/train_multitf.py --indicator rsi --timeframe 30m --crossfeat --target-type continuous --epochs 100 --batch-size 512 --cnn-filters 128 --lstm-hidden 128 --dense-hidden 64 --assets BTC;
python src/train_multitf.py --indicator macd --timeframe 1h --crossfeat --target-type continuous --epochs 100 --batch-size 512 --cnn-filters 128 --lstm-hidden 128 --dense-hidden 64 --assets BTC;
python src/train_multitf.py --indicator cci --timeframe 1h --crossfeat --target-type continuous --epochs 100 --batch-size 512 --cnn-filters 128 --lstm-hidden 128 --dense-hidden 64 --assets BTC;
python src/train_multitf.py --indicator rsi --timeframe 1h --crossfeat --target-type continuous --epochs 100 --batch-size 512 --cnn-filters 128 --lstm-hidden 128 --dense-hidden 64 --assets BTC 2>&1 | tee models/step5_regression.log
```

---

### ÉTAPE 6 — Analyses KPI complètes (~15min)

#### 6a. KPIs par architecture (LSTM vs GRU vs TCN)

```bash
# LSTM
python src/compare_all_models.py --crossfeat 2>&1 | tee models/step6a_kpi_lstm.log

# Résumé training histories
for f in models/training_history_*_crossfeat_*.json; do
    echo "=== $(basename $f) ==="
    python -c "import json; d=json.load(open('$f')); print(f'  Best epoch: {d[\"best_epoch\"]}, Val loss: {d[\"best_val_loss\"]:.4f}')"
done 2>&1 | tee models/step6a_all_histories.log
```

#### 6b. Cross-architecture vote (6 configs)

Nécessite modification de `analyze_cross_arch_switches.py` pour supporter tous les indicateurs (actuellement hardcodé MACD 30m).

```bash
python src/analyze_cross_arch_switches.py 2>&1 | tee models/step6b_cross_arch.log
```

#### 6c. Évaluation régression (6 modèles)

```bash
python tests/eval_regression.py --indicator macd --timeframe 30m --crossfeat;
python tests/eval_regression.py --indicator cci --timeframe 30m --crossfeat;
python tests/eval_regression.py --indicator rsi --timeframe 30m --crossfeat;
python tests/eval_regression.py --indicator macd --timeframe 1h --crossfeat;
python tests/eval_regression.py --indicator cci --timeframe 1h --crossfeat;
python tests/eval_regression.py --indicator rsi --timeframe 1h --crossfeat 2>&1 | tee models/step6c_regression_eval.log
```

#### 6d. Deep analysis régression (R² conditionnel)

```bash
python src/analyze_regression_deep.py --indicator macd --timeframe 30m;
python src/analyze_regression_deep.py --indicator cci --timeframe 30m;
python src/analyze_regression_deep.py --indicator rsi --timeframe 30m;
python src/analyze_regression_deep.py --indicator macd --timeframe 1h;
python src/analyze_regression_deep.py --indicator cci --timeframe 1h;
python src/analyze_regression_deep.py --indicator rsi --timeframe 1h 2>&1 | tee models/step6d_deep_regression.log
```

---

### ÉTAPE 7 — Comparaison binaire vs régression (~5min)

```bash
python src/compare_binary_vs_regression.py 2>&1 | tee models/step7_bin_vs_reg.log
```

---

### ÉTAPE 8 — Analyses avancées (~10min)

#### 8a. Magnitude filter (régression MACD 30m)

```bash
python src/analyze_magnitude_filter.py --indicator macd --timeframe 30m 2>&1 | tee models/step8a_magnitude.log
```

#### 8b. Magnitude dynamics (régression MACD 30m)

```bash
python src/analyze_magnitude_dynamics.py --indicator macd --timeframe 30m 2>&1 | tee models/step8b_dynamics.log
```

#### 8c. Cross-model discrimination (30m uniquement)

```bash
python src/analyze_switch_discrimination.py 2>&1 | tee models/step8c_cross_model.log
```

#### 8d. Cross-timeframe discrimination

```bash
python src/analyze_cross_tf_discrimination.py 2>&1 | tee models/step8d_cross_tf.log
```

---

### ÉTAPE 9 — Documentation finale et décision backtest (~15min)

1. Comparer AVANT/APRÈS normalisation MACD
2. Mettre à jour STATUS_v2.2.md (ou créer v3.0)
3. Identifier la meilleure configuration pour backtest PnL
4. Lancer le backtest

---

## Résumé du plan

| Étape | Description | Modèles | Temps |
|-------|-------------|---------|-------|
| 1 | Regénérer CSV BTC | 1 CSV | ~50 min |
| 2 | 6 LSTM crossfeat binaire | 6 | ~30 min |
| 3 | KPIs LSTM | analyse | ~5 min |
| 4 | 12 GRU + TCN crossfeat | 12 | ~60 min |
| 5 | 6 régression crossfeat | 6 | ~30 min |
| 6 | Analyses KPI complètes | analyses | ~15 min |
| 7 | Binaire vs régression | analyse | ~5 min |
| 8 | Analyses avancées | analyses | ~10 min |
| 9 | Documentation + décision | doc | ~15 min |
| **Total** | | **24 modèles + analyses** | **~3h30** |

## Scripts utilisés

| Script | Rôle | Étape |
|--------|------|-------|
| `src/prepare_multitf_csv.py` | Génération CSV | 1 |
| `src/train_multitf.py` | Entraînement | 2, 4, 5 |
| `src/compare_all_models.py` | KPIs comparatifs | 3, 6a |
| `src/analyze_cross_arch_switches.py` | Vote cross-archi | 6b |
| `tests/eval_regression.py` | Éval régression | 6c |
| `src/analyze_regression_deep.py` | R² conditionnel | 6d |
| `src/compare_binary_vs_regression.py` | Bin vs Reg | 7 |
| `src/analyze_magnitude_filter.py` | Filtre magnitude | 8a |
| `src/analyze_magnitude_dynamics.py` | Dynamique magnitude | 8b |
| `src/analyze_switch_discrimination.py` | Discrim cross-model | 8c |
| `src/analyze_cross_tf_discrimination.py` | Discrim cross-TF | 8d |
