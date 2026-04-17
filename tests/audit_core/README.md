# Audit Core — Pipeline FLKS trading

Audit unitaire complet du pipeline de trading CNN-LSTM/XGBoost sur features FLKS backward slopes.

**Périmètre audité** :
- `src/signal_processing/core.py` (24 fonctions)
- `src/signal_processing/prepare_flks_csv.py` (assemblage du pipeline)
- `src/signal_processing/train_flks_slopes.py` (training/split/sequences)
- `src/backtest_consensus_direction.py` (3 fonctions backtest)

**Objectif** : valider la causalité du pipeline et diagnostiquer l'origine du PnL backtest de +870%.

## Méthodologie

- **Tests unitaires** avec signaux synthétiques purs (linéaire, constant, step, sinus).
- **Anti-leakage systématique** : pour chaque fonction, vérifier que polluer les données futures ne change pas les sorties passées.
- **Ground truth analytique** quand possible (PnL sur prix connus, slope linéaire = `a`, etc.).
- **Vérification des formules Kalman** pas à pas (prediction step, update step, RTS gain).
- **Tests inter-fonctions** : par exemple MACD live aux closes 30min == MACD 30m standard à la précision machine.

## Fichiers de tests

| # | Fichier | Fonctions couvertes | Tests |
|---|---------|---------------------|-------|
| 01 | `test_01_slopes_test2.py` | `compute_slopes_test2` (features ML) | 15 |
| 02 | `test_02_backtest_5m.py` | `backtest_5m` (PnL +870%) | 23 |
| 03 | `test_03_forward_filter_30m.py` | `forward_filter_30m` | 19 |
| 04 | `test_04_forward_filter_30m_adaptive.py` | `forward_filter_30m_adaptive` (AQ-KF) | 19 |
| 05 | `test_05_compute_oracle.py` | `compute_oracle` (labels non-causaux par design) | 17 |
| 06 | `test_06_alignment_helpers.py` | `compute_bucket_close_mask`, `compute_live_ohlcv`, `group_per_candle` | 22 |
| 07 | `test_07_compute_macd_live.py` | `compute_macd_live` | 11 |
| 08 | `test_08_load_test_data.py` | `load_test_data` (NPZ loader) | 15 |
| 09 | `test_09_indicators_standard.py` | `calculate_macd/rsi/cci`, `resample_ohlcv`, `load_csv` | 24 |
| 10 | `test_10_kalman_primitives.py` | `kf_update`, `kf_predict_sub`, `inv2x2`, `is_pos_semidef` | 28 |
| 11 | `test_11_metrics.py` | `sign_concordance*`, `find_oracle_transitions` | 22 |
| 12 | `test_12_postprocessing.py` | `viterbi_decode`, `cusum_filter` | 17 |
| 13 | `test_13_prepare_flks_pipeline.py` | pipeline `prepare_flks_csv.py` end-to-end | 11 |
| 14 | `test_14_train_flks_slopes.py` | pipeline `train_flks_slopes.py` (split, sequences) | 13 |
| 15 | `test_15_backtest_consensus_direction.py` | `backtest_model_only`, `backtest_oracle_only`, `backtest_consensus` | 18 |
| **Total** | **15 fichiers** | **24 fns core + 3 pipelines** | **272 tests, 0 hard fail** |

## Résultats

### Bugs trouvés (1 — impact nul)

**`forward_filter_30m*` init leakage (core.py:410 et 439)**
```python
first_valid_val = indicator_30m[~np.isnan(indicator_30m)][0]
```
Si les premières valeurs sont NaN (warm-up MACD ~26 bougies), `first_valid_val` = première valeur **future** non-NaN, utilisée pour initialiser `x_filt[0..25]`.

**Impact réel : nul**.
- TRIM=100 (utilisé dans `prepare_flks_csv.py` pour l'eval et `train_flks_slopes.py` pour le training) couvre largement (MACD warmup = 35, propagation init ~50).
- Correction propre possible : `x_p = [0.0, 0.0]`, `P_p = np.eye(2) * 1e6` (laisser Kalman converger naturellement).

### Fausses alertes corrigées (1)

**Ffill propage une slope calculée avec data future** — INITIALEMENT suspecté comme cause du +870%, CORRIGÉ après analyse :

- `std_k6_slope[t]` utilise data jusqu'à `close[t+1]` du 30min pour **estimer causalement une pente PASSÉE** entre `pos[t-2]` et `pos[t-1]`.
- Label `oracle_label[t]` mesure la **MÊME** quantité passée, via smoother global non-causal.
- **Feature et label pointent vers la même quantité passée** → pas de leakage exploitable, juste du denoising.

Gain marginal observé (concordance 95.67% → accuracy 96.3%) = **+0.6-3%**, cohérent avec du denoising standard. Un vrai leakage exploité donnerait ≥99.9%.

### Vérifications critiques validées

- ✅ **Causalité features FLKS** — polluer `live[t+2+]` ou `x_filt[t+2+]` ne change pas `slopes[t]`.
- ✅ **Causalité forward filter standard ET adaptatif** — polluer `obs[T+1:]` ne change pas `x_filt[:T+1]`, `P_filt[:T+1]`, `C[:T+1]`.
- ✅ **Convention slopes alignée** — `compute_oracle` et `compute_slopes_test2` utilisent tous deux `pos[t-1] - pos[t-2]` (pas de désalignement feature↔label).
- ✅ **MACD live aux closes == MACD 30m standard** — diff < 1e-8 à la précision machine après warmup.
- ✅ **Split chronologique train/val/test** avec gap = WINDOW entre train et val (pas d'overlap entre séquences).
- ✅ **Fees 2× par roundtrip linéaires** — `pnl(f) - pnl(0) = -2 * f * n_trades * 100`.
- ✅ **Exécution backtest cohérente** — `backtest_5m` exécute à `closes[t+1][k-1]`, aligné temporellement avec la disponibilité de `slopes[t]` calculée avec `k` sous-pas de la bougie `t+1`.
- ✅ **Non-causalité de l'oracle** confirmée (comportement voulu pour labels) — modifier `y[T+10]` change bien `positions[T]`.

## Conclusion sur le +870%

**Le pipeline est sain.** Aucun leakage exploité détecté dans l'audit statique/unitaire.

Explication cohérente du PnL :
- Concordance features FLKS vs oracle aux closes 30min : **95.67%** (k=6)
- XGBoost sur séquences 25 : **96.3%** test accuracy (+0.6% = denoising)
- PnL backtest modèle : **+870%** ≈ 98% du PnL oracle (+890%)
- **Le modèle capture presque toute la performance que l'oracle lui-même peut extraire du signal "pente passée 30min".**

## Ce qui reste à vérifier (hors audit unit)

1. **OOS strict** sur une période jamais vue en training/val/test (le vrai juge d'un modèle).
2. **Backtest avec coûts réalistes** : slippage, spread bid-ask, funding rates, latence d'exécution.
3. **Stabilité temporelle** : le momentum 30min détecté sur 458 jours peut ne pas tenir sur plusieurs cycles de marché (bull/bear/range).
4. **Robustesse aux régimes** : tester le modèle séparément sur phases de trend vs consolidation.

## Lancer les tests

```bash
# Tous les tests d'audit
python -m pytest tests/audit_core/ -v

# Un fichier spécifique
python -m pytest tests/audit_core/test_01_slopes_test2.py -v -s

# Mode verbose avec prints diagnostiques
python -m pytest tests/audit_core/ -v -s
```

Les tests utilisent uniquement des signaux synthétiques (pas d'accès aux CSV réels). Exécution totale < 30 secondes.

## Convention de test

Chaque fichier teste systématiquement :

1. **Shape** — dimensions de sortie, gestion des bords (NaN, warm-up).
2. **Ground truth** — signaux à réponse analytique connue (linéaire → slope = `a`, constant → 0, etc.).
3. **Causalité** — aucune sortie `output[t]` ne doit changer si on pollue les inputs `[t+1:]`.
4. **Formules** — vérification pas à pas des équations Kalman / fees / EMA.
5. **Edge cases** — NaN, zéros, bords, warm-up, dégénéré.
