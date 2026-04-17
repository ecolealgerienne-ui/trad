# Audit unitaire `src/signal_processing/core.py`

Chaque fichier `test_NN_<fonction>.py` audite une fonction de `core.py` sur des signaux synthétiques purs (linéaire, constant, step, sinus). Pas de CSV réel.

## Lancement

Depuis la racine du repo :

```bash
# Un seul fichier, verbose
pytest tests/audit_core/test_01_slopes_test2.py -v -s

# Tous les tests audit
pytest tests/audit_core/ -v -s

# Un seul test, avec tracebacks
pytest tests/audit_core/test_01_slopes_test2.py::TestCausality::test_no_leak_from_live_t_plus_2 -v
```

Le flag `-s` affiche les `print()` diagnostiques (ex. valeurs observées vs théoriques).

## Liste des audits

| # | Fichier | Fonction auditée | Priorité |
|---|---------|------------------|----------|
| 01 | `test_01_slopes_test2.py` | `compute_slopes_test2` (features ML) | **CRITIQUE** |
| 02 | _à venir_ | `backtest_5m` (PnL +870%) | **CRITIQUE** |
| 03 | _à venir_ | `forward_filter_30m` + `_adaptive` | HAUTE |
| 04 | _à venir_ | `compute_oracle` (labels) | HAUTE |
| 05 | _à venir_ | Indicateurs live (macd/rsi/cci) | MOYENNE |
| 06 | _à venir_ | Kalman live std/aqkf | MOYENNE |
| 07 | _à venir_ | `backtest_30m` | MOYENNE |
| 08 | _à venir_ | `load_test_data` (NPZ) | MOYENNE |
| 09 | _à venir_ | Helpers (group_per_candle, metrics) | FAIBLE |
| 10 | _à venir_ | Primitives (kf_update, indicateurs standard) | FAIBLE |
| 11 | _à venir_ | Post-processing (Viterbi, CUSUM) | FAIBLE |

## Convention de test

Chaque fichier teste :

1. **Shape** — output dimensions, gestion des bords (NaN, warm-up).
2. **Ground truth** — signaux à réponse analytique connue (linéaire → slope = a).
3. **Causalité** — aucune sortie `output[t]` ne doit changer si on pollue les inputs futurs.
4. **Symétrie** — invariances (ex: inversion de signe, translation).
5. **Edge cases** — NaN, zéros, bords, warm-up.
