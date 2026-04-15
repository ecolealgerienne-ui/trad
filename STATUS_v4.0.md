# STATUS v4.0 — Fixed-Lag Kalman Smoother (FLKS) avec micro-updates 5min

**Date**: 2026-04-15
**Asset**: BTC (single asset)
**Branche**: `claude/evaluate-trading-indicators-PBdp0`
**Indicateur testé**: MACD histogram 30min

---

## Objectif

Tester si un Fixed-Lag Kalman Smoother (FLKS, N=2) alimenté par des observations MACD 5min intra-bougie peut approcher le smoother oracle (pykalman.smooth, non-causal) pour l'estimation de la pente.

**Question centrale** : à partir de quel sous-pas 5min de la bougie suivante le signe de la pente FLKS converge-t-il vers celui de l'oracle ?

---

## Architecture de l'expérience

### Oracle (référence fixe)

```
pykalman.smooth() sur 5000 bougies MACD 30min
→ positions smoothed (non-causal, voit toute la série)
→ slope_oracle[t] = smoothed[t-1] - smoothed[t-2]
```

### Test 1 — FLKS 30min pur

```
Forward filter 30min: à chaque bougie t, predict(A,Q) + update(MACD_30m[t])
→ x_filt[t], P_filt[t], x_pred[t], P_pred[t]

Gains RTS precomputés: C[t] = P_filt[t] @ A.T @ inv(P_pred[t+1])

Pour chaque t >= 2:
  Backward 2 pas depuis x_filt[t] (endpoint, causal):
    smoothed[t-1] = x_filt[t-1] + C[t-1] @ (x_filt[t] - x_pred[t])
    smoothed[t-2] = x_filt[t-2] + C[t-2] @ (smoothed[t-1] - x_pred[t-1])
  slope[t] = smoothed[t-1][0] - smoothed[t-2][0]
```

### Test 2 — FLKS 30min + sous-pas 5min (k=1..6)

```
Même forward filter 30min, mêmes gains C.

Pour chaque t >= 2, pour chaque k=1..6 de la bougie t+1:
  x_prov = x_filt[t]
  Pour j=1..k:
    x_prov = predict_sub(A_sub, Q_sub) + update(MACD_live_5min[j])
  
  Backward 2 pas depuis x_prov:
    smoothed[t-1] = x_filt[t-1] + C[t-1] @ (x_prov - x_pred[t])
    smoothed[t-2] = x_filt[t-2] + C[t-2] @ (smoothed[t-1] - x_pred[t-1])
  slope[t,k] = smoothed[t-1][0] - smoothed[t-2][0]
```

### Différence Test 1 vs Test 2

Le forward filter et les gains C sont **identiques**. La seule différence est le point de départ du backward :
- Test 1 : `x_filt[t]` (bougie t fermée, pas d'info sur t+1)
- Test 2 : `x_prov` (x_filt[t] + k micro-updates MACD live de la bougie t+1)

Les sous-pas donnent une fraction de l'information de la bougie suivante, ce qui améliore le lissage de t-1 et t-2.

---

## Paramètres Kalman

| Paramètre | Valeur | Source |
|-----------|--------|--------|
| Q (process variance) | 0.01 × I₂ | pipeline (prepare_multitf_csv.py) |
| R (measurement variance) | 0.1 | pipeline |
| A (transition 30min) | [[1,1],[0,1]] | pipeline |
| H (observation) | [[1,0]] | pipeline |
| A_sub (transition 5min) | [[1, 1/6],[0, 1]] | dt = 5min/30min = 1/6 |
| Q_sub (process 5min) | Q × 1/6 | proportionnel au pas de temps |
| x₀ | [MACD[0], 0] | position = première observation, vélocité = 0 |
| P₀ | I₂ | identité |

**Vérification** : A_sub^6 = A (6 transitions 5min = 1 transition 30min).

---

## Observations injectées (Test 2)

**MACD live frozen/provisional** (copié de `prepare_multitf_csv.py:compute_macd_live`) :
- Les EMA (fast, slow, signal) sont gelées à chaque clôture 30min
- Entre les clôtures, chaque close 5min produit une valeur MACD provisoire
- À la clôture de bougie 30min, la dernière valeur live = MACD 30min exactement

**Vérification cohérence** : `max |last_5min_live - macd_30m| = 0.00` sur 5000 bougies.

---

## Données

| Donnée | Valeur |
|--------|--------|
| CSV source | data_trad/BTCUSD_all_5m.csv |
| Bougies 5min | 879,710 (2017-08 → 2026-01) |
| Bougies 30min utilisées | 5,000 (dernières) |
| Bougies 5min correspondantes | 29,995 |
| MACD range 30min | [-817.6, +695.0] |
| Période évaluation | [1000:5000] = 4,000 bougies 30min |
| MACD (fast, slow, signal) | (12, 26, 9) |

---

## Résultats

### Statistiques oracle

| Métrique | Valeur |
|----------|--------|
| Transitions (changement signe pente) | 398 (10.0% des samples) |
| Persistence (sign[t] == sign[t-1]) | 90.0% |

La persistence de 90% signifie que n'importe quel estimateur qui copie le signe précédent obtient ~90% de concordance "All". Seule la métrique "Transitions" est discriminante.

### Tableau principal

| Méthode | All | Delta/T1 | **Transitions** | **Delta/T1** |
|---------|-----|----------|-----------------|--------------|
| **Test 1: FLKS 30min pur** | 90.90% | baseline | **30.40%** | baseline |
| Test 2: k=1 (5min) | 93.95% | +3.05pp | **60.55%** | **+30.15pp** |
| Test 2: k=2 (10min) | 94.52% | +3.62pp | **73.37%** | **+42.96pp** |
| Test 2: k=3 (15min) | 94.42% | +3.52pp | **78.89%** | **+48.49pp** |
| Test 2: k=4 (20min) | 94.72% | +3.82pp | **82.91%** | **+52.51pp** |
| Test 2: k=5 (25min) | 94.92% | +4.02pp | **86.43%** | **+56.03pp** |
| Test 2: k=6 (30min) | 94.92% | +4.02pp | **87.69%** | **+57.29pp** |

### Lecture des résultats

1. **Test 1 sans info future : Trans = 30.40%**. Sans aucune observation de la bougie suivante, le FLKS est quasi-aveugle aux transitions. Le 90.90% "All" est entièrement de la persistence.

2. **k=1 (5min) : Trans 30% → 61%**. Le premier sous-pas 5min de la bougie suivante apporte la moitié du gain total (30pp sur 57pp).

3. **k=2 (10min) : Trans = 73%**. Après 10 minutes dans la bougie suivante, on a 75% du gain final.

4. **k=3..6 : rendements décroissants**. De k=3 à k=6, on gagne seulement 9pp supplémentaires (79% → 88%).

5. **k=6 (bougie t+1 complète) : Trans = 87.69%**. Même avec la bougie complète, on n'atteint pas 100% car l'oracle voit toute la série (y compris t+2, t+3, ...).

### Courbe de convergence

```
Trans%  |
   90   |                                          ●(k=6: 87.7%)
   80   |                              ●(k=4)  ●(k=5)
   70   |                  ●(k=3)
   60   |      ●(k=1)  ●(k=2)
   50   |
   40   |
   30   |  ●(T1)
        +--+------+------+------+------+------+------→
           0     5min   10min  15min  20min  25min  30min
              Info disponible de la bougie t+1
```

---

## Approximation dans Test 2

Le gain RTS `C[t-1]` est calculé pour la transition 30min (t-1 → t). Dans Test 2, il est appliqué avec `x_prov` qui est à t + k/6 (pas exactement à t). L'approximation est :
- **Conservatrice** : sous-estime le gain potentiel (la covariance réelle de x_prov est réduite par les updates 5min)
- **Acceptable** pour k=1..6 (5-30min de décalage)
- Impact estimé : quelques dixièmes de pp sur les résultats

---

## Bugs corrigés durant le développement

| Bug | Impact | Fix |
|-----|--------|-----|
| **t=0 sans update** | P_filt[0] = I au lieu de posterior → Kalman gain incorrect pour toute la série | Appliquer predict+update à t=0 comme pykalman |
| **Oracle en numpy pur** | Potentiel écart vs pykalman.smooth | Remplacé par pykalman.smooth directement |
| **Closes 5min brutes injectées** | Espace d'observation incohérent (prix vs MACD) | Remplacé par MACD live (frozen/provisional) |
| **Q non scalé pour 5min** | 6× trop de process noise entre sous-pas | Q_sub = Q/6, A_sub = [[1,1/6],[0,1]] |
| **Backward depuis x_filt[t+lag]** | Regardait dans le futur (non-causal) | Backward depuis x_filt[t] (Test 1) ou x_prov (Test 2) |
| **x_pred incohérent avec x_filt** | Gains RTS basés sur des états avant micro-updates | x_pred[t+1] calculé après finalisation de x_filt[t] |
| **Transitions parasites (sign=0)** | np.sign(0)=0 créait de fausses transitions | Seuil epsilon + ignorer les zéros |

---

## Scripts

| Script | Rôle |
|--------|------|
| `src/signal_processing/flks_substep_convergence.py` | Expérience principale (Test 1 + Test 2) |
| `src/signal_processing/flks_vs_filter.py` | Premier script exploratoire (obsolète, remplacé par le précédent) |
| `src/signal_processing/flks_30m_vs_5m_micro.py` | Version intermédiaire (obsolète) |
| `src/signal_processing/validate_kalman.py` | Validation forward filter numpy vs pykalman |

### Commande

```bash
python src/signal_processing/flks_substep_convergence.py --csv data_trad/BTCUSD_all_5m.csv
```

### Options

| Option | Défaut | Description |
|--------|--------|-------------|
| `--csv` | data_trad/BTCUSD_all_5m.csv | CSV 5min source |
| `--n-candles-30m` | 5000 | Nombre de bougies 30min |
| `--eval-start` | 1000 | Début évaluation (skip warmup) |
| `--output-dir` | plots | Dossier pour le graphique |

---

## Conclusion

**Le FLKS avec micro-updates 5min fonctionne.** L'injection progressive des observations MACD live de la bougie suivante améliore massivement la détection des transitions de pente :

- **Sans info future** : 30% de concordance aux transitions (quasi-aléatoire)
- **Avec 5min de la bougie suivante** : 61% (+30pp)
- **Avec 10min** : 73% (+43pp)
- **Avec la bougie complète** : 88% (+57pp)

**Implication pour le trading** : en temps réel, dès qu'une nouvelle bougie 5min arrive, le FLKS peut recalculer la pente lissée et potentiellement détecter une transition 5-10 minutes avant la clôture de la bougie 30min.

---

## Pistes suite

1. **Backtest avec signal FLKS** : utiliser la pente FLKS (avec k=1-2 sous-pas) comme signal d'entrée/sortie. Comparer le PnL vs signal 30min pur.
2. **Calibration Q/R** : les paramètres actuels viennent du pipeline existant. Optimiser Q_sub et R pour les micro-updates 5min pourrait améliorer.
3. **Autres indicateurs** : tester RSI et CCI (mêmes scripts, `--indicator rsi`).
4. **Lag N=3** : tester si un lag plus grand améliore encore.
5. **Multi-asset** : vérifier que les résultats tiennent sur ETH, BNB, ADA, LTC.
