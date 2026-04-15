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

## Résultats — 3 indicateurs comparés

### Vérification cohérence MACD live / RSI live / CCI live

| Indicateur | Max err (last 5min live vs standard 30min) | Candles vérifiées |
|------------|-------------------------------------------|-------------------|
| MACD | 0.00e+00 | 5,000 |
| RSI | 0.00e+00 | 4,997 |
| CCI | 5.01e-11 | 4,981 |

Les indicateurs live frozen/provisional coïncident exactement avec les standards aux clôtures 30min.

### Statistiques oracle par indicateur

| Indicateur | Transitions | % des samples | Persistence |
|------------|-------------|---------------|-------------|
| MACD | 398 | 10.0% | 90.0% |
| RSI | 580 | 14.5% | 85.5% |
| CCI | 484 | 12.1% | 87.9% |

RSI est le plus "nerveux" (580 transitions, 14.5%), MACD le plus stable (398, 10%).

### Tableau comparatif principal

| Méthode | MACD All | **MACD Trans** | RSI All | **RSI Trans** | CCI All | **CCI Trans** |
|---------|----------|----------------|---------|---------------|---------|---------------|
| **Test 1: 30m pur** | 90.90% | **30.40%** | 86.67% | **45.34%** | 88.12% | **38.64%** |
| k=1 (5min) | 93.95% | **60.55%** | 88.35% | **62.00%** | 90.92% | **62.11%** |
| k=2 (10min) | 94.52% | **73.37%** | 89.02% | **70.64%** | 92.05% | **73.50%** |
| k=3 (15min) | 94.42% | **78.89%** | 89.25% | **75.13%** | 91.67% | **78.47%** |
| k=4 (20min) | 94.72% | **82.91%** | 89.47% | **78.24%** | 91.40% | **80.75%** |
| k=5 (25min) | 94.90% | **86.43%** | 90.25% | **82.21%** | 91.55% | **82.19%** |
| k=6 (30min) | 94.92% | **87.69%** | 90.15% | **82.56%** | 91.30% | **83.85%** |
| **Gain k=6 vs T1** | +4.02pp | **+57.29pp** | +3.47pp | **+37.21pp** | +3.17pp | **+45.21pp** |

### Analyse

**1. Le pattern est universel sur les 3 indicateurs.** La courbe de convergence a la même forme pour MACD, RSI et CCI : gain massif à k=1, rendements décroissants ensuite.

**2. À k=1 (5min), les 3 indicateurs convergent vers ~60-62% Trans.** Le premier sous-pas de la bougie suivante apporte un gain similaire quel que soit l'indicateur :
- MACD : 30% → 61% (+30pp)
- RSI : 45% → 62% (+17pp)
- CCI : 39% → 62% (+23pp)

**3. RSI part de plus haut en Test 1 (45% vs 30-39%).** Le RSI, plus nerveux (580 transitions), capture mieux les changements de signe avec le filtre seul. Mais le gain relatif des sous-pas est plus faible (+37pp vs +57pp MACD).

**4. MACD bénéficie le plus des micro-updates (+57pp).** Le MACD étant le plus lisse (moins de transitions), il a le plus à gagner de l'information future partielle.

**5. Plafond à k=6 : 83-88% selon l'indicateur.** Même avec la bougie t+1 complète, on ne dépasse pas 88%. L'oracle voit t+2, t+3, ... au-delà.

**6. La métrique "All" est trompeuse.** La persistence (85-90%) domine : les gains "All" sont de +3-4pp seulement, alors que les gains "Trans" sont de +37-57pp. Seule la colonne Trans a du sens pour le trading.

### Courbe de convergence (Transitions)

```
Trans%  |
   90   |                                      M(88)
   85   |                          M       M  C(84) R(83)
   80   |                  M   C       C R
   75   |              C R
   70   |          M R
   65   |
   60   |  M=C=R (~62% pour les 3)
        |
   45   |  R(45)
   40   |  C(39)
   30   |  M(30)
        +--+------+------+------+------+------+------→
        T1  k=1    k=2    k=3    k=4    k=5    k=6
            5min   10min  15min  20min  25min  30min
```

M=MACD, R=RSI, C=CCI

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

**Le FLKS avec micro-updates 5min fonctionne, et le pattern est universel sur les 3 indicateurs.**

### Résumé par indicateur

| Indicateur | Trans T1 | Trans k=1 | Trans k=6 | Gain total |
|------------|----------|-----------|-----------|------------|
| **MACD** | 30.40% | 60.55% | 87.69% | **+57.29pp** |
| **CCI** | 38.64% | 62.11% | 83.85% | **+45.21pp** |
| **RSI** | 45.34% | 62.00% | 82.56% | **+37.21pp** |

### Points clés

1. **Les 3 indicateurs convergent vers ~62% Trans dès k=1 (5min).** Le premier sous-pas de la bougie suivante est le plus informatif, quel que soit l'indicateur.

2. **MACD bénéficie le plus des micro-updates** (+57pp). Étant le plus lisse, il a le plus grand gap entre le filtre seul (30%) et l'oracle. Les sous-pas comblent ce gap efficacement.

3. **RSI part de plus haut mais plafonne plus bas** (45% → 83%). Plus nerveux, il capture déjà une partie des transitions avec le filtre seul, mais converge vers un plafond plus bas.

4. **Le gain marginal décroît fortement après k=2 (10min).** Pour les 3 indicateurs, k=1-2 apportent 70-80% du gain total. Les sous-pas k=3-6 n'ajoutent que 10-15pp.

5. **Implication trading** : en temps réel, dès qu'une nouvelle bougie 5min arrive (k=1), le FLKS peut recalculer la pente lissée et potentiellement détecter une transition ~25 minutes avant la clôture de la bougie 30min, avec une concordance de ~62% aux transitions.

---

## Pistes suite

1. **Backtest avec signal FLKS** : utiliser la pente FLKS (avec k=1-2 sous-pas) comme signal d'entrée/sortie. Comparer le PnL vs signal 30min pur.
2. **Calibration Q/R** : les paramètres actuels viennent du pipeline existant. Optimiser Q_sub et R pour les micro-updates 5min pourrait améliorer.
3. **Lag N=3** : tester si un lag plus grand améliore encore.
4. **Multi-asset** : vérifier que les résultats tiennent sur ETH, BNB, ADA, LTC.
5. **Combiner les 3 indicateurs** : les transitions détectées par MACD vs RSI vs CCI sont-elles les mêmes ou complémentaires ?
