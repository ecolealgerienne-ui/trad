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

### Tableau comparatif principal (après correction backward 3 pas)

| Méthode | MACD All | **MACD Trans** | RSI All | **RSI Trans** | CCI All | **CCI Trans** |
|---------|----------|----------------|---------|---------------|---------|---------------|
| **Test 1: 30m pur** | 90.90% | **30.40%** | 86.67% | **45.34%** | 88.12% | **38.64%** |
| k=1 (5min) | 93.87% | **58.54%** | 88.32% | **61.49%** | 91.00% | **61.70%** |
| k=2 (10min) | 94.60% | **71.36%** | 89.25% | **69.60%** | 91.97% | **70.81%** |
| k=3 (15min) | 94.75% | **75.88%** | 89.75% | **73.92%** | 92.17% | **75.36%** |
| k=4 (20min) | 95.22% | **79.15%** | 90.10% | **76.17%** | 92.32% | **77.85%** |
| k=5 (25min) | 95.47% | **79.90%** | 91.60% | **79.45%** | 92.72% | **78.88%** |
| k=6 (30min) | 95.82% | **80.90%** | 92.32% | **81.52%** | 93.12% | **80.75%** |
| **Gain k=6 vs T1** | +4.92pp | **+50.50pp** | +5.65pp | **+36.18pp** | +5.00pp | **+42.11pp** |

### Analyse

**1. Le pattern est universel sur les 3 indicateurs.** La courbe de convergence a la même forme pour MACD, RSI et CCI : gain massif à k=1, rendements décroissants ensuite.

**2. À k=1 (5min), les 3 indicateurs convergent vers ~59-62% Trans.** Le premier sous-pas de la bougie suivante apporte un gain similaire quel que soit l'indicateur :
- MACD : 30% → 59% (+28pp)
- RSI : 45% → 61% (+16pp)
- CCI : 39% → 62% (+23pp)

**3. RSI part de plus haut en Test 1 (45% vs 30-39%).** Le RSI, plus nerveux (580 transitions), capture mieux les changements de signe avec le filtre seul. Mais le gain relatif des sous-pas est plus faible (+37pp vs +57pp MACD).

**4. MACD bénéficie le plus des micro-updates (+57pp).** Le MACD étant le plus lisse (moins de transitions), il a le plus à gagner de l'information future partielle.

**5. Plafond à k=6 : ~81% pour les 3 indicateurs.** Même avec la bougie t+1 complète, on n'atteint pas 100%. L'oracle voit t+2, t+3, ... au-delà.

**6. La métrique "All" est trompeuse.** La persistence (85-90%) domine : les gains "All" sont de +3-4pp seulement, alors que les gains "Trans" sont de +37-57pp. Seule la colonne Trans a du sens pour le trading.

### Courbe de convergence (Transitions, après correction 3 pas)

```
Trans%  |
   80   |                              M(81) R(82) C(81)
   75   |              M   C       M R C
   70   |          M C   R
   65   |
   60   |  M=C=R (~59-62% pour les 3)
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

## Correction du backward Test 2 — Pas 1 manquant

### Bug identifié

Le backward Test 2 initial (2 pas) injectait `x_prov` directement dans le lissage de t-1, en sautant le lissage de t. Cela sur-pondérait l'info des sous-pas.

### Fix : backward 3 pas

```python
# Pas 1 : lisser t avec x_prov (transition partielle A_sub^k)
A_k = A_SUB^k           # transition de k sous-pas
Q_k = Q_SUB * k         # process noise cumulé
x_pred_partial = A_k @ x_filt[t]
P_pred_partial = A_k @ P_filt[t] @ A_k.T + Q_k
C_partial = P_filt[t] @ A_k.T @ inv(P_pred_partial)
sm_t = x_filt[t] + C_partial @ (x_prov - x_pred_partial)

# Pas 2 : lisser t-1 avec smoothed[t]
sm_t1 = x_filt[t-1] + C[t-1] @ (sm_t - x_pred[t])

# Pas 3 : lisser t-2 avec smoothed[t-1]
sm_t2 = x_filt[t-2] + C[t-2] @ (sm_t1 - x_pred[t-1])
```

### Impact de la correction

Les chiffres avant/après correction (MACD Trans) :

| k | Avant (2 pas) | Après (3 pas) | Écart |
|---|---------------|---------------|-------|
| k=1 | 60.55% | **58.54%** | -2.0pp |
| k=2 | 73.37% | **71.36%** | -2.0pp |
| k=6 | 87.69% | **80.90%** | -6.8pp |

L'effet augmente avec k : plus de sous-pas = plus d'atténuation par C_partial. Le plafond passe de ~88% à ~81%.

---

## Comparaison FLKS vs LSTM (STATUS_v3.0)

Le LSTM CNN (STATUS_v3.0) et le FLKS répondent à la même question : détecter les changements de signe de la pente oracle.

| Méthode | MACD Trans | RSI Trans | CCI Trans | Causal | Modèle |
|---------|------------|-----------|-----------|--------|--------|
| **LSTM crossfeat 30m** | **49.2%** | **39.3%** | **44.2%** | Oui | CNN-LSTM entraîné |
| **FLKS T1 (30m pur)** | **30.4%** | **45.3%** | **38.6%** | Oui | Kalman forward seul |
| FLKS T2 k=1 (5min) | 58.5% | 61.5% | 61.7% | +5min délai | Kalman + 1 sous-pas |
| FLKS T2 k=6 (30min) | 80.9% | 81.5% | 80.8% | +30min délai | Kalman + 6 sous-pas |

**Observation clé** : le LSTM (49% MACD) se situe entre FLKS T1 (30%) et FLKS T2 k=1 (59%). Le réseau de neurones entraîné sur 6 features capture l'équivalent de "quelques minutes d'info future" grâce aux patterns non-linéaires, mais un simple filtre linéaire avec 5 minutes de données réelles le bat.

---

## Backtest PnL

### Méthodologie

- **Signal** : signe de la pente FLKS → LONG (+1) ou SHORT (-1), toujours en position
- **Exécution** : Oracle/T1 au close[t] ≈ open[t+1] ; T2 k=N au close du step N de la bougie t+1
- **Frais** : 0.1% par trade (entrée + sortie séparées = 0.2% round trip)
- **Période** : 4,000 bougies 30min = ~83 jours (2025-09 → 2026-01)
- **Buy & Hold** : -19.69% sur la même période

### Résultats sans filtre

| Méthode | MACD PnL | RSI PnL | CCI PnL |
|---------|----------|---------|---------|
| Oracle | +123.4% | +168.3% | +135.3% |
| T1: 30m pur | -52.8% | -134.9% | -111.5% |
| T2: k=1 | -40.9% | -151.6% | -104.4% |
| T2: k=6 | -36.1% | -137.5% | -100.7% |
| Buy & Hold | -19.7% | -19.7% | -19.7% |

**Problème identifié** : trop de trades parasites (467-822 vs 399-580 pour l'oracle). Les micro-reversals quand la pente oscille autour de zéro détruisent le PnL.

### Grid search 2D : Holding minimum × Seuil de magnitude

Deux filtres combinés :
- **Seuil** : ne reverser que si `|pente| > threshold` (filtre les pentes faibles)
- **Holding minimum** : ne pas reverser avant N bougies 30min (filtre les micro-flips)

Seuils calibrés sur les percentiles de |slope| MACD T1 : P50=10.9, P75=22.0, P90=36.8.

### Résultats MACD (best configs positives)

| Config | PnL | Trades | WR | vs B&H |
|--------|-----|--------|-----|--------|
| **T2:k=1 hold=8 thr=P90** | **+54.3%** | **68** | **66.2%** | **+74pp** |
| T2:k=1 hold=0 thr=P90 | +49.2% | 70 | 64.3% | +69pp |
| T2:k=2 hold=10 thr=P75 | +26.4% | 145 | — | +46pp |
| T2:k=2 hold=10 thr=0 | +21.6% | 297 | — | +41pp |
| T1 hold=10 thr=P50 | +31.7% | 203 | — | +51pp |
| T1 hold=8 thr=P90 | +33.5% | 62 | — | +53pp |
| T1 hold=15 thr=P90 | +34.6% | 56 | — | +54pp |

### Résultats RSI et CCI

| Indicateur | Best config | PnL | Trades |
|------------|------------|-----|--------|
| **RSI** | T1 hold=0 thr=P50 | +9.8% | 3 |
| **CCI** | T2:k=1 hold=15 thr=P75 | +12.3% | 163 |

RSI inutilisable avec les seuils MACD (pentes dans un espace d'échelle différent → 0-3 trades). CCI modeste.

### Pattern dans la grille MACD

Le coin bas-droite (hold élevé + seuil élevé) est systématiquement positif. Le coin haut-gauche (pas de filtre) est systématiquement négatif. Les deux filtres sont complémentaires :
- Le seuil élimine les pentes faibles (micro-oscillations autour de 0)
- Le holding empêche les flips rapides sur les pentes fortes

### Avertissement suroptimisation

Les seuils sont calibrés sur les mêmes données que l'évaluation (P50/P75/P90 des slopes T1 sur [1000:5000]). Les résultats sont biaisés à la hausse. Validation sur une autre période nécessaire.

---

## Filtre Adaptatif AQ-KF (Myers-Tapley)

### Principe

Remplacer Q fixe par Q estimé en ligne à partir des innovations du filtre.

À chaque pas t :
1. Predict + Update standard
2. Innovation : `v[t] = z[t] - H @ x_pred[t]`
3. Sur fenêtre glissante W=30 : `C_vv = mean(v[k]²)`
4. Théorique : `S = H @ P_pred @ H.T + R`
5. Si `delta = C_vv - S > 0` : Q veut augmenter
6. Myers-Tapley : `Q_new = delta × C_rts @ C_rts.T` où `C_rts = P_filt[t] @ A.T @ inv(P_pred[t+1])`
7. Clipping : `Q_current = clip(Q_new, Q×0.1, Q×Q_max_factor)`

Le backward FLKS réutilise les mêmes fonctions `compute_slopes_test1/test2` — seul le forward filter change.

### Diagnostic : delta systématiquement positif

```
Delta (C_vv - S) sur MACD 30min BTC :
  min=11.56  median=581.65  P95=2134  max=14230
  delta > 0 : 100.0% du temps
  delta < 0 : 0.0% du temps
```

Le MACD 30min a un bruit de processus structurellement plus élevé que Q_fixe = 0.01. Delta ne s'inverse jamais — le filtre fixe sous-estime systématiquement le bruit.

### Q_max sweep : trouver le plafond optimal

| Q_max | Q effectif | K médian | AQ T1 Trans | AQ T2 k=3 Trans | Sous-pas effet |
|-------|-----------|---------|-------------|-----------------|----------------|
| **Q×10 (0.1)** | 0.1 | **0.82** | **74.37%** | **82.41%** | **+8pp** |
| Q×50 (0.5) | 0.5 | 0.94 | 74.62% | 75.38% | +0.8pp |
| Q×100 (1.0) | 1.0 | 0.97 | 74.12% | 74.37% | +0.3pp |
| Q×500 (5.0) | 5.0 | 0.99 | 73.62% | 73.87% | +0.3pp |

**Q×10 est optimal.** Au-delà, K sature → pas de filtrage → sous-pas sans effet.

### Divergence sans clipping

Sans Q_max, Q[1,1] explose à 14,300 (×1.4M vs fixe) → K=1.0 → `x_filt = z` (pas de filtrage) → AQ T2 k=1..6 tous identiques (73.37%).

### Résultats AQ-KF vs Standard (Q×10)

| Méthode | Standard Trans | AQ-KF Trans | Gain AQ |
|---------|---------------|-------------|---------|
| T1 (30m pur) | 30.40% | **74.37%** | **+43.97pp** |
| T2 k=1 (5min) | 58.54% | **79.65%** | +21.11pp |
| T2 k=3 (15min) | 75.88% | **82.41%** | +6.53pp |
| T2 k=6 (30min) | 80.90% | 81.66% | +0.75pp |

### Conclusion AQ-KF

1. **AQ-KF T1 seul (74%) surpasse le Standard T2 k=2 (71%).** L'adaptation de Q vaut ~2 sous-pas d'avance.
2. **AQ-KF T2 k=3 = 82.4% = meilleur résultat global en concordance.** Combine adaptive Q + 3 sous-pas.
3. **Si on a les sous-pas 5min, le standard est presque aussi bon** (80.9% à k=6 vs 82.4%).
4. **Le vrai avantage de l'AQ-KF est le T1 pur** : 74% vs 30% — utile si on n'a pas accès aux données 5min.

### Backtest PnL AQ-KF vs Standard (MACD)

Grid search 2D (holding × threshold) sur 8 méthodes (4 standard + 4 AQ-KF).

**Top 5 configs MACD (toutes méthodes confondues) :**

| Rang | Config | PnL | Trades | WR |
|------|--------|-----|--------|-----|
| 1 | **AQ:k=6 hold=8 thr=P75** | **+59.5%** | 184 | 55.4% |
| 2 | AQ:k=2 hold=10 thr=P75 | +58.3% | 176 | — |
| 3 | Std T2:k=1 hold=8 thr=P90 | +54.3% | 68 | 66.2% |
| 4 | Std T2:k=1 hold=10 thr=P90 | +51.0% | 68 | — |
| 5 | AQ:k=6 hold=6 thr=P75 | +49.2% | 194 | — |

**Observations :**

1. **L'AQ-KF produit le meilleur PnL absolu (+59.5%)**, battant le standard (+54.3%) de +5pp.
2. **Le seuil optimal AQ-KF est P75 (22.0)**, pas P90 (36.8). Le filtre adaptatif change l'échelle des pentes.
3. **L'AQ-KF fait plus de trades** (184 vs 68) car les pentes adaptatives sont plus stables et déclenchent plus de reversals qui tiennent.
4. **Le standard T2:k=1 thr=P90 reste excellent** : 68 trades, 66% WR, signal plus propre mais PnL légèrement inférieur.
5. **Toutes les configs positives battent Buy & Hold (-19.7%)** de +70pp à +79pp.

**Zone chaude par méthode :**

| Méthode | Zone optimale | PnL max |
|---------|---------------|---------|
| Standard | hold=8, thr=P90 | +54.3% |
| **AQ-KF** | **hold=8-10, thr=P75** | **+59.5%** |

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
| **Backward Test 2 en 2 pas** | Sautait le lissage de t, sur-pondérait x_prov | Ajout Pas 1 avec C_partial (transition A_sub^k) |
| **Exécution backtest au close[t]** | T2 exécutait 5-30min trop tôt | T2 exécute au close du step k de candle t+1 |

---

## Scripts

| Script | Rôle |
|--------|------|
| `src/signal_processing/flks_substep_convergence.py` | Concordance de signe (Test 1 + Test 2 k=1..6, 3 indicateurs) |
| `src/signal_processing/flks_backtest_pnl.py` | Backtest PnL avec grid search holding × threshold |
| `src/signal_processing/flks_vs_filter.py` | Premier script exploratoire (obsolète) |
| `src/signal_processing/flks_30m_vs_5m_micro.py` | Version intermédiaire (obsolète) |
| `src/signal_processing/validate_kalman.py` | Validation forward filter numpy vs pykalman |

### Commandes

```bash
# Concordance de signe (3 indicateurs)
python src/signal_processing/flks_substep_convergence.py --csv data_trad/BTCUSD_all_5m.csv

# Backtest PnL (grid search holding × threshold)
python src/signal_processing/flks_backtest_pnl.py --csv data_trad/BTCUSD_all_5m.csv

# Validation forward filter
python src/signal_processing/validate_kalman.py --csv data_trad/BTCUSD_all_5m.csv
```

---

## Conclusion

### Concordance de signe (traitement du signal)

**Le FLKS avec micro-updates 5min améliore massivement la détection des transitions :**

| Indicateur | Trans T1 | Trans k=1 | Trans k=6 | Gain total |
|------------|----------|-----------|-----------|------------|
| **MACD** | 30.40% | 58.54% | 80.90% | **+50.50pp** |
| **CCI** | 38.64% | 61.70% | 80.75% | **+42.11pp** |
| **RSI** | 45.34% | 61.49% | 81.52% | **+36.18pp** |

Le FLKS bat le LSTM CNN (49% MACD Trans, STATUS_v3.0) dès k=1 (59%) avec un simple filtre linéaire.

### Backtest PnL (trading réel)

**La concordance ne se traduit en PnL qu'avec des filtres agressifs :**

- Sans filtre : toutes les variantes perdent (-37% à -158%)
- Le problème n'est pas la qualité du signal mais les **micro-reversals parasites**

**Best configs MACD :**

| Rang | Config | PnL | Trades | WR |
|------|--------|-----|--------|-----|
| 1 | **AQ:k=6 hold=8 thr=P75** | **+59.5%** | 184 | 55.4% |
| 2 | AQ:k=2 hold=10 thr=P75 | +58.3% | 176 | — |
| 3 | Std T2:k=1 hold=8 thr=P90 | +54.3% | 68 | 66.2% |

Buy & Hold sur la même période : **-19.7%**.

### Filtre adaptatif AQ-KF

**L'AQ-KF améliore à la fois la concordance ET le PnL :**

| Métrique | Standard (meilleur) | AQ-KF (meilleur) |
|----------|--------------------|--------------------|
| Concordance Trans | 80.90% (T2 k=6) | **82.41%** (T2 k=3) |
| PnL | +54.3% (T2 k=1 hold=8 thr=P90) | **+59.5%** (k=6 hold=8 thr=P75) |
| Trades | 68 | 184 |
| WR | 66.2% | 55.4% |

L'AQ-KF produit un signal de meilleure qualité en concordance (+1.5pp) ET en PnL (+5.2pp). Le standard fait moins de trades mais avec un WR plus élevé — deux philosophies différentes.

### Validation Out-of-Sample (OOS)

**Test des 2 configs gagnantes avec paramètres FIXÉS sur 3 périodes :**

| Période | Durée | Config Std | Config AQ | Oracle | B&H |
|---------|-------|-----------|-----------|--------|-----|
| **In-sample [1000:5000]** | 83j | **-15.9%** (35t, 34%) | **-49.0%** (159t, 39%) | +22.8% | +8.0% |
| OOS-early [0:1000] | 21j | +1.6% (12t, 50%) | -8.9% (38t, 42%) | +20.3% | +1.8% |
| **OOS-next [5000:10000]** | 83j | **+41.1%** (81t, 62%) | **+49.8%** (225t, 52%) | +133.9% | -22.8% |

**⚠️ SUROPTIMISATION CONFIRMÉE :**

1. **In-sample échoue** : les seuils optimisés sur [5000:10000] ne fonctionnent pas sur [1000:5000]. Les deux configs perdent.
2. **OOS-next = la période d'optimisation** : les +41/+49% sont sur la même période où les seuils ont été calibrés (les 5000 dernières bougies du CSV).
3. **L'Oracle varie de +22.8% à +133.9%** selon la période. La période récente (marché baissier, B&H -22.8%) est structurellement plus favorable au signal MACD.

**Conclusion OOS** : les seuils fixes (P75=22.0, P90=36.8) ne généralisent pas. Le signal MACD existe mais son amplitude varie avec le régime de marché. Des seuils adaptatifs (calibrés sur fenêtre glissante) seraient nécessaires.

### Ce que cette session a prouvé

1. Le signal de pente **existe** (Oracle +22% à +134% selon la période)
2. Le FLKS **détecte** les transitions mieux que le LSTM (59% vs 49% avec 5min de délai)
3. Le FLKS **ne suffit pas seul** — les micro-reversals détruisent le PnL
4. **Seuil + holding minimum** rendent le signal profitable sur la période d'optimisation
5. **L'AQ-KF améliore le T1 de +44pp** — l'adaptation de Q vaut ~2 sous-pas d'avance
6. **AQ-KF T2 k=3 = 82.4%** — meilleur concordance de la session
7. **⚠️ Suroptimisation confirmée** — seuils fixes ne généralisent pas hors échantillon
8. **Le régime de marché domine** — Oracle +134% en bear vs +23% en bull/range

---

## ML Pipeline avec features AQ-KF

### LSTM vs XGBoost sur features AQ-KF (BTC, MACD 30m)

Features identiques : macd_30m_live (brut), macd_30m_filtered (AQ-KF position), macd_30m_velocity (AQ-KF vélocité).
Labels identiques : oracle pykalman.smooth() (non-causal).

| Métrique | LSTM Original | LSTM AQ-KF | XGBoost AQ-KF |
|----------|---------------|------------|---------------|
| Val Accuracy | 89.8% | 91.1% | 91.0% |
| Ratio switchs | 2.5× | 2.8× | 2.9× |
| Justified (±6) | 57.4% | 59.4% | 59.6% |
| Spurious (>20) | 18.0% | 20.0% | 19.9% |
| Within 6 steps | 90.8% | 93.0% | 93.2% |
| Prob before trans | 0.50 | 0.50 | 0.50 |

Les 3 modèles convergent vers le même plafond (~91% acc, ~60% justified, ~20% spurious).

### Feature importance XGBoost

```
f2_step24 (velocity au dernier step) : 31.0%
f2_step23 (velocity à l'avant-dernier) : 14.1%
f2_step18 : 8.4%
→ velocity = 85% de l'importance totale
→ Le modèle détecte les transitions APRÈS qu'elles commencent, pas avant
```

### Discriminabilité des switches (faux vs vrai)

Test XGBoost faux_up vs vrai_up / faux_down vs vrai_down :

| Direction | Faux samples | Vrai samples | Test accuracy | Verdict |
|-----------|-------------|-------------|---------------|---------|
| **UP** | 652 | 1,447 | **87.8%** | DISTINGUABLE |
| **DOWN** | 653 | 1,485 | **89.6%** | DISTINGUABLE |

**Les patterns sont distinguables.** Les vrais switches ont :
- Velocity plus forte (7.1 vs 2.7 pour UP)
- MACD live plus loin de zéro (-64 vs +1.8 pour UP)

Les faux switches se produisent quand le MACD oscille autour de 0 avec peu de momentum.

### Filtre simple : velocity + macd_live

Grid search de seuils sur les prédictions XGBoost :

| Config | Switches | Ratio | Justified | Spurious | Détection |
|--------|----------|-------|-----------|----------|-----------|
| Baseline | 6,574 | 2.9× | 59.6% | 19.9% | 94.0% |
| vel=5.2 | 3,233 | 1.4× | 61.8% | 15.8% | 76.8% |
| vel=10.4, macd=23.7 | 1,967 | 0.9× | 40.9% | — | 42.3% |

Le filtre réduit le ratio mais la détection chute proportionnellement.

### Backtest PnL avec filtre

| Config | PnL | Trades | WR |
|--------|-----|--------|-----|
| Baseline (no filter) | -1,299% | 6,575 | 20.8% |
| vel=5.2, hold=8 | -591% | 2,944 | 34.6% |
| vel=10.4, macd=23.7, hold=8 | **-313%** | 1,939 | 41.7% |
| **Buy & Hold** | **+42.4%** | 1 | — |

**Aucune configuration n'est rentable.** Le meilleur (-313%) est encore loin du B&H (+42%).

### Backtest consensus : le signal EST rentable quand le modèle confirme

**Test clé** : suivre l'oracle mais trader uniquement les transitions que le modèle détecte aussi (±6 steps, même direction).

| Method | PnL | Trades | WR |
|--------|-----|--------|-----|
| **Oracle** (toutes transitions) | **+889%** | 2,285 | 66.3% |
| **Consensus** (oracle + modèle confirme) | **+614%** | **2,008** | **64.4%** |
| Modèle seul (tous switches) | -1,299% | 6,575 | 20.8% |
| Buy & Hold | +42% | 1 | — |

**Détails consensus :**
- Oracle transitions : 2,284
- Modèle confirme (±6 steps, même direction) : **2,138 (93.6%)**
- Modèle ne confirme pas : 146 (6.4%)
- Trades effectifs : 2,008

**Analyse :**

Le consensus capture **69% du PnL oracle** (+614/+889) avec 88% des trades. Le modèle confirme 93.6% des transitions dans la bonne direction.

Décomposition du PnL modèle seul :

| | Trades | PnL estimé |
|---|---|---|
| Transitions correctes (~2,008) | **+614%** |
| Faux switches (~4,567) | **~-1,913%** |
| **Total modèle** | **-1,299%** |

**Le signal de direction est excellent quand il est correct** (64% WR, +614%). Le problème est exclusivement les **4,567 faux switches** qui détruisent +1,913% de PnL.

### Discriminabilité des faux switches — POST-HOC (⚠️ biais)

Test XGBoost : les faux switches sont-ils distinguables des vrais dans les features ?

| Direction | Test accuracy | Verdict |
|-----------|--------------|---------|
| UP (faux_up vs vrai_up) | **87.8%** | Distinguable post-hoc |
| DOWN (faux_down vs vrai_down) | **89.6%** | Distinguable post-hoc |

**Caractéristiques discriminantes au moment du switch :**

| Feature | Faux switch | Vrai switch | Clé |
|---------|-------------|-------------|-----|
| macd_live | ~0 (neutre) | ±64 (loin de 0) | **Amplitude** |
| velocity | ±2.7 (faible) | ±7.1 (forte) | **Momentum** |

**⚠️ BIAIS IMPORTANT** : cette discriminabilité est **post-hoc**. Le test utilise les labels oracle pour séparer faux/vrai APRÈS coup. En production, on n'a pas l'oracle.

Le modèle principal a **déjà** ces features (macd_live, velocity) dans ses inputs. S'il ne les utilise pas pour éviter les faux switches, c'est parce que :
- Sa tâche est de prédire la **direction** à chaque step, pas de prédire si son prochain switch sera vrai ou faux
- Le filtre simple (velocity > 5.2) qui utilise cette info → PnL = -591% (toujours négatif)
- Les 88% de discriminabilité post-hoc ne se traduisent pas en filtre temps réel rentable

### Conclusion ML Pipeline

Le **consensus (+614%) est un upper bound inaccessible** en production (nécessite l'oracle).

Le modèle confirme 93.6% des transitions oracle → le signal direction est bon quand il est correct. Mais les 4,567 faux switches (-1,913% PnL) ne peuvent pas être filtrés en temps réel avec les features actuelles.

Le plafond est **structurel** : les features prix-dérivées (MACD, Kalman, velocity) ne permettent pas de distinguer en temps réel un vrai retournement d'une oscillation autour de zéro.

---

## Synthèse finale de la session

### Ce qui fonctionne (traitement du signal)

| Résultat | Valeur |
|----------|--------|
| FLKS concordance Trans (AQ T2 k=3) | **82.4%** |
| AQ-KF T1 seul (sans sous-pas) | **74.4%** |
| FLKS bat LSTM aux transitions | 59% vs 49% |
| Consensus PnL (upper bound, nécessite oracle) | +614% |
| Modèle détecte 93.6% des transitions oracle | 2,138/2,284 |

### Ce qui ne fonctionne pas (trading réel)

| Résultat | Valeur |
|----------|--------|
| FLKS PnL (OOS, seuils fixes) | -16% à -49% (suroptimisation) |
| ML PnL modèle seul (toutes configs) | -313% à -1,299% |
| Filtre velocity+macd_live | -591% à -313% |
| Discriminabilité post-hoc (88%) | ne se traduit pas en filtre rentable |
| Buy & Hold | +42% |

### Le diagnostic final

1. Le **signal de pente MACD existe** : Oracle +889%, modèle confirme 93.6% des transitions
2. Le **consensus est un upper bound** (+614%) — inaccessible sans oracle
3. Les **4,567 faux switches** détruisent le PnL (-1,913%)
4. Les faux switches sont distinguables **post-hoc** (88%) mais **pas en temps réel** avec les features actuelles
5. Le **plafond est structurel** : features prix-dérivées insuffisantes pour filtrer le bruit en temps réel
6. Le modèle sait **QUOI** (direction correcte 91%) mais pas **QUAND switcher** (faux switches indistinguables en temps réel)

---

## Viterbi Post-Processing — Premier PnL Positif ML

### Principe

Au lieu de seuiller chaque prédiction indépendamment (`prob > 0.5 → UP`), on applique un **décodage Viterbi** sur toute la séquence de probabilités. Une matrice de transition pénalise les switches :

```
transition = [[p, 1-p],    p = self-transition probability
              [1-p, p]]    (0.9 à 0.99)
```

Le Viterbi trouve la séquence d'états **globalement optimale** en considérant à la fois les probabilités du modèle ET le coût de switching. Pas de réentraînement nécessaire.

**Référence** : Viterbi (1967), utilisé en audio/vidéo pour le problème d'"anti-flickering".

### Résultats

| Méthode | PnL | Switches | Ratio | WR |
|---------|-----|----------|-------|----|
| Oracle | +889% | 2,284 | 1.0× | 66.3% |
| **Viterbi p=0.99** | **+21.0%** | **2,618** | **1.1×** | **45.8%** |
| Viterbi p=0.97 | -23.1% | 2,708 | 1.2× | 44.6% |
| Viterbi p=0.95 | -49.8% | 2,766 | 1.2× | 43.7% |
| CUSUM h=8 | -344% | 1,800 | 0.8× | 39.5% |
| Baseline (seuil 0.5) | -1,299% | 6,574 | 2.9× | 20.8% |
| Buy & Hold | +42.4% | — | — | — |

### Analyse

1. **Viterbi p=0.99 = premier PnL positif ML de la session** (+21%). Réduit les switches de 6,574 → 2,618 (-60%).
2. **p=0.99 signifie 99% de probabilité de rester dans l'état actuel**. Il faut une évidence très forte pour switcher — cohérent avec le diagnostic (le modèle doit être conservateur).
3. **CUSUM échoue** : réduit les switches mais perd le timing. Le Viterbi est supérieur car il optimise la séquence globalement.
4. **PnL encore sous Buy & Hold** (+21% vs +42%). Le Viterbi améliore mais ne résout pas complètement.
5. **WR = 45.8%** (vs 20.8% baseline). Le filtre élimine les trades les plus perdants.

### ⚠️ Limitations

- p=0.99 est un paramètre optimisé sur les données de test (risque de suroptimisation)
- Le Viterbi standard est **non-causal** (utilise la séquence complète). En production, il faudrait un Viterbi online (forward-only) qui serait moins performant
- +21% reste sous Buy & Hold (+42%)

---

## FLKS Slopes comme Features ML — Percée

### Découverte du problème

Les tests FLKS montraient 74-82% de concordance aux transitions, mais le LSTM ne recevait que la `velocity` brute (68% Trans) et la position `filtered`. Les pentes FLKS backward (qui utilisent l'info de la bougie courante pour lisser les positions passées) n'étaient **pas dans les features** du modèle.

### Fix : pentes FLKS Standard comme features

CSV généré avec `prepare_flks_csv.py` :
- 879,710 lignes (résolution 5min, forward-fill des pentes 30min)
- Features : `std_k1_slope` à `std_k6_slope` (6 pentes Standard FLKS)
- Label : `oracle_label_macd_30m` (pykalman.smooth, inchangé)

Validation concordance sur 146k bougies 30min (toute la série BTC) :

| Méthode | Std All | Std Trans |
|---------|---------|-----------|
| k=1 (5min) | 93.24% | 57.20% |
| k=3 (15min) | 94.58% | 73.81% |
| k=6 (30min) | 95.67% | 80.82% |

Les chiffres tiennent sur 8 ans de données.

### Résultats XGBoost sur FLKS slopes

| KPI | Anciennes features | **FLKS slopes** | Amélioration |
|-----|-------------------|-----------------|--------------|
| **Test Accuracy** | 91.2% | **96.3%** | **+5.1pp** |
| **Ratio switches** | 2.9× | **1.2×** | **÷2.4** |
| **Justified** | 59.6% | **89.4%** | **+30pp** |
| **Spurious** | 19.9% | **7.6%** | **-12pp** |
| Within 6 steps | 93.2% | **98.6%** | +5.4pp |
| Instant (0 step) | 55.5% | **82.8%** | +27pp |
| Grey zone [0.4,0.6] | 3.6% | **1.1%** | -2.5pp |
| 0 switches/plateau | 23.4% | **70.1%** | **+47pp** |

### Détails KPIs

**Switches** : 2,631 modèle vs 2,283 oracle (ratio 1.2×). Seulement 348 faux switches au lieu de 4,290 avec les anciennes features.

**Détection** : 99.6% des transitions oracle détectées (2,273/2,283). 82.8% détectées instantanément (latence 0).

**Précision** : 89.4% des switches du modèle sont à ±6 steps d'une vraie transition. 64.3% sont exactement au bon moment.

**Plateaux** : 70.1% des plateaux n'ont aucun switch parasite (vs 23.4% avant).

### Feature importance XGBoost

```
k6_step24 (pente k=6 au dernier step)    : 29.4%
k5_step24 (pente k=5 au dernier step)    : 12.6%
k6_step23 (pente k=6 à l'avant-dernier)  :  4.0%
→ Les pentes les plus récentes et les plus complètes dominent
```

### Pourquoi ça marche

Les anciennes features (velocity brute) avaient 68% de concordance aux transitions. Les pentes FLKS backward ont 57-81% selon le sous-pas. Le modèle reçoit directement le signal qui a été **validé** dans les tests FLKS — pas une approximation dégradée.

Le passage de 2.9× à 1.2× ratio montre que les pentes FLKS contiennent assez d'information pour que le modèle distingue les vrais retournements du bruit, sans post-processing.

---

## Synthèse finale de la session

### Ce qui fonctionne

| Résultat | Valeur |
|----------|--------|
| FLKS concordance Trans (AQ T2 k=3) | **82.4%** |
| AQ-KF T1 seul (sans sous-pas) | **74.4%** |
| FLKS bat LSTM aux transitions | 59% vs 49% |
| Consensus PnL (upper bound, nécessite oracle) | +614% |
| Viterbi p=0.99 PnL (anciennes features) | +21.0% |
| **XGBoost FLKS slopes : 96.3% accuracy** | **Ratio 1.2×** |
| **89.4% justified, 7.6% spurious** | **vs 59.6% / 19.9%** |
| **82.8% détection instantanée** | **vs 55.5%** |

### Ce qui ne fonctionne pas

| Résultat | Valeur |
|----------|--------|
| ML avec anciennes features (live, filtered, velocity) | 91% acc, ratio 2.9× |
| Filtre velocity+macd_live | -591% à -313% PnL |
| CUSUM | -344% PnL |

### Le diagnostic final

1. Le **signal de pente MACD existe** : Oracle +889%, concordance 57-82% selon la méthode
2. Le problème était dans les **features** : le modèle ne recevait pas les pentes FLKS backward
3. Avec les **bonnes features** (FLKS slopes k=1..6) : accuracy 96.3%, ratio 1.2×
4. Les faux switches passent de **4,290 à 348** (-92%)
5. **Backtest PnL à confirmer** avec ces nouvelles prédictions

---

## Pistes pour la suite

1. **Backtest PnL** avec les prédictions FLKS slopes (ratio 1.2× devrait être rentable)
2. **Viterbi post-processing** sur les prédictions FLKS (ratio 1.2× → ~1.0× ?)
3. **AQ-KF slopes** en plus des Standard : comparer les 2 filtres en features
4. **LSTM** sur les mêmes features pour comparer XGBoost vs LSTM
5. **Validation OOS** : vérifier que 96.3% tient sur une période séparée
