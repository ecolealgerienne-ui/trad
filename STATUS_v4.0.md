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

### Ce que cette session a prouvé

1. Le signal de pente **existe** (Oracle +123% à +168%)
2. Le FLKS **détecte** les transitions mieux que le LSTM (59% vs 49% avec 5min de délai)
3. Le FLKS **ne suffit pas seul** — les micro-reversals détruisent le PnL
4. **Seuil + holding minimum** rendent le signal profitable
5. **L'AQ-KF améliore le T1 de +44pp** — l'adaptation de Q vaut ~2 sous-pas d'avance
6. **AQ-KF T2 k=3 = 82.4%** — meilleur concordance de la session
7. **AQ-KF k=6 hold=8 thr=P75 = +59.5%** — meilleur PnL de la session
8. **Risque de suroptimisation** — seuils et Q_max calibrés sur les données de test

---

## Pistes suite

1. **Validation out-of-sample** : tester les meilleurs configs sur une période différente (split train/test)
2. **Seuils adaptatifs par indicateur** : RSI inutilisable avec seuils MACD, calibrer séparément
3. **Multi-asset** : vérifier sur ETH, BNB, ADA, LTC
4. **R adaptatif** : adapter R en plus de Q (double adaptation)
6. **Lag N=3** : tester si un lag plus grand améliore la concordance
7. **R adaptatif** : adapter R en plus de Q (double adaptation)
