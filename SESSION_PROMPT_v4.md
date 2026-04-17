# Prompt pour la prochaine session

```
CONTEXTE
=========
Projet de trading crypto algorithmique (repo: ecolealgerienne-ui/trad).
Branche: claude/evaluate-trading-indicators-PBdp0
Asset: BTC uniquement, timeframe 30min, indicateur MACD.

HISTORIQUE RAPIDE
=================
Sessions précédentes (STATUS_v2.0 à v3.0):
- Pipeline LSTM/GRU/TCN sur MACD/RSI/CCI 30min et 1h
- ~40 expériences, plafond structurel: 91% accuracy mais R² négatif aux transitions
- Backtest PnL: -1,182% à -1,786% (vs Buy & Hold +40%)
- Conclusion: le modèle sait QUOI (direction 91%) mais pas QUAND (transitions)

Session actuelle (STATUS_v4.0, 57 commits):
- Approche traitement du signal: Fixed-Lag Kalman Smoother (FLKS)
- Filtre adaptatif AQ-KF (Myers-Tapley, Q adaptatif en ligne)
- Percée: pentes FLKS backward comme features ML → 96.3% accuracy, ratio 1.2×

RÉSULTAT CLÉ
=============
XGBoost entraîné sur pentes FLKS Standard (std_k1_slope à std_k6_slope):
- Test accuracy: 96.3% (vs 91% avec anciennes features)
- Switch ratio: 1.2× (vs 2.9× avant — quasi-identique à l'oracle)
- Justified: 89.4% (vs 59.6%)
- Spurious: 7.6% (vs 19.9%)
- Backtest PnL modèle seul: +870% sur 458 jours (exécution close 5min)
- Oracle PnL: +890% sur la même période
- Buy & Hold: +38.8%

⚠️ CES RÉSULTATS NÉCESSITENT UN AUDIT. Le PnL est élevé (~1.9%/jour)
et des bugs d'alignement ont été trouvés et corrigés pendant la session.

ARCHITECTURE DU SIGNAL
=======================
1. MACD brut calculé sur bougies 30min (EMA 12/26/9)
2. Kalman forward filter (Q=0.01, R=0.1, A=[[1,1],[0,1]])
3. FLKS backward 2 pas: à chaque bougie t, lisser t-1 et t-2
   slope[t] = smoothed[t-1] - smoothed[t-2]
4. Variante k=1..6: entre chaque bougie 30min, injecter les MACD live
   5min provisoires pour affiner le lissage
5. Oracle: pykalman.smooth() sur toute la série (non-causal, référence)
6. Label: sign(slope_oracle) → UP/DOWN

CONCORDANCE VÉRIFIÉE SUR 146k BOUGIES 30min (8 ans BTC):
| Méthode     | Std All | Std Trans | AQ All | AQ Trans |
|-------------|---------|-----------|--------|----------|
| T1 (0 pas)  | 89.93%  | 26.68%    | 89.47% | 73.65%   |
| k=1 (5min)  | 93.24%  | 57.20%    | 89.93% | 79.96%   |
| k=3 (15min) | 94.58%  | 73.81%    | 90.14% | 82.19%   |
| k=6 (30min) | 95.67%  | 80.82%    | 90.21% | 81.73%   |

PIPELINE ACTUEL (5 scripts)
============================
1. src/signal_processing/prepare_flks_csv.py
   - Lit CSV brut 5min → resample 30min → MACD → Kalman Standard + AQ-KF
   - FLKS backward slopes T1 + k=1..6 pour les 2 filtres
   - Oracle labels (pykalman.smooth)
   - Sort: data/prepared/BTCUSD_flks_features.csv (880k lignes × 22 colonnes)

2. src/signal_processing/train_flks_slopes.py
   - Lit le CSV features → split 70/15/15 → z-score train only
   - Séquences 25 steps × 6 features (std_k1..k6_slope)
   - XGBoost (ou LSTM) → NPZ avec prédictions + closes + dates

3. src/backtest_consensus_direction.py
   - Lit NPZ → backtest Oracle, Modèle seul, Consensus
   - Exécution au close 5min (via closes embarquées dans le NPZ)

4. src/analyze_predictions_aqkf.py
   - KPIs: latence, switchs, justified/spurious, distribution proba

5. src/signal_processing/core.py (620+ lignes)
   - Toutes les fonctions partagées: load_csv, Kalman, FLKS, métriques, backtest

FICHIERS DE DONNÉES
====================
- data_trad/BTCUSD_all_5m.csv — CSV brut 5min (879,710 lignes)
- data/prepared/BTCUSD_flks_features.csv — Features FLKS (879,710 × 22)
- data/prepared/macd_30m_dataset.npz — Prédictions + closes + dates test

BUGS TROUVÉS ET CORRIGÉS (session v4.0)
=========================================
1. t=0 sans update Kalman (P_filt[0] = I au lieu de posterior)
2. Oracle numpy pur ≠ pykalman.smooth (remplacé par pykalman directement)
3. Closes 5min brutes injectées au lieu de MACD live (espace incohérent)
4. Q non scalé pour 5min (6× trop de process noise)
5. Backward depuis x_filt[t+lag] (regardait le futur, non-causal)
6. x_pred incohérent avec x_filt (gains RTS basés sur états avant micro-updates)
7. Backward Test 2 en 2 pas au lieu de 3 (sautait le lissage de t)
8. Exécution backtest au close[t] pour Test 2 (5-30min trop tôt)
9. CSV source différent entre training et backtest (désalignement total)
10. Closes non incluses dans NPZ (trades = 0)
11. Discriminabilité 88% post-hoc ≠ filtre temps réel (biais oracle)

DOCUMENTATION CLÉS
===================
- STATUS_v4.0.md — Résultats complets (862 lignes)
- STATUS_v3.0.md — Résultats pré-FLKS (LSTM/GRU/TCN)
- SESSION_SUMMARY.md — Résumé sessions précédentes

OBJECTIF DE CETTE SESSION
===========================
1. AUDIT COMPLET du pipeline (prepare → train → backtest):
   - Vérifier que les features FLKS sont causales (pas de data leakage)
   - Vérifier que l'oracle smooth ne fuit pas dans les features
   - Vérifier l'alignement closes/labels/prédictions
   - Vérifier que le split train/val/test est correct
   - Comprendre le +870% PnL: réaliste ou artifact?

2. Si l'audit passe → validation OOS sur une période séparée

3. Tester avec AQ-KF slopes en plus des Standard (complémentarité
   montrée: corrélation 0.935, AQ détecte 217 transitions exclusives)
```
