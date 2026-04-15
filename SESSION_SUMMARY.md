# SESSION SUMMARY — Multi-Timeframe Trading Model Experiments

**Date**: 2026-04-13 → 2026-04-15 (3 jours)
**Repo**: ecolealgerienne-ui/trad
**Branch**: claude/review-trading-projects-aJuC4
**Asset testé**: BTC (single asset)

---

## Ce qui a été fait

### Pipeline construit from scratch

1. **`prepare_multitf_csv.py`** — Génère des CSV multi-timeframe (30min + 1h) à résolution 5min
   - OHLCV live-style (reproduit le comportement de l'API Binance : dernière bougie en formation)
   - Indicateurs (MACD, RSI, CCI) avec EMA incrémentale freeze/provisional
   - Kalman causal (filter_update, forward-only)
   - Oracle labels non-causaux (kf.smooth, RTS smoother)
   - Oracle slopes continus (pour régression)
   - Vélocité Kalman (state[1])
   - Validation atol=1e-10 à chaque clôture de bougie
   - Détection de closure par changement de bucket (gère les gaps)

2. **`train_multitf.py`** — Entraînement avec de nombreuses options
   - `--crossfeat` : 6 features (3 indicateurs × live + filtered) pour 30m, 12 pour 1h
   - `--arch {cnn-lstm, cnn-gru, tcn}` : 3 architectures
   - `--target-type {binary, continuous}` : classification ou régression
   - `--window {12, 25, 50}` : taille de la fenêtre
   - Z-score normalization par asset, stats train only
   - Split chronologique 70/15/15 avec gap=window

3. **Scripts d'analyse** (10+) :
   - `analyze_predictions.py` — KPIs trading (latence, switchs, spurious, plateaux)
   - `compare_all_models.py` — Comparaison des 6 modèles
   - `analyze_regression_deep.py` — R² conditionnel (transition vs plateau)
   - `analyze_magnitude_filter.py` — Filtre par magnitude de pente
   - `analyze_magnitude_dynamics.py` — Trajectoire temporelle autour des switchs
   - `analyze_switch_discrimination.py` — Discrimination cross-model
   - `analyze_cross_tf_discrimination.py` — Discrimination cross-timeframe
   - `analyze_cross_arch_switches.py` — Vote cross-architecture
   - `compare_binary_vs_regression.py` — Binaire vs régression
   - `backtest_pnl_v1.py` — Backtest PnL avec 4 configurations

---

## ~40 expériences réalisées

### Features testées
| Config | Features | Ratio switchs |
|--------|----------|--------------|
| Single baseline | 2 (live + filtered) | 2.8× |
| + velocity | 3 | 2.7× |
| **Crossfeat** | **6** (3 ind × 2) | **2.2-2.5×** |
| Cross + velocity | 9 | 2.4× |

### Architectures testées
| Arch | Val loss | Ratio |
|------|---------|-------|
| CNN-LSTM | 0.2382 | **2.5×** (meilleur ratio) |
| CNN-GRU | **0.2317** (meilleur loss) | 2.5× |
| TCN | **0.2243** | 2.4× |

### Régression (6 modèles)
| Modèle | R² global | R² transition | R² plateau |
|--------|----------|--------------|-----------|
| macd_30m | 0.91 | **−0.24** | 0.92 |
| rsi_1h | 0.75 | **−1.07** | 0.77 |

R² négatif aux transitions pour les 6 modèles.

### Filtrage des switchs
| Approche | Meilleur ratio |
|----------|---------------|
| Cross-model | 1.0-1.4× |
| Cross-timeframe | 1.0-1.7× |
| Magnitude | 1.2× |
| Magnitude dynamique | 1.4× |
| **R_strong_agree** | **1.2×** |
| **Vote unanime 3 archi** | **0.5×** |

### Backtest PnL (BTC, 2024-2026)
| Config | PnL Net | Trades | WR |
|--------|---------|--------|-----|
| Ultra-conservateur | −1,779% | 5,595 | 15.5% |
| Conservateur | −1,182% | 2,746 | 21.1% |
| Modéré | −1,786% | 5,640 | 15.7% |
| Agressif | −1,768% | 5,628 | 16.3% |
| **Buy & Hold** | **+40%** | 1 | N/A |

---

## Conclusion définitive

**Le modèle sait QUOI (direction 91%) mais pas QUAND (transitions R²<0).**

Le signal Oracle (non-causal, kf.smooth) montre que le signal EXISTE (+8,316% PnL net pour 30min pur). Mais il nécessite de voir le futur. Toutes les tentatives d'approximation causale (classification, régression, cross-model, cross-archi, magnitude, fenêtre) butent sur le même plafond : les transitions crypto sont imprévisibles depuis les features prix seules.

### Preuves du plafond structurel
- Persistence baseline (98%) bat tous les modèles (83-91%)
- R² négatif aux transitions pour 6/6 modèles régression
- 56-68% d'erreurs partagées entre 3 architectures
- Probabilité avant transition = probabilité mid-plateau
- Window 12/25/50 → même accuracy
- Backtest PnL : 0/4 configs profitable, 0/4 bat Buy & Hold

---

## Fichiers clés sur le repo

### Scripts
```
src/prepare_multitf_csv.py      — Génération CSV (live + oracle labels)
src/train_multitf.py            — Entraînement (3 archi, binaire/régression, crossfeat)
src/backtest_pnl_v1.py          — Backtest PnL 4 configs
src/compare_all_models.py       — KPIs comparatifs
src/analyze_predictions.py      — KPIs trading détaillés
src/analyze_regression_deep.py  — R² conditionnel
src/compare_binary_vs_regression.py
src/analyze_cross_arch_switches.py
src/analyze_magnitude_filter.py
src/analyze_magnitude_dynamics.py
```

### Documentation
```
STATUS_v3.0.md   — Résultats post-normalisation MACD (complet)
STATUS_v2.2.md   — Récap pré-normalisation (tous les tableaux)
STATUS_v2.1.md   — Analyses détaillées (KPIs, cross-model, régression)
STATUS_v2.0.md   — Résultats Oracle et pipeline initial
PLAN_v3.0.md     — Plan de reprise post-normalisation
PREVIOUS_EXPERIMENTS_SUMMARY.md — Historique du projet original (Phase 1-2.18)
```

### Données (sur la machine de l'utilisateur, pas dans le repo)
```
data/prepared/BTCUSD_multitf_macd_rsi_cci.csv  — CSV BTC (~491 MB)
data/prepared/*_crossfeat_dataset.npz           — Prédictions binaires
data/prepared/*_crossfeat_regression_dataset.npz — Prédictions régression
data/prepared/*_crossfeat_cnngru_dataset.npz    — Prédictions GRU
data/prepared/*_crossfeat_tcn_dataset.npz       — Prédictions TCN
models/best_model_*.pth                          — ~22 modèles entraînés
```

---

## Ce qui n'a PAS été essayé (pistes pour la suite)

1. **Volume** — existe dans les CSV mais jamais utilisé comme feature ML
2. **Données non-prix** — funding rates, order book, liquidations, sentiment
3. **Multi-asset training** — entraîner sur 5 assets au lieu de BTC seul
4. **LLM / approche générative** — paradigme fondamentalement différent
5. **Event-driven** — détecter des événements plutôt que prédire continuellement
6. **Timeframe plus long** (4h, daily) — réduire encore le bruit
7. **Post-processing avancé** — hysteresis, holding minimum sur les signaux existants

---

## Prompt pour la prochaine session

```
CONTEXTE
Je travaille sur un projet de trading crypto algorithmique (repo: ecolealgerienne-ui/trad).
Après 3 jours intensifs et ~40 expériences, j'ai conclu que la prédiction
directionnelle à partir d'indicateurs techniques (MACD, RSI, CCI) sur des
timeframes 30min et 1h atteint un plafond structurel :

- Accuracy 91% sur la direction (R²=0.91 sur les plateaux)
- Mais R² NÉGATIF aux transitions (les moments critiques pour le trading)
- Backtest PnL : -1,182% à -1,786% vs Buy & Hold +40%
- Le signal Oracle (non-causal) montre +8,316% — le signal EXISTE mais
  nécessite de voir le futur

Le pipeline complet est en place :
- CSV multi-timeframe live-style (Binance API behavior)
- Training CNN-LSTM/GRU/TCN avec crossfeat 6-12 features
- Scripts d'analyse KPIs, régression, discrimination, backtest

DOCUMENTATION CLÉS :
- STATUS_v3.0.md : résultats post-normalisation (complet)
- STATUS_v2.2.md : récap de toutes les expériences
- PREVIOUS_EXPERIMENTS_SUMMARY.md : historique du projet original

OBJECTIF DE CETTE SESSION :
[Définir ici : explorer le volume comme feature ? données non-prix ?
approche LLM ? multi-asset ? autre ?]
```
