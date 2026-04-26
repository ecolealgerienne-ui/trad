# NEW SESSION PROMPT — Post-Foundation-Finetune (clôture v14)

**Branche précédente clôturée** : `claude/resume-foundation-finetuning-PRT1w`
**Dernière mise à jour** : 2026-04-26
**Statut projet `foundation_finetune`** : ✅ **CLÔTURÉ** (Phase 14, mur documenté)

---

## TL;DR de ce que tu reprends

Le projet `experiments/foundation_finetune/` est **clos**. 14 phases ont rigoureusement démontré que **les indicateurs OHLC-derived (RSI/MACD/CCI) ne suffisent pas** pour la prédiction directionnelle crypto 5min, quel que soit le modèle (Chronos LoRA, XGBoost, Logistic), la cible (Kalman indicateur ou Kalman close), ou la formulation (régression Pearson ou méta-labeling Triple Barrier).

**Le mur est dans les données, pas dans l'architecture.**

Voir `experiments/foundation_finetune/README.md` Phase 14 pour le détail empirique de la dernière clôture.

---

## Findings empiriques exploitables pour la suite

| Finding | Référence | Implication |
|---|---|---|
| RSI/CCI/MACD = 3 projections du même signal latent (Pearson 0.86 inter-modèles) | Phase 2.13, Phase 14 | Pas la peine d'en mettre 3, 1 suffit |
| ATR + volume_spike + vc_score >> yhat indicateurs (XGBoost gain 212 vs 35) | Phase 14 | **Volume/volatilité = vrais signaux exploitables** |
| best_lag = +1 systémique = plafond autocorr 0.93 | Phase 1-9, 2.10, 14 | Pearson causal plafonne ~0.6-0.7 quel que soit le modèle |
| Précision méta-labeling sur indicateurs plafonne 30-44% top 1% | Phase 2.18, 14 | **Inaccessible pour trading sans changement structurel** |
| RANGE_LOW_VOL (71% du test crypto récent) non-tradable | Phase 14 | 71% des données diluent les métriques |
| FLKS sub-step + skip connection = winner pour reconstruction Kalman | Phase 13 | Setup gardé en référence si on revient sur tâche similaire |

---

## Direction recommandée pour la prochaine session

### **Option A — Données vraiment orthogonales (priorité haute)**

C'est la **SEULE direction non encore testée** par le projet. Les 5 confirmations du mur viennent toutes de l'utilisation de signaux **dérivés du close 5min** (RSI/MACD/CCI, volume spike, ATR — ces 2 derniers étant partiellement dérivés). Pour casser le mur, il faut des signaux **vraiment indépendants** :

| Signal | Source | Difficulté d'accès | Pourquoi orthogonal |
|---|---|---|---|
| **Funding rate** Binance perpétuels | API Binance gratuite | ★ facile | Reflète positionnement net longs/shorts (pas dans OHLCV) |
| **Open Interest** | API Binance/Bybit | ★ facile | Volume notional ouvert (pas dans OHLCV) |
| **Liquidations** (long/short) | Coinglass / Hyperdash | ★★ moyen | Évènements forcés, pas d'overlap OHLC |
| **OBV** (On-Balance Volume) | Calculable depuis volume | ★ trivial | Cumul volume signé, partiellement orthogonal |
| **Bid-Ask Spread** | Données tick-level Binance | ★★★ difficile | Microstructure pure |
| **Order Book Depth** | Snapshots WebSocket | ★★★ difficile | Pression d'achat/vente directe |
| **Sentiment social** | LunarCrush / Santiment | ★★ moyen + payant | Externe au marché |

**Recommandation pragmatique** : commencer par **funding rate + OBV** (gratuits, faciles). Setup proposé :
- Récupérer funding rate BTCUSDT-PERP (8h granularité, interpoler à 5min)
- Calculer OBV depuis le volume du CSV existant
- Refaire un mini-pipeline meta-labeling avec uniquement ces features (PAS les indicateurs)
- Test : ces 2 features seules battent-elles les 22 features de Phase 14 ?

Si OUI → on a une vraie direction.
Si NON → confirme que crypto 5min n'a pas l'alpha exploitable dans ces signaux gratuits.

### Option B — Changement de timeframe (15min, 30min, 1h)

Le projet a tout fait en 5min. Les indicateurs sont peut-être trop bruyants à cette résolution. Tester les mêmes setups en 30min ou 1h pourrait :
- Réduire le bruit microstructurel
- Augmenter le signal-to-noise
- Naturellement réduire le nombre de trades

**Mais** : ne casse pas la redondance fondamentale RSI/MACD/CCI. Probablement plafond à un Pearson légèrement supérieur, mais même mur structurel.

**Coût** : faible (rebuild datasets + relance training). **Gain attendu** : marginal mais réel.

### Option C — Pivot vers détection de régime + stratégie conditionnelle

Au lieu de prédire direction/transition, classifier le **régime de marché** et avoir des stratégies différentes par régime. Le projet a déjà un classifieur régime (CNN-LSTM, 83% accuracy 2026-01-14, voir CLAUDE.md). On peut :
- Construire une stratégie qui ne trade qu'en TREND (où la précision est meilleure : 28%)
- Backtester avec maker fees + frais réalistes
- Voir si même un signal faible peut être profitable avec bonne sélection de timing

**Pas un nouveau modèle** : exploitation des findings existants.

### Option D — Backtest réaliste du signal actuel

Le projet n'a JAMAIS backtesté de manière rigoureuse :
- Maker fees (0.02% Binance) au lieu de taker (0.04%)
- Slippage réaliste
- Position sizing dynamique par confiance
- Stop dynamique adaptatif (pas TP/SL fixes)

Avec Phase 14 top 10% à precision 25%, est-ce qu'avec une excellente exécution ça peut être positif ? **Inconnu** car jamais simulé proprement.

---

## Architecture du repo (lecture seule absolue sur certains modules)

```
trad/
├── src/                              ← LECTURE SEULE (indicateurs, filtres, data utils)
├── experiments/
│   ├── slope_improvement/            ← LECTURE SEULE (projet AQ-KF clos)
│   └── foundation_finetune/          ← LECTURE SEULE (clos Phase 14)
│       ├── README.md                 ← Référence complète des 14 phases
│       └── ...
└── data_trad/                        ← Données CSV BTCUSD_all_5m.csv etc.
```

**Tout nouveau projet** : créer `experiments/<nom_nouveau>/` avec son propre README.

---

## Données disponibles

| Asset | Fichier | Période | Granularité |
|---|---|---|---|
| BTC | `data_trad/BTCUSD_all_5m.csv` | 2017-08 → 2026-01 | 5min |
| ETH | `data_trad/ETHUSD_all_5m.csv` | idem | 5min |
| BNB | `data_trad/BNBUSD_all_5m.csv` | idem | 5min |
| ADA | `data_trad/ADAUSD_all_5m.csv` | idem | 5min |
| LTC | `data_trad/LTCUSD_all_5m.csv` | idem | 5min |

Datasets Foundation Finetune (laissés en place pour référence) :
- `data/foundation/{rsi,macd,cci}_btc_close_kalman_5min.npz` (~26 MB chacun)
- `data/foundation/meta_btc_close_kalman.npz` (~50 MB, 22 features + Triple Barrier labels)

Modèles entraînés (laissés en place) :
- `models/specialist_{rsi,macd,cci}/chronos-t5-tiny_lora.pt`
- `models/meta_classifier/{logistic_regression.pkl,xgboost.json}`

---

## Question d'amorçage pour la nouvelle session

Avant de coder quoi que ce soit, l'utilisateur doit choisir parmi :

1. **(A) Funding rate + OBV** : ajouter données vraiment orthogonales et tester si elles débloquent la précision méta-labeling
2. **(B) Timeframe 30min/1h** : refaire le pipeline existant à granularité plus large
3. **(C) Régime conditionnel** : backtester avec stratégie TREND-only
4. **(D) Backtest réaliste** : simulation propre du signal Phase 14 avec maker fees + slippage
5. **(E) Autre direction** : que l'utilisateur précise

**Ne pas commencer à coder avant que l'utilisateur ait tranché.**

---

## Conventions à respecter (héritées du projet)

1. **Per-asset processing** pour calcul indicateurs (CLAUDE.md règle stricte)
2. **Anti-leakage per-split** pour cibles non-causales (Kalman RTS, etc.)
3. **Méthodologie Logistic baseline OBLIGATOIRE** avant XGBoost/MLP/PatchTST (Phase 2.17)
4. **Stratification par régime** dans toutes les analyses (CNN-LSTM régime classifier disponible)
5. **Walk-forward validation** ou purge/embargo si chevauchement temporel des labels
6. **Lecture seule absolue** sur `src/` et `experiments/{slope_improvement,foundation_finetune}/`
7. **Conventions commit** : message clair + `https://claude.ai/code/session_<id>` à la fin
8. **Stop hook** : commit + push à chaque ajout de fichier

---

## Si l'utilisateur reste indécis

Recommandation par défaut : **Option A — Funding rate + OBV**.

Raison : c'est la **SEULE des 4 options qui peut RÉELLEMENT casser le mur des 5 confirmations**. Les options B, C, D ne font qu'exploiter ou contourner le signal existant — elles ne ramènent pas de nouvelle information dans le système.

Si A ne casse pas le mur (precision toujours plafonnée à 30-40% top 1%), alors le verdict définitif est : **les données crypto disponibles publiquement (OHLCV + funding + OI + OBV) ne contiennent pas l'alpha pour cette tâche**. Et là il faudra accepter, soit pivoter vers exécution (option D) soit changer fondamentalement de problème (autre asset class, autre horizon, autre KPI).

---

## Contexte cumulé du projet (pour brief expert)

Le projet est mature : 14 phases sur `foundation_finetune` + multiples phases en amont (1-9, 2.x avec sous-phases jusqu'à 2.18). **Aucune des architectures** (CNN-LSTM, Chronos LoRA, XGBoost, Random Forest, Logistic) n'a réussi à dépasser **~44% precision top 1%** sur prédiction directionnelle / transition detection à partir d'OHLC-derived features uniquement.

Citation finale Phase 2.18 (validée par expert finance, toujours valide en Phase 14) :
> *"Le pipeline est scientifiquement correct mais le signal primaire des indicateurs techniques manque d'alpha exploitable. Documenté depuis 20 ans dans la littérature académique (Zohren 2019, Krauss 2017, López de Prado 2018)."*

Le **vrai débat** pour la suite : faut-il chercher l'alpha dans des **données nouvelles** (orthogonales) ou accepter que l'alpha disponible publiquement n'est pas suffisant pour cette stratégie et **pivoter vers une autre stratégie** (stat arb, market making, mean reversion court-terme avec exécution maker, etc.) ?
