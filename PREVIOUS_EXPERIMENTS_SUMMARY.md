# Résumé des expérimentations trading crypto - Projet "trad"

**Date de génération** : 2026-04-13
**Objectif** : Fournir un état des lieux factuel à un assistant IA travaillant sur une approche LLM

---

## Vue d'ensemble

- **Projet unique identifié** : `trad` (aucun autre projet trading détecté sur le filesystem ni sur le compte GitHub `ecolealgerienne-ui`)
- **Période active** : 2026-01-01 → 2026-01-14 (14 jours d'activité intense)
- **Volume** : 513 commits, 128 fichiers Python (62 src + 66 tests), ~61,500 lignes de code, ~23,300 lignes de documentation
- **Auteurs** : 490 commits par Claude (assistant IA), 23 commits par l'utilisateur (merges PR)
- **Progression** : Montée en complexité très rapide, du pipeline basique au meta-labeling académique en 14 jours

---

## Données utilisées

- **Source** : Fichiers CSV locaux (pré-téléchargés, pas d'API live)
- **Symbols** : BTC, ETH, BNB, ADA, LTC (5 crypto majeurs, paires USD)
- **Timeframe de base** : 5 minutes (récupération des données)
- **Horizon de prédiction** : 30 minutes (labels = pente indicateurs 30min, estimée à chaque bougie 5min avec un Step Index indiquant la position dans la bougie 30min en cours)
- **Timeframe 15min** : également testé
- **Période couverte** : ~8.5 ans de données BTC (2017-08 → 2026-01), ~4.3M séquences sur 5 assets
- **Splits** : 70% train / 15% val / 15% test (split temporel strict, test = données les plus récentes)

---

## Stack technique

| Composant | Technologies |
|-----------|-------------|
| **Deep Learning** | PyTorch (CNN-LSTM custom) |
| **ML classique** | scikit-learn (Logistic Regression), XGBoost, Random Forest |
| **Indicateurs** | Bibliothèque `ta`, calculs custom (RSI, MACD, CCI) |
| **Filtres signal** | Kalman (pykalman), Butterworth/Octave (scipy), Decycler (Ehlers) |
| **Données** | numpy, pandas, fichiers .npz compressés |
| **Arbre de décision** | CART pour apprentissage de règles |
| **GPU** | CUDA (entraînement sur GPU utilisateur) |

---

## Architecture du modèle principal

**CNN-LSTM Multi-Output** :
```
Input: (batch, 25, features) — 25 timesteps × 1-3 features selon indicateur
  → CNN 1D (64-96 filtres, kernel=3)
  → LayerNorm (conditionnel selon indicateur)
  → LSTM (64 hidden, 2 couches, dropout 0.2-0.35)
  → Shortcut connection (optionnel, pour CCI uniquement)
  → Dense partagé (32-64 neurones)
  → Têtes de sortie indépendantes
Output: probabilités binaires (direction UP/DOWN)
```

**Particularités** :
- Configuration hybride par indicateur (LayerNorm activé pour MACD, désactivé pour CCI/RSI)
- BCEWithLogitsLoss pour MACD/CCI, BCELoss standard pour RSI
- Features purifiées : RSI/MACD reçoivent seulement `c_ret` (1 feature), CCI reçoit `h_ret, l_ret, c_ret` (3 features)

---

## Chronologie des phases et résultats

### Jour 1 (01 janv.) — Pipeline de base

- **Approche** : CNN-LSTM multi-output, 3 indicateurs (RSI, CCI, MACD), labels = pente filtrée
- **Labels** : `filtered[t-2] > filtered[t-3]` (direction de la pente passée du signal filtré)
- **Résultat initial** : ~78-80% accuracy
- **Bugs critiques corrigés** : look-ahead bias (trades à t au lieu de t+1), data leakage dans splits

### Jours 2-3 (02-03 janv.) — Optimisations architecture

- **Clock-Injected** : Données récupérées en 5min, indicateurs calculés aussi en 30min, Step Index (position 1-6 dans la bougie 30min en cours), labels = pente 30min. 7 features au total (3 indicateurs 5min + 3 indicateurs 30min + step index). Résultat : **85.1% accuracy** (vs 83.3% baseline de cette phase, soit +1.8% — chiffre obsolète, dépassé ensuite par Dual-Binary à 92.4%)
- **Multi-View Learning** (synchroniser features avec cible) : **-0.7%** → abandonné
- **Single-output** (1 modèle par indicateur) : gain négligeable
- **Bollinger Bands** : retiré (impossible à synchroniser, toujours lag +1)

### Jour 4 (04 janv.) — Premiers backtests et state machine

- **Backtest réel** : Modèle prédit bien (~85%) mais génère trop de micro-trades
- **Bug double sigmoid** corrigé (prédictions écrasées vers 0.5)
- **State machine** : 6 signaux (3 ML × 2 filtres), mode STRICT
- **CART** : Découverte que la volatilité décide SI on agit (100% importance), ML décide la direction
- **Hysteresis** implémentée : réduction -73% trades (seuils 0.4/0.6)
- **Résultat state machine** : PnL brut +1,305%, mais 67,893 trades × 0.2% frais = PnL net **-12,231%**

### Jour 5 (05 janv.) — Architecture Dual-Binary et purification

- **Purification inputs** : Retirer features non causalement liées (H/L pour RSI/MACD) → réduction 60% bruit
- **Dual-Binary** (Direction + Force/Vélocité) : 2 outputs par indicateur
- **Résultats finaux Dual-Binary** :
  - MACD : 92.4% Direction, 81.5% Force
  - CCI : 89.3% Direction, 77.5% Force
  - RSI : 87.4% Direction, 74.0% Force
- **Validation experts** : Architecture hybride LayerNorm/BCEWithLogitsLoss par indicateur

### Jour 6 (06 janv.) — Data Audit et validation experts

- **Walk-forward analysis** sur 83 périodes : patterns stables à 100%
- **2 experts indépendants** valident l'approche (niveau "recherche académique")
- **Découverte** : Relabeling > Suppression pour les données "pièges"
- **Stacking/Ensemble** : 0/9 tests réussis → abandonné (indicateurs = projections du même signal latent)

### Jour 7 (07 janv.) — Consensus et filtres croisés

- **Consensus Direction 4/6** : seul sweet spot fonctionnel (+6,983% Oracle)
- **Force seule** : Win Rate 15-21% → catastrophique, signal non directionnel
- **Confidence Veto Rules** : -3.9% trades seulement → insuffisant
- **Bug direction flip** : créait 2 trades au lieu de 1 → corrigé
- **Résultat Holding 30p** : PnL brut **+110.89%** (signal fonctionne !), mais PnL net **-2,976%** (frais)

### Jour 8 (08 janv.) — Direction-Only et filtres ATR

- **Direction-Only** (retrait de Force) : stable ou amélioré sur tous indicateurs
- **Kalman > Octave** confirmé systématiquement (+1.1% à +4.0%)
- **ATR Structural** : réduit 50% trades mais Win Rate se dégrade → échec
- **ATR ML-Aware** : -0.4% trades seulement → échec (98% des trades = direction flips)
- **Diagnostic transitions** : Le modèle rate **42% des retournements** (58% transition accuracy vs 92.5% global)

### Jour 9 (09 janv.) — Shortcut, fusion, indépendance

- **Shortcut Last-N Steps** : +6% accuracy pour CCI uniquement (multi-features)
- **Weighted Probability Fusion** : 0/12 configurations améliorent le baseline → abandonné
- **Indépendance indicateurs** : Corrélation Oracle **1.000** entre RSI/CCI/MACD → même signal latent
- **Preuve** : 80.6% recouvrement des erreurs, 14.3% complémentarité seulement

### Jour 10 (10 janv.) — Changement de formule et Meta-Labeling

- **Nouvelle formule labels** : `filtered[t] > filtered[t-1]` (signal immédiat vs retardé)
- **Résultat Oracle** : Win Rate 53-57% (vs 33% ancien), PnL net **+14k à +23k%**
- **Trade-off accepté** : Accuracy baisse (92% → 81%) mais Win Rate +20-24%
- **Meta-Labeling Phase 2.17** (López de Prado) :
  - Logistic Regression : Precision 68.41%, ROC AUC 0.5846
  - Découverte majeure : `confidence_spread` coefficient 10× supérieur aux autres features
- **Backtest Meta-Model** : Win Rate **diminue** au lieu d'augmenter → Triple Barrier labels ≠ backtest réel

### Jour 11 (11 janv.) — Aligned Labels et 3 modèles meta

- **Aligned Labels** : Labels recalculés selon la vraie stratégie (signal reversal)
- **3 modèles testés** :
  - Logistic Regression : +24.62% PnL net (15 mois), ~20% annualisé
  - XGBoost : +24.62% PnL net, identique
  - Random Forest : **+28.65% PnL net**, ~23% annualisé (meilleur, 94 trades)
- **Validation experte** : "Pipeline scientifiquement correct, mais signal primaire insuffisant"

### Jours 12-14 (12-14 janv.) — Classificateur de régimes

- **3 classes** : RANGE_LOW_VOL, RANGE_HIGH_VOL, TREND
- **XGBoost avec 20 features** : 98.95% accuracy → **DATA LEAKAGE confirmé** (features = formule des labels)
- **CNN-LSTM avec raw returns** : **86.33%** accuracy (seul résultat valide)
- **Dernier commit** : 14 janvier 2026, puis **3 mois d'inactivité** jusqu'à aujourd'hui

---

## Synthèse des métriques clés

### Accuracy ML (test set)

| Phase | Modèle | Indicateur | Accuracy | Notes |
|-------|--------|------------|----------|-------|
| Baseline | CNN-LSTM 3 feat | Moyenne | 83.3% | Premier modèle |
| Clock-Injected | CNN-LSTM 7 feat | Moyenne | 85.1% | Données 5min + indicateurs 30min + step index |
| Dual-Binary | CNN-LSTM purified | MACD | 92.4% | Meilleur accuracy |
| Direction-Only (t vs t-1) | CNN-LSTM | MACD | 81.1% | Nouvelle formule |
| Regime Classifier | CNN-LSTM raw | 3 classes | 86.3% | Valide (pas leakage) |

### Performance trading (test set, ~15 mois)

| Phase | Stratégie | Trades | Win Rate | PnL Net |
|-------|-----------|--------|----------|---------|
| Oracle ancien (t-2 vs t-3) | Labels parfaits | 68,924 | 33.4% | **-4,116%** |
| Oracle nouveau (t vs t-1) | Labels parfaits | 68,924 | **53.4%** | **+14,359%** |
| ML Entry + Oracle Exit | Grid search | ~13,444 | 22.1% | **-2,082%** |
| Meta-Model (Random Forest) | Aligned, threshold 0.9 | 94 | 45.7% | **+28.65%** |
| Holding 30p | ML + durée min | 30,876 | 29.6% | **-2,976%** (brut +110%) |

---

## Approches testées et abandonnées (liste exhaustive)

| # | Approche | Résultat | Raison d'abandon |
|---|----------|----------|------------------|
| 1 | Multi-View Learning | -0.7% | Synchroniser features réduit diversité |
| 2 | Bollinger Bands | Lag +1 | Impossible à synchroniser |
| 3 | Stacking/Ensemble (9 configs) | 0/9 réussis | Indicateurs = même signal latent |
| 4 | Force comme filtre (STRONG/WEAK) | -354% à -800% | Force non corrélée avec profitabilité |
| 5 | Confidence Veto Rules | -3.9% trades | Insuffisant |
| 6 | ATR Structural Filter | WR dégradé | Filtre quantité, pas qualité |
| 7 | ATR ML-Aware Filter | -0.4% trades | 98% trades = direction flips |
| 8 | Kalman Sliding Window | -19% à -30% | Lag massif détruit signal |
| 9 | Octave Sliding Window | -37% à -116% | Pire que Kalman, overtrading |
| 10 | Weighted Probability Fusion | 0/12 configs | Amplifie le bruit |
| 11 | Transition-Only mode | -749% | Détruit les continuations |
| 12 | Weighted Transition Loss | -6.5% | Dégradation transition accuracy |
| 13 | Triple Barrier Meta-Labels | WR diminue | Mismatch labels vs stratégie réelle |
| 14 | XGBoost Regime (20 features) | 98.95% | Data leakage confirmé |
| 15 | ML Entry + Oracle Exit (5 assets) | 60% négatifs | Signal primaire insuffisant |
| 16 | Dual-Binary (Direction+Force) | Force inutile | Abandonné pour Direction-Only |

---

## Problème fondamental identifié

**Le gap entre accuracy ML et performance trading n'a jamais été comblé.**

```
Accuracy ML labels :  81-92% (excellent)
Win Rate trading :    22-34% (catastrophique)
Gap :                 -49 à -70%

Cause identifiée : Le modèle rate 42% des retournements (transitions).
  - Bon en continuation (90% du dataset) → accuracy globale haute
  - Mauvais en transition (10% du dataset) → Win Rate effondré
  - En trading, seules les transitions comptent (entrées/sorties)
```

**Le seul résultat positif** : Meta-model Random Forest (threshold 0.9) avec +28.65% sur 15 mois (~23% annualisé), mais sur seulement 94 trades → fragile statistiquement et jugé "insuffisant pour crypto" par l'expert (~100-300% Buy & Hold).

**Verdict final de l'expert finance quantitative** :
> "Le problème n'est PAS l'implémentation du meta-labeling (qui est correcte). Le problème est que la prédiction directionnelle à partir d'indicateurs techniques n'a pas d'edge exploitable. C'est documenté depuis 20 ans."

---

## Ce qui n'a PAS été essayé

| Approche | Statut | Notes |
|----------|--------|-------|
| **LLM / approches génératives** | Absent | Jamais exploré dans ce projet |
| **Sentiment analysis** (news, social) | Absent | Aucune donnée textuelle utilisée |
| **Order book / carnet d'ordres** | Absent | Mentionné comme piste (Cartea 2015) mais jamais implémenté |
| **Cross-exchange arbitrage** | Absent | Pas dans le scope du projet |
| **Funding rates / données dérivés** | Absent | Mentionné comme feature potentielle, jamais utilisé |
| **Reinforcement Learning** | Absent | Pas de gym/stable_baselines dans les imports |
| **Transformer / Attention** | Absent | Architecture restée CNN-LSTM uniquement |
| **Données tick-by-tick** | Absent | Mentionné comme option (microstructure) |
| **Multi-timeframe ensemble** | Partiellement | 5min avec estimation 30min testée (Clock-Injected), 15min aussi testé |
| **Volume / OBV comme signal** | Absent | Mentionné comme piste, jamais implémenté comme feature ML |
| **Returns forecasting (régression)** | Absent | Recommandé par expert (Gu, Kelly & Xiu 2020), pas implémenté |

---

## Contexte pour l'assistant IA

L'utilisateur est un développeur **expérimenté en programmation** (capable de gérer un projet 61k lignes avec GPU, PyTorch, pipeline complet) mais **en apprentissage en finance quantitative** (progression rapide de zéro à meta-labeling López de Prado en 14 jours, avec validation par 2 experts académiques).

Le **sérieux méthodologique est élevé** : split temporel strict, vérification causality, data audit walk-forward sur 83 périodes, validation experte multiple, documentation exhaustive (~23k lignes). Chaque approche a été testée rigoureusement avant abandon.

Le **point aveugle principal** est la dépendance exclusive aux indicateurs techniques dérivés du prix (RSI, MACD, CCI), qui sont mathématiquement prouvés comme trois projections du même signal latent 1D (corrélation 1.000). Le projet n'a jamais incorporé de sources d'information véritablement indépendantes (sentiment, order flow, données on-chain, données macro).

Les **forces à exploiter** : infrastructure de données solide (5 assets, 8.5 ans, pipeline reproductible), rigueur expérimentale, connaissance approfondie des pièges (data leakage, look-ahead bias, frais destructeurs), et capacité à itérer rapidement.
