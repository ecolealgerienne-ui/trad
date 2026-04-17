# Diagnostics — XGBoost Progressive (MACD / RSI / CCI × 30m)

Document de synthèse des mesures réalisées sur le pipeline progressif.
Objectif : documenter toutes les mesures (oracle, modèle, filtres,
comparaison inter-indicateurs) pour référence future.

---

## 1. Setup

- **Données source** : `data_trad/BTCUSD_all_5m.csv` (880k rows, 2017-08 → 2026-01)
- **Indicateur** : MACD (12/26/9 défaut)
- **TF** : 30 min (6 sous-pas 5min par bougie)
- **Labels** : `sign(oracle.slope[t_ref])` avec oracle = pykalman RTS smoother
  non-causal sur closes 30m (pas de sous-pas), ffill sur les 6 rows 5min
- **Features** : `[slope_progressive, step_k]` (2 features tabulaires)
  - `slope_progressive` : FLKS slope au step_k courant (slope_t1 à k=0,
    slope_k1..k5 sinon), z-scorée sur train
  - `step_k` : indice sous-pas ∈ {0..5}, brut
- **Splits chronologiques** :
  - train : 614,957 rows (2017-08 → 2023-06, 2141 jours)
  - val   : 131,776 rows (2023-06 → 2024-09, 458 jours)
  - test  : 131,777 rows (2024-09 → 2025-12, 458 jours)
- **Fees** : 0.001 par côté (0.1% entry + 0.1% exit = 0.2% round-trip)
- **Exec** : `close_5m[i+1]` (lag 1 tick 5min, règle "pas de trade à xx:00/xx:30")

---

## 2. Oracle — Borne supérieure (référence)

Signal = `y_{split}_continuous` (ffill de `oracle.slope[t_ref]`).

| Split | Jours | Trades | Tr/j | WR | PF | Sharpe | PnL Brut | Fees | PnL Net | B&H | Alpha | PnL/an |
|-------|-------|--------|------|-----|-----|--------|----------|------|---------|-----|-------|--------|
| train | 2141 | 10,805 | 5.05 | 49.0% | 2.02 | 0.202 | +6,016% | 2161% | **+3,856%** | +655% | +3,200% | ~+657%/an |
| val | 458 | 2,339 | 5.11 | 45.1% | 1.63 | 0.148 | +838% | 467% | **+371%** | +112% | +259% | ~+296%/an |
| test | 458 | 2,281 | 4.98 | 45.6% | 1.58 | 0.146 | +807% | 456% | **+351%** | +36% | +315% | ~+280%/an |

**Remarque** : fees absorbent ~365%/an (5 trades/j × 0.2% × 365). PnL brut
oracle ≈ 645-1020%/an, PnL net ≈ 280-657%/an.

### Amélioration FLKS vs Forward naïf (mesurée sur 180j)

| Variante | Concordance | Gain total | Gain marginal |
|----------|-------------|------------|---------------|
| Forward naïf | 87.94% | — | — |
| slope_t1 (backward pur, k=0) | 90.48% | +2.54% | +2.54% |
| slope_k1 (1 sous-pas) | 93.65% | +5.71% | **+3.17%** 🔥 |
| slope_k2 | 94.49% | +6.55% | +0.84% |
| slope_k3 | 94.93% | +6.99% | +0.44% |
| slope_k4 | 95.40% | +7.46% | +0.47% |
| slope_k5 | 95.59% | +7.65% | +0.19% |

**Rendements décroissants** : le 1er sous-pas apporte +3.17% (plus que tout
le backward pur), puis chaque sous-pas suivant apporte moins (divisé par
~17 entre k1 et k5).

---

## 3. Entraînement XGBoost

- Hyperparams : `n_estimators=500`, `max_depth=6`, `lr=0.1`,
  `objective=binary:logistic`, `early_stopping_rounds=20`
- `best_iteration=41` → convergence rapide
- Dataset parfaitement équilibré : UP ratio 49.86-50.46% sur 3 splits

### Métriques classification

| Split | AUC | Acc | F1 | UP ratio |
|-------|-----|-----|-----|----------|
| train | 0.9762 | 93.69% | 0.9367 | 49.94% |
| val | 0.9788 | 93.52% | 0.9356 | 50.46% |
| test | **0.9836** | **93.75%** | 0.9373 | 49.86% |

**Pas d'overfit** : test ≈ val ≥ train (écart ≤ 0.16%).

### Feature importance (gain)

| Feature | Importance |
|---------|------------|
| slope_progressive | **99.03%** |
| step_k | 0.97% |

`step_k` est quasi-ignoré : XGBoost discrimine directement via les valeurs
de slope (distributions différentes selon k).

---

## 4. Backtest Model — threshold 0.5 (pas de filtre)

Signal = `+1 si proba > 0.5 sinon -1`.

| Split | Trades | vs Oracle | WR | PF | Sharpe | PnL Brut | Fees | PnL Net | Capture |
|-------|--------|-----------|-----|-----|--------|----------|------|---------|---------|
| train | 14,559 | +3,754 | 29.9% | 0.65 | -0.132 | +62% | 2912% | **-2,850%** | -74% |
| val | 3,171 | +832 | 26.3% | 0.51 | -0.214 | -2% | 634% | **-636%** | -171% |
| test | 3,107 | +826 | 27.5% | 0.54 | -0.207 | +32% | 621% | **-590%** | -168% |

**Catastrophe** : 93.75% accuracy → PnL Net fortement négatif.

### Diagnostic du paradoxe

- **+826 trades de plus que Oracle sur test** (3107 vs 2281)
- **WR divisé par 1.7** (45.6% → 27.5%)
- **PnL brut divisé par 25** (+807% → +32%)
- **Les 6.25% d'erreurs ciblent les transitions** et créent des micro-flips
  parasites qui mangent fees + ratent les gros mouvements

---

## 5. Distribution des probas (test set)

| Intervalle | Count | % | Barre |
|------------|-------|---|-------|
| [0.00, 0.05) | 59,171 | 44.90% | ██████████████████████ |
| [0.05, 0.10) | 4,863 | 3.69% | █ |
| [0.10, 0.20) | 1,259 | 0.96% | |
| [0.20, 0.30) | 427 | 0.32% | |
| [0.30, 0.40) | 216 | 0.16% | |
| [0.40, 0.50) | 200 | 0.15% | |
| [0.50, 0.60) | 122 | 0.09% | |
| [0.60, 0.70) | 196 | 0.15% | |
| [0.70, 0.80) | 368 | 0.28% | |
| [0.80, 0.90) | 1,340 | 1.02% | |
| [0.90, 0.95) | 5,113 | 3.88% | █ |
| [0.95, 1.00) | 58,502 | 44.39% | ██████████████████████ |

**Probabilités ultra-bimodales** :
- 89.3% aux extrêmes [0, 0.05) + [0.95, 1.0)
- 0.56% dans [0.30, 0.70]
- 3.13% dans [0.10, 0.90]
- 10.70% dans [0.05, 0.95]

Le modèle est **sur-confiant** : il prédit avec certitude même sur ses erreurs.

---

## 6. Grid Hysteresis asymétrique — ÉCHEC

Principe : `proba > high → LONG`, `proba < low → SHORT`, zone morte =
conserve position.

### Résultats test (16 configs, top 5)

| Low | High | DeadZn | Trades | WR | PF | Sharpe | PnL Net | Capture |
|-----|------|--------|--------|-----|-----|--------|---------|---------|
| 0.30 | 0.70 | 0.6% | 2,975 | 28.6% | 0.55 | -0.203 | **-567%** | -161% |
| 0.35 | 0.70 | 0.5% | 2,983 | 28.5% | 0.55 | -0.203 | -568% | -162% |
| 0.30 | 0.65 | 0.5% | 2,993 | 28.5% | 0.55 | -0.204 | -570% | -162% |
| ... toutes identiques à ~1-2% près | | | | | | | | |

**Diagnostic d'échec** :
- Dead zone filtrée = 0.1% à 0.6% (négligeable)
- Le grid hysteresis n'élimine QUE 130 trades sur 3107 (-4%)
- Amélioration PnL Net = +23% (-590% → -567%), insuffisant
- Cause : les probas sont quasi-binaires, 89% aux extrêmes → une zone
  morte [0.3, 0.7] ne rencontre presque rien

**Conclusion** : **hysteresis en probabilité inutile** ici. Le modèle
ne "hésite" pas, il se trompe avec confiance.

---

## 7. Grid Persistence temporelle — ÉCHEC PARTIEL

Principe :
- **CONFIRM** : signe change seulement si N pas consécutifs nouveaux
- **MIN_HOLD** : après un flip, prochain flip bloqué avant +N pas 5min

### Résultats test (20 configs, top 10)

| mHold | confirm | Trades | WR | PF | Sharpe | PnL Brut | PnL Net | Capture | Δ vs baseline |
|-------|---------|--------|-----|-----|--------|----------|---------|---------|----------------|
| 12 (1h) | 6 (30min) | 2,535 | 30.8% | 0.60 | -0.175 | +63% | **-444%** | **-127%** | **+146%** 🥇 |
| 0 | 6 | 2,551 | 30.4% | 0.60 | -0.177 | +59% | -451% | -129% | +139% |
| 3 | 6 | 2,551 | 30.4% | 0.60 | -0.177 | +59% | -451% | -129% | +139% |
| 6 | 6 | 2,551 | 30.4% | 0.60 | -0.177 | +59% | -451% | -129% | +139% |
| 24 (2h) | 6 | 2,439 | 30.8% | 0.59 | -0.182 | +35% | -452% | -129% | +138% |
| 24 | 2 | 2,561 | 31.4% | 0.59 | -0.182 | +40% | -472% | -135% | +118% |
| 24 | 3 | 2,541 | 31.6% | 0.58 | -0.184 | +35% | -473% | -135% | +117% |
| 24 | 1 | 2,607 | 31.6% | 0.58 | -0.187 | +41% | -481% | -137% | +109% |
| 12 | 3 | 2,683 | 30.6% | 0.58 | -0.190 | +30% | -506% | -144% | +84% |
| 12 | 2 | 2,713 | 29.9% | 0.57 | -0.191 | +29% | -514% | -146% | +76% |

### Baselines

| Ref | Trades | WR | PF | PnL Net |
|-----|--------|-----|-----|---------|
| Oracle | 2,281 | 45.6% | 1.58 | +351% |
| Model t=0.5 | 3,107 | 27.5% | 0.54 | -590% |
| Meilleur persistence | 2,535 | 30.8% | 0.60 | **-444%** |

### Diagnostic d'échec partiel

- **Meilleur gain** : +146% PnL Net vs baseline (-590% → -444%)
- **Trades** : 3107 → 2535 (-18%, se rapproche de l'Oracle 2281)
- **WR** : 27.5% → 30.8% (+3.3%)
- **PF** : 0.54 → 0.60 (toujours perdant <1)
- **Capture** : -168% → -127% (toujours négative)

**Patterns** :
- `confirm=6` domine toutes les top configs → nécessité forte de confirmation
- `min_hold` a un effet secondaire (les configs 0/3/6/12 avec confirm=6 sont
  quasi-identiques, car la confirmation suffit déjà à imposer un holding)
- `confirm=1` (aucun filtre temporel) = identique à baseline (cohérent)

**Conclusion** : la persistence temporelle **aide mais ne suffit pas**.
Le modèle ne capture que 5-8% du PnL brut d'Oracle même avec le meilleur
filtre. Le problème est plus profond que le flickering.

---

## 8. Synthèse & pistes d'amélioration

### Observations clés

1. **Accuracy ≠ PnL** : 93.75% accuracy classification → -590% PnL, car les
   erreurs du modèle **ciblent les moments critiques** (transitions).
2. **Modèle sur-confiant** : 89% des probas aux extrêmes, pas d'hésitation
   même sur les erreurs → hysteresis probabiliste impuissante.
3. **Flickering partiellement temporel** : persistence (confirm=6, 30min)
   enlève 18% de trades mais laisse -444% PnL (encore catastrophique).
4. **Rapport marginal** : meilleur modèle capture **~5-8% du PnL brut
   Oracle** (+63% brut vs +807% brut oracle sur test).

### Pistes non encore testées

| # | Piste | Principe | Effort |
|---|-------|----------|--------|
| 1 | **Calibration isotonique** des probas | Sortir de la quasi-bimodalité, rendre les probas interprétables | Faible |
| 2 | **Régularisation XGBoost** | `max_depth=3`, `min_child_weight=20`, `reg_lambda=10` pour moins de sur-confiance | Faible |
| 3 | **Filter step_k** | Ne trader qu'à step_k ≥ 3 (concordance FLKS 94.9%+) | Faible |
| 4 | **Feature engineering** | Ajouter `slope_t1`, `forward_velocity`, `is_transition_proxy`, etc. | Moyen |
| 5 | **Training objectif PnL-aware** | Loss qui pénalise les flips parasites (custom loss) | Élevé |
| 6 | **LSTM/TCN sur séquences** | Capturer dynamique temporelle (12-25 rows) | Élevé |
| 7 | **Classification 3 classes** | UP / DOWN / INCERTAIN (cible ternaire, proba centre ferme par design) | Moyen |
| 8 | **Meta-model filter** | Apprendre "quelles prédictions XGBoost faut-il exécuter" sur val set | Moyen |

### Prochaine étape recommandée

**Combinaison pistes 2 + 3** (effort faible, attaque direct la cause) :
1. Re-entraîner XGBoost avec régularisation forte pour adoucir les probas
2. Filtrer par `step_k ≥ 3` dans le backtest
3. Garder persistence (confirm=6, min_hold=12)

Si capture passe > +20% PnL Oracle → piste viable. Sinon, passer aux
pistes structurelles (6 : LSTM, 7 : 3 classes, 8 : meta-model).

---

## Annexe — Commandes de reproduction

```bash
# 1. Préparation (progressif sur tout l'historique)
python scripts/prepare_progressive_data.py --indicator macd --tf 30

# 2. Oracle backtest (3 splits)
python scripts/backtest_progressive.py --npz data/prepared/dataset_macd_30m_full_progressive.npz --split test

# 3. Entraînement
python scripts/train_progressive.py --npz data/prepared/dataset_macd_30m_full_progressive.npz

# 4. Backtest Model + Oracle comparaison
python scripts/backtest_progressive.py --npz data/prepared/dataset_macd_30m_full_progressive.npz --preds data/prepared/preds_macd_30m_full_progressive.npz --split test

# 5. Grids de filtres
python scripts/grid_hysteresis.py --split test
python scripts/grid_persistence.py --split test

# 6. Amélioration FLKS vs forward
python scripts/validate_flks_improvement.py --indicator macd --tf 30 --days 180
```

## 10. Extension — CNN-LSTM en remplacement de XGBoost

Script : `scripts/train_cnn_lstm_progressive.py` — drop-in replacement
du XGBoost. Pipeline strictement identique :
- Même NPZ d'entrée
- Même format preds NPZ (suffixe `_cnnlstm`)
- Compatible `backtest_progressive.py` et `cross_validation_indicators.py`
- Ne touche à aucun script existant

### 10.1 Architecture

```
Input (batch, window=24, n_features=2)
  → Conv1D(32 filters, kernel=3, padding=same)
  → LayerNorm + ReLU + Dropout(0.3)
  → LSTM(32 hidden × 2 layers)
  → Dense(32 → 1) + BCEWithLogitsLoss
```

Séquences construites via rolling window avec padding début (répète
1ère ligne) pour garder l'alignement avec `dates/closes/y` — pas de
perte de rows.

18,273 paramètres total (vs ~100-400 arbres XGBoost).

### 10.2 Résultats classification CNN-LSTM vs XGBoost

| Indicateur | XGBoost Test | CNN-LSTM Test | Δ Acc |
|------------|--------------|---------------|-------|
| MACD | AUC 0.9836 / Acc 93.75% | **AUC 0.9882 / Acc 94.42%** | **+0.67%** |
| CCI  | AUC 0.9737 / Acc 91.64% | **AUC 0.9798 / Acc 92.49%** | **+0.85%** |
| RSI  | AUC 0.9637 / Acc 89.26% | **AUC 0.9676 / Acc 89.89%** | **+0.63%** |

**CNN-LSTM bat XGBoost en classification sur les 3 indicateurs**
(+0.63 à +0.85% accuracy). Convergence propre, pas d'overfit
(train ≈ val ≈ test).

### 10.3 Résultats backtest CNN-LSTM vs XGBoost

| Indic | XGBoost Trades | CNN-LSTM Trades | Δ Trades | XGBoost PnL Net | CNN-LSTM PnL Net | **Δ PnL** |
|-------|----------------|-----------------|----------|-----------------|------------------|-----------|
| MACD | 3,107 | 4,653 | **+50%** | -590% | **-911%** | **-321%** |
| CCI  | 4,159 | 5,615 | +35% | -780% | **-1,079%** | -299% |
| RSI  | 6,007 | 7,787 | +30% | -1,176% | **-1,519%** | -343% |

| Indic | XGBoost WR | CNN-LSTM WR | Δ WR |
|-------|------------|-------------|------|
| MACD | 27.5% | 20.8% | **-6.7%** |
| CCI  | 24.8% | 20.1% | -4.7% |
| RSI  | 19.6% | 17.2% | -2.4% |

### 10.4 Loi structurelle découverte

**Pattern universel** (3 indicateurs × 2 modèles) :

> Gagner +0.7% d'accuracy classification = +40% de trades = -320% de PnL Net.

**Cause profonde** : la capacité plus fine du CNN-LSTM lui permet de
détecter **correctement** des micro-oscillations du signal qui étaient
"ignorées" par XGBoost. Mais chaque micro-détection → flip de position
→ trade parasite × fees.

**Plus précis ponctuellement = plus bruité décisionnellement**.

Ce n'est **pas un overfit** (val ≈ test sur les 3 modèles). C'est une
**tension intrinsèque** entre :
- Métrique ML pointwise : récompense la justesse sample par sample
- Métrique trading : pénalise l'instabilité décisionnelle temporelle

### 10.5 Conclusion : le problème n'est pas le modèle

Après XGBoost + CNN-LSTM × MACD/CCI/RSI × 3 filtres (threshold, hysteresis,
persistence) × cross-validation 3 modèles, **aucune configuration n'arrive
à capturer une fraction positive du PnL Oracle**.

**Le problème n'est PAS** :
- ❌ le modèle (XGBoost ≈ CNN-LSTM, pattern identique)
- ❌ l'indicateur (MACD ≈ CCI ≈ RSI, échecs proportionnels)
- ❌ les features (slope_progressive + step_k suffisamment riches,
    AUC 0.97+)
- ❌ le filtrage (aucune combinaison hysteresis/persistence ne rend
    le PnL positif)

**Le problème EST** :
- ⚠️ la cible binaire `sign(oracle.slope[t_ref])` à chaque row 5min
- ⚠️ la structure "décision à chaque tick" qui force un flip au moindre
    bruit
- ⚠️ l'alignement "ML-classification → trading binaire" qui crée un
    paradoxe inversé entre les 2 métriques

### 10.6 Pivots stratégiques proposés

| # | Pivot | Mécanisme | Effort |
|---|-------|-----------|--------|
| 1 | **Régression du rendement futur** | Prédire `close[t+N] - close[t]` au lieu de `sign(slope)` → valeur continue, seuil de trade calibrable | Moyen |
| 2 | **3 classes UP / NEUTRE / DOWN** | Labels `tanh(slope/σ)` seuillés → modèle peut dire "je ne sais pas" | Moyen |
| 3 | **Loss PnL-aware** | Loss custom qui pénalise chaque flip (proportionnel aux fees) → modèle optimise PnL directement | Élevé |
| 4 | **Cadence de décision** | Décider toutes les 30min au lieu de chaque 5min → moins de micro-flips par construction | Faible |
| 5 | **Multi-horizon labels** | Prédire direction à 30min, 1h, 2h → trade uniquement si les 3 convergent | Moyen |
| 6 | **Features volume/microstructure** | Ajouter signal VRAIMENT indépendant des slopes Kalman/FLKS | Moyen |

**Recommandation** : **pivot 4 en premier** (effort faible, attaque
directement la cause : cadence trop fine). Puis pivot 2 (3-classes)
pour rendre l'inaction apprenable par le modèle.

### 10.7 Commandes CNN-LSTM reproductibles

```bash
# Train CNN-LSTM pour chaque indicateur
python scripts/train_cnn_lstm_progressive.py --npz data/prepared/dataset_macd_30m_full_progressive.npz
python scripts/train_cnn_lstm_progressive.py --npz data/prepared/dataset_rsi_30m_full_progressive.npz
python scripts/train_cnn_lstm_progressive.py --npz data/prepared/dataset_cci_30m_full_progressive.npz

# Backtest (même script que XGBoost, juste --preds différent)
python scripts/backtest_progressive.py --npz data/prepared/dataset_<ind>_30m_full_progressive.npz --preds data/prepared/preds_<ind>_30m_full_progressive_cnnlstm.npz --split test
```

---

## Fichiers de référence

- `results/oracle_reference.json` : PnL Oracle + Model (3 indicateurs) JSON
- `data/prepared/dataset_{macd,rsi,cci}_30m_full_progressive.npz` : datasets
- `data/prepared/preds_{macd,rsi,cci}_30m_full_progressive.npz` : prédictions
- `models/xgb_progressive_{macd,rsi,cci}_30m_full.pkl` : modèles entraînés

---

## 9. Extension — 3 indicateurs (MACD / RSI / CCI)

Après le diagnostic sur MACD, génération et entraînement des 2 autres
indicateurs (RSI, CCI) avec pipeline strictement identique (mêmes splits,
mêmes dates, même trim 100, mêmes ratios 70/15/15).

### 9.1 Alignement des datasets (prérequis cross-validation)

**Vérification statique** (code) : les 3 datasets partagent nécessairement
les mêmes index car :
- Même `df_5m` source
- Même resample `df_tf` (agnostique de l'indicateur)
- Même `drop_incomplete_last` (agnostique)
- Même trim (100 bougies TF = 600 rows 5min chaque côté)
- Même split ratios (0.70/0.15/0.15)
- `prepare_features_and_labels_progressive` retourne un DataFrame de
  longueur `len(df_5m) - 2*trim_5m` indépendante de l'indicateur
  (NaN de warmup remplacés par 0 via `fillna(0)`, pas droppés)

**Validation empirique** : script `scripts/validate_indicator_alignment.py`
vérifie bit-exactement :
- Métadonnées (tf, ratios, gap, trim)
- dates_train/val/test (ns precision)
- indices_train/val/test (int64)
- closes_train/val/test (float64 exact)
- df_5m_* et df_tf_* (sources)
- Sanity check inverse : X_test et y_test_binary DOIVENT différer

### 9.2 Entraînement XGBoost (mêmes hyperparams que MACD)

| Indicateur | AUC Train | AUC Val | AUC Test | Acc Test | F1 Test | best_iter | Imp. step_k |
|------------|-----------|---------|----------|----------|---------|-----------|-------------|
| **MACD** | 0.9762 | 0.9788 | **0.9836** | **93.75%** | 0.9373 | 41 | 0.97% |
| **CCI** | 0.9715 | 0.9712 | 0.9737 | 91.64% | 0.9170 | 71 | 0.88% |
| **RSI** | 0.9627 | 0.9623 | 0.9637 | 89.26% | 0.8930 | 116 | 0.83% |

**Observations** :
- **Ordre de difficulté** : MACD < CCI < RSI (MACD converge en 41 iter
  contre 116 pour RSI → MACD "plus facile" pour XGBoost)
- **Pas d'overfit** sur aucun : test ≈ val ≈ train pour les 3
- **`step_k` quasi-ignoré partout** (<1%) — pattern commun aux 3
- **Dataset équilibré** : UP ratio 50.27-50.51% sur les 3 splits × 3 indicateurs

### 9.3 Backtest test set (458 jours, fees 0.1%)

| Indicateur | Oracle Trades | Oracle PnL Net | Model Trades | Model PnL Net | Ratio trades | Capture |
|------------|---------------|----------------|--------------|---------------|--------------|---------|
| **MACD** | 2,281 | +351% | 3,107 | **-590%** | ×1.36 | -168% |
| **CCI** | 2,763 | +434% | 4,159 | **-780%** | ×1.50 | -180% |
| **RSI** | 3,261 | **+601%** | 6,007 | **-1,176%** | ×1.84 | -196% |

**Détails Oracle (test set)** :

| Indic | Trades | WR | PF | Sharpe | PnL Brut | Fees | PnL Net | Alpha B&H |
|-------|--------|-----|-----|--------|----------|------|---------|-----------|
| MACD | 2,281 | 45.6% | 1.58 | 0.146 | +807.14% | 456.20% | **+350.94%** | +314.67% |
| CCI | 2,763 | 45.7% | 1.69 | 0.164 | +986.33% | 552.60% | **+433.73%** | +397.45% |
| RSI | 3,261 | 47.0% | 1.96 | 0.199 | +1253.39% | 652.20% | **+601.19%** | +564.91% |

**Détails Model t=0.5 (test set)** :

| Indic | Trades | WR | PF | Sharpe | PnL Brut | Fees | PnL Net | Alpha B&H |
|-------|--------|-----|-----|--------|----------|------|---------|-----------|
| MACD | 3,107 | 27.5% | 0.54 | -0.207 | +31.56% | 621.40% | **-589.84%** | -626.11% |
| CCI | 4,159 | 24.8% | 0.49 | -0.238 | +51.42% | 831.80% | **-780.38%** | -816.65% |
| RSI | 6,007 | 19.6% | 0.40 | -0.305 | +25.14% | 1201.40% | **-1176.26%** | -1212.53% |

### 9.4 Patterns inter-indicateurs

**Paradoxe inversé accuracy ↔ PnL** :
- RSI : Acc **la plus basse** (89.26%) mais Oracle PnL **le plus haut** (+601%)
- MACD : Acc **la plus haute** (93.75%) mais Oracle PnL **le plus bas** (+351%)
- Implication : la nervosité d'un indicateur = plus d'opportunités (meilleur
  Oracle) mais plus difficile à prédire (accuracy plus faible)

**Sur-trading proportionnel à la nervosité de l'oracle** :
- MACD : Model fait ×1.36 les trades d'Oracle (826 trades parasites)
- CCI : ×1.50 (1,396 trades parasites)
- RSI : ×1.84 (**2,746 trades parasites**, catastrophique)
- Plus l'indicateur est nerveux, plus les erreurs du modèle se transforment
  en micro-flips destructeurs

**Win Rate divisé par ~2** chez les 3 :
- Oracle 45.6-47.0% (toujours > hasard)
- Model 19.6-27.5% (toujours < hasard)
- Pattern structurel identique → problème commun, pas spécifique à un indicateur

**Aucun indicateur ne sort gagnant** isolément. Mais profils différents :
- **MACD** = "stable mais peu d'edge" (le plus prévisible, le moins rentable)
- **CCI** = "milieu" (ratio équilibré)
- **RSI** = "edge maximal mais ingérable" (le plus rentable, le plus nerveux)

### 9.5 Cross-validation 3 indicateurs — RÉSULTATS

Script : `scripts/cross_validation_indicators.py`, test set 458 jours.

#### Matrice de corrélation Pearson (probas)

|      | MACD | CCI | RSI |
|------|------|-----|-----|
| MACD | 1.0000 | 0.7184 | 0.6634 |
| CCI  | 0.7184 | 1.0000 | **0.8496** |
| RSI  | 0.6634 | 0.8496 | 1.0000 |

**Moyenne hors diagonale : 0.7438** → modèles **assez similaires**. CCI et
RSI sont le plus proches (0.85) — cohérent : tous deux oscillateurs.
MACD plus indépendant (moyenne 0.69 avec les 2 autres).

#### Accord binaire (sign match)

| Paire | Accord | N |
|-------|--------|---|
| MACD == CCI | 82.94% | 109,294 / 131,777 |
| MACD == RSI | 79.45% | 104,696 / 131,777 |
| CCI == RSI | 86.82% | 114,411 / 131,777 |
| **Unanimité 3/3** | **74.60%** | 98,312 / 131,777 |

#### Taux d'erreur individuel (vs propre oracle)

| Indicateur | Taux erreur | Count |
|------------|-------------|-------|
| MACD | 6.25% | 8,231 |
| CCI | 8.36% | 11,020 |
| RSI | 10.74% | 14,158 |

#### Erreurs conditionnelles : `P(err[b] | err[a])` vs baseline

| Condition | P(err[b] | err[a]) | Baseline P(err[b]) | **Ratio** |
|-----------|---------------------|--------------------|-----------|
| err[cci] \| err[macd] | 24.94% | 8.36% | **2.98×** |
| err[rsi] \| err[macd] | 25.25% | 10.74% | 2.35× |
| err[macd] \| err[cci] | 18.63% | 6.25% | 2.98× |
| err[rsi] \| err[cci] | 39.52% | 10.74% | **3.68×** |
| err[macd] \| err[rsi] | 14.68% | 6.25% | 2.35× |
| err[cci] \| err[rsi] | 30.76% | 8.36% | 3.68× |

**Erreur simultanée des 3 modèles : 0.77%** (1,016 rows)

**Interprétation critique** :
- Ratios 2.35-3.68× → **erreurs corrélées** (quand un modèle se trompe,
  les autres ont 2-4× plus de risque de se tromper aussi)
- Mais erreur 3/3 rare (0.77%) → les 3 se trompent rarement TOUS en même temps
- Donc : erreurs partagées sur **certaines** zones critiques (transitions)
  mais avec des timings légèrement différents

#### Accuracy vs oracle-consensus (majorité des 3 oracles)

| Stratégie | Accuracy |
|-----------|----------|
| MACD | 84.62% |
| CCI | 89.19% |
| RSI | 86.58% |
| **CONSENSUS-MAJ (vote 2/3)** | **91.10%** 🎯 |

**Consensus améliore l'accuracy de +1.91%** vs meilleur individuel (CCI).

#### Backtest comparatif — test set (458 jours)

| Stratégie | Trades | WR | PF | Sharpe | PnL Brut | Fees | **PnL Net** | Capt |
|-----------|--------|-----|-----|--------|----------|------|-------------|------|
| Oracle MACD | 2,281 | 45.6% | 1.58 | 0.146 | +807% | 456% | **+351%** | +76% |
| Oracle CCI | 2,763 | 45.7% | 1.69 | 0.164 | +986% | 553% | **+434%** | +94% |
| Oracle RSI | 3,261 | 47.0% | 1.96 | 0.199 | +1253% | 652% | **+601%** | +130% |
| Model MACD | 3,107 | 27.5% | 0.54 | -0.207 | +32% | 621% | -590% | -128% |
| Model CCI | 4,159 | 24.8% | 0.49 | -0.238 | +51% | 832% | -780% | -169% |
| Model RSI | 6,007 | 19.6% | 0.40 | -0.305 | +25% | 1201% | -1176% | -255% |
| **Consensus Majorité 2/3** | 4,599 | 22.3% | 0.46 | -0.260 | +28% | 920% | **-891%** | -193% |
| **Consensus Unanimité 3/3** | 2,707 | **30.2%** | **0.57** | -0.192 | +38% | 541% | **-503%** | -109% |

Référence "Capt" = moyenne des 3 Oracles PnL Net (+462%).

Unanimité 3/3 : action sur 98,312 / 131,777 rows = **74.60%** du temps.

#### Paradoxe : meilleure accuracy ≠ meilleur PnL

**Consensus Majorité 2/3** :
- ✅ Accuracy +1.91% vs CCI (91.10% > 89.19%)
- ❌ PnL Net -891% (vs -590% MACD seul) → **PIRE de -302%**
- Raison : 4,599 trades (entre CCI 4,159 et RSI 6,007), les erreurs
  mieux réparties créent plus de flips parasites en moyenne

**Consensus Unanimité 3/3** :
- ✅ PnL Net -503% vs MACD -590% → **+87% vs best individuel**
- ✅ Trades réduits à 2,707 (proche Oracle MACD 2,281)
- ✅ WR remonte à 30.2% (vs MACD 27.5%)
- ❌ Toujours très négatif (capture -109% de Oracle moyen)
- Raison : filtre les flips parasites mais **rate les transitions**
  où un modèle a raison avant les autres (25.40% des rows)

### 9.6 Conclusion cross-validation

**Les 3 indicateurs ne sont PAS suffisamment diversifiés** :
- Corrélation probas 0.66-0.85
- Erreurs partagées (ratio 2-4× baseline conditionnelle)
- Les 3 échouent aux **mêmes moments critiques** (transitions de marché)

**Le consensus simple ne sauve pas** :
- Majorité 2/3 : **détruit** (-302% vs best individuel)
- Unanimité 3/3 : **améliore marginalement** (+87%) mais toujours très négatif

**Implications** :
- ❌ Voter classique (majorité) **inadapté** au trading
- ⚠️ Filtrage par unanimité aide mais insuffisant seul
- ❌ Les 3 indicateurs dérivés de prix (closes) ont trop d'information
  commune pour former un ensemble efficace
- ✅ Il faut du **signal VRAIMENT indépendant** (volume, microstructure,
  order flow, données exogènes)

### 9.7 Pistes post-cross-validation

| # | Piste | Principe | Espoir |
|---|-------|----------|--------|
| 1 | **Régularisation XGBoost forte** | `max_depth=3, reg_λ=10, min_child=20` | Probas moins bimodales → hysteresis redevient utile |
| 2 | **Filter step_k ≥ 3** | Trader uniquement quand concordance FLKS ≥ 94.93% | Réduction trades ~50% |
| 3 | **Unanimité 3/3 + persistence** | Combiner filtres (gains additifs ?) | Réduction trades >50% |
| 4 | **Meta-model pondéré (stacking)** | XGBoost level-2 sur probas + step_k | Apprend quand faire confiance à qui |
| 5 | **Features volume/volatilité** | Signal VRAIMENT indépendant | Casser la corrélation des erreurs |
| 6 | **3-classes UP/DOWN/INCERTAIN** | Label ternaire → probas moins bimodales par design | Hysteresis devient faisable |
| 7 | **LSTM/TCN sur séquences 5min** | Capturer dynamique temporelle (25-50 rows) | Mieux détecter les transitions |
| 8 | **Loss PnL-aware** | Pénaliser les flips parasites dans la loss | Attaque direct la cause |

**Recommandation** : **piste 1 + 2** en combo (effort faible, attaques
directes : sur-confiance + instabilité début de bougie) puis, si échec,
pivot vers **piste 5** (volume) ou **piste 7** (LSTM).

### 9.6 Commandes reproductibles

```bash
# Prépa 3 datasets
python scripts/prepare_progressive_data.py --indicator macd --tf 30
python scripts/prepare_progressive_data.py --indicator rsi --tf 30
python scripts/prepare_progressive_data.py --indicator cci --tf 30

# Validation alignement
python scripts/validate_indicator_alignment.py --tf 30 --period full

# Training
python scripts/train_progressive.py --npz data/prepared/dataset_macd_30m_full_progressive.npz
python scripts/train_progressive.py --npz data/prepared/dataset_rsi_30m_full_progressive.npz
python scripts/train_progressive.py --npz data/prepared/dataset_cci_30m_full_progressive.npz

# Backtest individuel (Oracle + Model côte à côte)
python scripts/backtest_progressive.py --npz data/prepared/dataset_<ind>_30m_full_progressive.npz --preds data/prepared/preds_<ind>_30m_full_progressive.npz --split test

# Cross-validation (prochaine étape)
python scripts/cross_validation_indicators.py --split test
```
