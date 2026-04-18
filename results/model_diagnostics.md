# Diagnostics — Pipeline Progressif MACD / RSI / CCI × 30m

Document de synthèse des mesures réalisées sur le pipeline progressif.
Objectif : documenter toutes les mesures (oracle, modèles XGBoost & CNN-LSTM,
filtres, cross-validation indicateurs, diagnostic Model∩Oracle) pour
référence future.

---

## 📊 Résumé exécutif

| Étape | Résultat clé |
|-------|--------------|
| Oracle reference (3 splits × 3 indic.) | +280 à +657% PnL/an (borne supérieure) |
| Amélioration FLKS vs forward | +7.65% concordance (step_k=5) |
| XGBoost classification | AUC 0.96-0.98, Acc 89-94% (pas d'overfit) |
| XGBoost backtest | **-590% à -1,176% PnL Net** (catastrophe) |
| Grid hysteresis probas | **ÉCHEC** (dead zone <1% car probas bimodales 89%) |
| Grid persistence temporelle | **ÉCHEC PARTIEL** (+146% gain, toujours -444%) |
| Cross-validation 3 indicateurs | Corr 0.74, erreurs corrélées 2-4× |
| Consensus Majorité 2/3 | **ÉCHEC** (-302% vs best individuel) |
| Consensus Unanimité 3/3 | +87% PnL mais toujours -503% |
| **CNN-LSTM en remplacement XGBoost** | **+0.7% acc mais -320% PnL** (paradoxe aggravé) |
| **DIAGNOSTIC Model ∩ Oracle** | **RSI +87% POSITIF**, MACD -11%, CCI -25% |
| **Filtre adaptatif AQ-KF** | +23% capture brute XGBoost MACD (44→67%), neutre CNN-LSTM |
| **🏆 Cible `slope_lag=0` (pente récente)** | **Oracle ×2, RSI CNN-LSTM = +443% PnL Net** (record) |
| ❌ Combinaison Adaptive + Lag=0 | Antagoniste : -66 PnL vs Standard+lag=0 (effets redondants) |
| ❌ Filtre externe ATR (architecture 2 étages) | Best -127% sur Model pur (gain +2078% mais pas positif), edge brut/trade neutre |
| 🏆 **Pipeline Meta-Classifier (architecture 2 étages bis)** | **+135.13% PnL Net** mode 'all' (in-sample partiel, AUC test out-of-sample 0.69) |
| ⚠️ **Validation Option B OOB rigoureuse (CONCLUSION)** | **Pattern réel mais marginal** : AUC OOB 0.65, PnL OOB best -787% à 4,245 trades, +50% à 2 trades (artefact). **Pas exploitable en prod actuelle**. |

### 🎯 Loi structurelle découverte

> **Gagner +0.7% accuracy classification = +40% trades = -320% PnL Net**

### 🔍 Décomposition causale (CNN-LSTM)

- **Switch** (flips parasites) : **~85% de la destruction** ← cause dominante
- **Timing/Lag** : ~15% (capture brute 55% uniforme quand signe correct)

### 📐 Formule opérationnelle

> **PnL Net potentiel ≈ 55% × PnL Oracle Net** (si stabilisation switch)

### ✅ Piste validée empiriquement

**Stabiliser le signal** (réduire flips parasites) par ordre de priorité :
1. Cadence 30min (effort faible) — mécaniquement moins de rows = moins de flips
2. 3-classes UP/NEUTRE/DOWN (effort moyen) — inaction apprenable
3. Loss PnL-aware (effort élevé) — optimise PnL directement

### ⚠️ Limite Kalman atteinte (section 12)

L'AQ-KF (Adaptive Q Kalman Filter) confirme un **plafond structurel à
55-73% capture brute** indépendamment du filtre :
- AQ-KF aide XGBoost (capture brute MACD : 44% → 67%)
- AQ-KF n'aide pas CNN-LSTM (déjà à 55%)
- AQ-KF dégrade le PnL pur (sur-trading hors transitions)
- Persistence ne sauve pas l'AQ-KF
- **Kalman exploré, plafond identifié → pivot vers structurel**

### 🏆 Pivot `slope_lag=0` — record absolu (section 13)

Cible `sign(positions[t] - positions[t-1])` (pente récente) au lieu de
`sign(positions[t-1] - positions[t-2])` (pente passée).

**L'Oracle lui-même double en PnL Net** :
- MACD Oracle : +351% → **+784%** (×2.23)
- RSI Oracle : +601% → **+1,208%** (×2.01)

**Model ∩ Oracle — records absolus** :
- **MACD CNN-LSTM lag=0 : +301%** (×28 vs lag=1)
- **RSI CNN-LSTM lag=0 : +443%** 🏆 (×5.1 vs lag=1)

**Règle découverte** :
- XGBoost → Adaptive + lag=1 (+84% RSI)
- CNN-LSTM → Standard + lag=0 (+443% RSI)
- Les 2 pipelines sont **complémentaires**, pas substituables

### ❌ Combinaison Adaptive + Lag=0 — antagoniste (section 14)

Test sur RSI CNN-LSTM (config la plus prometteuse) :
- Adaptive + lag=0 PnL filtré : **+377%** (vs Standard + lag=0 : **+443%**)
- Capture brute : 55% (vs 59% Standard + lag=0)
- **-66 points de PnL** : les 2 modifications ne se cumulent pas

**Cause** : adaptive ajoute du bruit redondant que le LSTM (déjà bon avec
lag=0) n'a pas demandé. Pour chaque modèle, **une seule** modification
améliore — pas les 2.

### ❌ Filtre externe ATR (architecture 2 étages) — échec validé (section 15)

Test sur RSI CNN-LSTM lag=0 — filtre ATR appliqué EN AVAL du modèle (pas
comme feature interne) pour bloquer les signaux dans des conditions
défavorables. 3 grids testés (vol haute / basse / moyenne).

**Résultats** :
- Best (vol haute [0.005, 0.010]) : **-127% PnL Net** (vs Model pur -2,205%)
- Gain massif : +2,078 points, mais reste négatif
- WR amélioré (14.7% → 32.6%) mais PnL Brut/trade reste neutre

**Diagnostic** : le modèle a un edge brut **quasi-nul par trade** (-0.0018%).
Quel que soit le filtre ATR, les fees (0.2%/trade) dominent.

### 🏆 Pipeline Meta-Classifier — PoC validée (section 16)

Architecture en 2 étages : modèle direction (CNN-LSTM) + meta-classifier
(XGBoost) qui apprend à filtrer les **flips parasites**.

**Pipeline** : extract flips → features (12 finales) → 2 XGBoost
(LONG / SHORT spécialisés) → backtest avec meta-filter.

**In-sample partiel mode 'all'** : +135% PnL Net (AUC test 0.69) — biais
memorization confirmé par Option B.

**Apports scientifiques** : architecture 2 étages validée, label
`is_profitable_flip` 3× plus discriminant, ATR top feature, asymétrie
LONG/SHORT confirmée.

### ⚠️ Validation Option B OOB rigoureuse — CONCLUSION FINALE (section 17)

Train meta sur **flips val** (out-of-sample modèle direction), test sur
**flips test** (out-of-sample meta ET modèle direction).

**Résultats classification OOB** :
- LONG : AUC test = **0.6581** ✅ (gap -3.7% vs in-sample, signal réel)
- SHORT : AUC test = **0.6468** ✅ (gap -3.2%, MAIS best_iter=2 = sous-entraîné)

**Backtest OOB pratique** :
- Aux seuils sélectifs (0.65+) : **2 trades sur 458 jours** (artefact, SHORT
  classifier rejette tout)
- Aux seuils raisonnables (0.50) : **-787% PnL Net** (4,245 trades)
- Pas de sweet spot exploitable

**Verdict honnête** :
- ✅ Pattern réel détecté (AUC OOB > 0.65)
- ✅ Architecture techniquement viable
- ❌ **Amplitude trop faible pour battre fees 0.2%/trade** en production
- ❌ Le +135% in-sample précédent était bien de l'overfit (confirmé)

**Pistes pour rendre exploitable** (non testées) :
- Maker fees (0.02% au lieu de 0.1% taker) : ÷10 fees → potentiellement positif
- Catégorie 2-3 features (HMM régime, multi-TF agreement)
- Ré-entrainer SHORT (scale_pos_weight modéré, max_depth=5)
- Architecture deep (Transformer, multi-task)

**Sujet meta-classifier CLOS** dans son état actuel.

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

## 11. Diagnostic "Model ∩ Oracle" — Décomposition Switch vs Timing

Test clé : filtrer les signaux du modèle par accord avec l'Oracle pour
isoler la cause de l'échec PnL.

**Principe** : à chaque row 5min, si `sign(model) == sign(oracle)` alors
on trade, sinon on conserve la position. ⚠️ Non utilisable en prod
(requiert l'oracle en live), c'est un outil de diagnostic.

Scripts :
- `src/signal_processing/core.py::backtest_5min_filtered_by_oracle` (copie
  fidèle de `backtest_5min_progressive` + filtre d'accord oracle)
- `scripts/backtest_model_filtered_by_oracle.py` (diagnostic 3 stratégies)

### 11.1 Résultats CNN-LSTM × 3 indicateurs (test set)

| Indicateur | Oracle Net | Model pur Net | **Model ∩ Oracle Net** | Capture brute | Gain vs Model pur | Trades bloqués |
|------------|------------|---------------|-------------------------|---------------|-------------------|----------------|
| **RSI** | +601.19% | -1,518.86% | **+86.77%** ✅ | 59% | +1,605.63% | 7,788 |
| **MACD** | +350.94% | -910.66% | **-10.79%** ≈0 | 55% | +899.86% | 4,141 |
| **CCI** | +433.73% | -1,078.95% | **-25.13%** ≈0 | 53% | +1,053.82% | 5,301 |

**Observations clés** :
1. **RSI filtré est POSITIF** (+86.77%) — premier PnL Net positif de tout
   le diagnostic
2. Les 3 indicateurs filtrés convergent **proches du break-even**
3. **Capture brute uniforme 53-59%** entre les 3 indicateurs —
   caractéristique intrinsèque de l'alignement model ↔ oracle
4. Même nombre de trades que l'Oracle (2,277-3,239 vs 2,281-3,261)

### 11.2 Comparaison XGBoost vs CNN-LSTM (MACD)

| Stratégie | XGBoost | CNN-LSTM | Interprétation |
|-----------|---------|----------|----------------|
| Model pur PnL Net | -590% | -911% | CNN-LSTM pire en vanilla |
| **Model ∩ Oracle PnL Net** | **-100%** | **-11%** ✅ | CNN-LSTM **9× meilleur** |
| Capture brute | 44% | 55% | CNN-LSTM capture plus |
| Trades bloqués | 2,965 | 4,141 | CNN-LSTM ×1.4 plus d'instabilité |
| Gain vs Model pur | +490% | +900% | CNN-LSTM bénéficie plus du filtre |

**Paradoxe résolu** :
- CNN-LSTM détecte mieux le signal **quand il est d'accord** avec Oracle
  (+55% capture vs +44% XGBoost)
- MAIS CNN-LSTM génère **plus de désaccords** (4,141 vs 2,965 = ×1.4)
- → Le signal intrinsèque est meilleur, mais l'instabilité décisionnelle
  détruit tout

### 11.3 Décomposition Switch vs Timing

**Switch** (flips parasites causés par désaccords model/oracle) :
- Part de la destruction PnL : **~85%** en moyenne sur CNN-LSTM
- Cause : chaque désaccord sign déclenche un flip inutile × fees × 2

**Timing/Lag** (entrées/sorties mal calées même quand signe correct) :
- Part de la destruction PnL : **~15%**
- Cause : même avec le bon signe, Model entre/sort en retard de 1-5 rows
  vs Oracle → manque le meilleur prix → capture brute 55% au lieu de 100%

### 11.4 Formule découverte

> **PnL Net potentiel ≈ 55% × PnL Oracle Net** (si on stabilise les flips)

Pour RSI : 0.55 × +601% = **+330% PnL Net théorique**
Pour MACD : 0.55 × +351% = +193% PnL Net théorique
Pour CCI : 0.55 × +434% = +239% PnL Net théorique

Actuellement atteint (filtrage parfait oracle) :
- RSI : **+87% PnL Net** (26% de l'optimum théorique)
- MACD : -11% (0% atteint)
- CCI : -25% (0% atteint)

Les gaps sont dûs au fait que même filtré, les trades payent fees sur des
transitions retardées. La cadence 30min ou la réduction du nombre de rows
pourrait les réduire.

### 11.5 Conclusion stratégique

**Le switch est le problème DOMINANT (~85%)**. La piste "stabiliser le
signal" est **validée empiriquement** :

- ✅ Si on éliminait 100% des flips parasites → PnL Net positif sur les 3
- ✅ CNN-LSTM > XGBoost pour cette piste (meilleur signal brut intrinsèque)
- ✅ RSI > MACD > CCI en PnL potentiel (Oracle plus élevé)

**Pistes recommandées par ordre de priorité** :

1. **Cadence 30min** (effort faible) : réduit mécaniquement les flips
   (6× moins de rows → 6× moins d'occasions de se tromper)
2. **3-classes UP/NEUTRE/DOWN** (effort moyen) : le modèle peut dire
   "je ne sais pas" dans les zones grises → moins de désaccords forcés
3. **Loss PnL-aware** (effort élevé) : pénalise les flips dans la loss
   → le modèle apprend la stabilité

### 11.6 Commandes

```bash
# Diagnostic 3 indicateurs × 2 modèles
for ind in macd cci rsi; do
  for model in "" "_cnnlstm"; do
    python scripts/backtest_model_filtered_by_oracle.py \
      --npz data/prepared/dataset_${ind}_30m_full_progressive.npz \
      --preds data/prepared/preds_${ind}_30m_full_progressive${model}.npz \
      --split test
  done
done
```

---

## 12. Filtre adaptatif AQ-KF — exploration et conclusion

Test : remplacer le forward filter standard par **AQ-KF** (Adaptive Q
Kalman Filter, `forward_filter_30m_adaptive` dans `core.py`) qui augmente
dynamiquement Q autour des transitions à forte innovation.

**Modification non-breaking** : flag `--adaptive` dans
`prepare_progressive_data.py`, suffixe `_adaptive` dans les NPZ et modèles.
Aucun script existant cassé. Oracle inchangé (toujours RTS smoother).

### 12.1 Impact classification (test set)

| Indicateur × Modèle | Standard AUC | Adaptive AUC | Standard Acc | Adaptive Acc |
|---------------------|--------------|--------------|--------------|--------------|
| MACD XGB | 0.9836 | 0.9632 | 93.75% | **90.01%** (-3.74%) |
| MACD CNN-LSTM | 0.9882 | 0.9859 | 94.42% | 93.86% (-0.56%) |
| RSI XGB | 0.9637 | 0.9055 | 89.26% | **81.98%** (-7.28%) |

**L'adaptive dégrade la classification** :
- XGBoost très impacté (-3.7 à -7.3%)
- CNN-LSTM peu impacté (-0.6%) → il s'adapte au signal plus bruité

**Cause** : AQ-KF rend les slopes plus réactives autour des transitions →
moins prédictibles ponctuellement.

### 12.2 Impact backtest Model pur (sans filtre oracle)

| Indicateur × Modèle | Standard PnL Net | Adaptive PnL Net | Delta |
|---------------------|------------------|------------------|-------|
| MACD XGB | -590% | **-796%** | **-206 pire** |
| MACD CNN-LSTM | -911% | -849% | +62 (∼=) |
| RSI XGB | -1176% | **-1598%** | **-422 pire** |

**Sans filtre oracle, l'adaptive est PIRE** : plus de trades parasites
parce que la réactivité augmente le bruit hors transitions.

### 12.3 Impact backtest Model ∩ Oracle (avec filtre oracle)

| Indicateur × Modèle | Std PnL Net | Adapt PnL Net | Std capture brute | Adapt capture brute |
|---------------------|-------------|---------------|-------------------|---------------------|
| **MACD XGB** | -100% | **+84%** ✅ | 44% | **67%** (+23%) 🔥 |
| MACD CNN-LSTM | -11% | -12% | 55% | 55% (=) |
| **RSI XGB** | (n/a) | **+258%** ✅ | (n/a) | **73%** |
| RSI CNN-LSTM | +87% | (à compléter) | 59% | (à compléter) |

**L'adaptive améliore drastiquement le timing QUAND signe correct** :
- MACD XGBoost : capture brute 44% → **67%** (+52% relatif)
- Premier PnL net positif pour MACD XGBoost (+84%)
- RSI XGBoost : Model ∩ Oracle = +258% (3× le RSI std)

### 12.4 Impact grid persistence sur adaptive

| Best persistence MACD | Standard | Adaptive |
|------------------------|----------|----------|
| min_hold/confirm | 12/6 | 24/6 |
| PnL Net | -444% | **-609%** |

**La persistence ne sauve PAS l'adaptive** : pire qu'en standard (-609 vs
-444). Les transitions plus brèves de l'adaptive mettent en échec les
heuristiques temporelles classiques.

### 12.5 Conclusion sur l'AQ-KF

**Ce que confirme l'AQ-KF** :
- ✅ Les slopes adaptatives capturent **mieux les transitions** (+14% absolu
  capture brute en MACD XGB filtré oracle, +52% relatif)
- ✅ Particulièrement utile pour XGBoost (modèle "rigide" qui bénéficie
  d'un meilleur prétraitement du signal)
- ❌ Impact CNN-LSTM négligeable (modèle déjà bon en feature extraction)
- ❌ Augmente le bruit hors transitions (sur-trading en mode "Model pur")
- ❌ Persistence inadaptée à la dynamique adaptive

**Verdict architectural** :

> **Le choix Standard vs Adaptive est conditionnel au modèle aval** :
> - **XGBoost** + **filtre oracle** (ou équivalent forte stabilisation) → Adaptive
> - **CNN-LSTM** ou **production sans filtre** → Standard

**Insight final sur Kalman** :

Toutes les variations exploré du Kalman (forward standard, forward
adaptatif, FLKS backward 1..5 sous-pas, RTS oracle) montrent un **plafond
structurel à ~55-73% capture brute** quand le signe est correct. Ce
plafond est **caractéristique de la cible binaire ffill** :

- Même avec un filtre parfait sur les features, on ne peut pas synchroniser
  exactement les transitions du modèle avec celles de l'oracle
- Le décalage temporel résiduel (1-5 rows 5min) crée un manque-à-gagner de
  ~30-45% sur les vrais trades
- Aucune variante de Kalman ne dépasse ce plafond

**On a fait le tour de Kalman**. Les pistes restantes sont structurelles :
- **Cible** : régression rendement, multi-horizon, 3-classes
- **Cadence** : 30min vs 5min
- **Loss** : PnL-aware au lieu de BCE
- **Architecture** : transformer attention, multi-task, meta-learning

### 12.6 Commandes reproductibles

```bash
# Génération adaptative (3 indicateurs)
python scripts/prepare_progressive_data.py --indicator macd --tf 30 --adaptive
python scripts/prepare_progressive_data.py --indicator rsi --tf 30 --adaptive
python scripts/prepare_progressive_data.py --indicator cci --tf 30 --adaptive

# Train + diagnostic Model ∩ Oracle
for ind in macd rsi cci; do
  python scripts/train_progressive.py --npz data/prepared/dataset_${ind}_30m_full_progressive_adaptive.npz
  python scripts/train_cnn_lstm_progressive.py --npz data/prepared/dataset_${ind}_30m_full_progressive_adaptive.npz
  python scripts/backtest_model_filtered_by_oracle.py --npz data/prepared/dataset_${ind}_30m_full_progressive_adaptive.npz --preds data/prepared/preds_${ind}_30m_full_progressive_adaptive.npz --split test
done
```

---

## 13. Cible `slope_lag=0` — pente récente vs pente passée

Test : remplacer la pente de l'oracle `positions[t-1] - positions[t-2]`
(legacy) par `positions[t] - positions[t-1]` (plus récente de 30min TF).

**Modification non-breaking** : param `slope_lag=1` (défaut legacy) dans
`compute_oracle`, `compute_oracle_labels`,
`prepare_features_and_labels_progressive`. Flag `--slope-lag {0,1}` dans
`prepare_progressive_data.py`. NPZ suffixe `_lag0` (vide pour défaut).

### 13.1 Impact sur l'Oracle lui-même (× 2)

| | Standard (lag=1) | Lag=0 | Delta |
|---|------------------|-------|-------|
| **MACD Oracle PnL Net** | +351% | **+784%** | **×2.23** 🔥 |
| MACD Oracle WR | 45.6% | 56.6% | +11.0% |
| MACD Oracle PF | 1.58 | 2.99 | +89% |
| MACD Oracle Sharpe | 0.146 | 0.326 | +123% |
| **RSI Oracle PnL Net** | +601% | **+1,208%** | **×2.01** 🔥 |
| RSI Oracle WR | 47.0% | 59.5% | +12.5% |
| RSI Oracle PF | 1.96 | 4.29 | +119% |
| RSI Oracle Sharpe | 0.199 | 0.394 | +98% |

**Découverte structurelle majeure** : la cible `lag=1` était **sous-optimale
de ~100%**. La pente "pente actuelle" capture 2× plus d'alpha que la pente
"pente passée de 30min".

### 13.2 Impact classification

| Modèle × Indic | Lag=1 AUC | Lag=0 AUC | Delta AUC | Lag=1 Acc | Lag=0 Acc | Delta Acc |
|----------------|-----------|-----------|-----------|-----------|-----------|-----------|
| MACD XGBoost | 0.9836 | 0.9201 | -6.35% | 93.75% | 84.83% | **-8.92%** |
| MACD CNN-LSTM | 0.9882 | 0.9646 | -2.36% | 94.42% | 89.90% | -4.52% |
| RSI CNN-LSTM | 0.9676 | 0.9139 | -5.37% | 89.89% | 82.86% | -7.03% |

**XGBoost encaisse plus** (-8.92%) que CNN-LSTM (-4.52%). La cible plus
locale nécessite de la mémoire temporelle que XGBoost n'a pas (modèle
tree-based sans contexte séquentiel).

### 13.3 Impact Model pur (sans filtre oracle)

| Modèle × Indic | Lag=1 Trades | Lag=0 Trades | Lag=1 PnL Net | Lag=0 PnL Net |
|----------------|--------------|--------------|---------------|---------------|
| MACD XGBoost | 3,107 | 3,111 | -590% | -592% |
| MACD CNN-LSTM | 4,653 | 7,531 | -911% | **-1,525%** |
| RSI CNN-LSTM | 7,787 | 10,927 | -1,176% | **-2,205%** |

**Model pur ne bénéficie pas du lag=0** : plus de flips parasites car la
cible plus locale génère plus de transitions à détecter (et à confondre).

### 13.4 Impact Model ∩ Oracle — RÉSULTATS RECORDS

| Modèle × Indic | Lag=1 PnL Net | Lag=0 PnL Net | Delta | Capture brute lag=1 | Capture brute lag=0 |
|----------------|---------------|---------------|-------|---------------------|---------------------|
| MACD XGBoost | -100% | **-132%** ❌ | pire | 44% | **26%** (pire) |
| **MACD CNN-LSTM** | -11% | **+301%** ✅ | **×28** 🏆 | 55% | **61%** |
| **RSI CNN-LSTM** | +87% | **+443%** ✅ | **×5.1** 🏆 | 59% | **59%** (stable) |

**MACD CNN-LSTM `lag=0` Model ∩ Oracle = +301% PnL Net**
**RSI CNN-LSTM `lag=0` Model ∩ Oracle = +443% PnL Net** → **RECORD**

### 13.5 Insight clé — le plafond NE bouge pas mais la base × 2

**Capture brute reste à ~60%** pour CNN-LSTM lag=0 (vs ~55% en lag=1) :
- MACD : 55% → 61% (+6%)
- RSI : 59% → 59% (=)

Le **plafond structurel 55-73%** est peu impacté. Mais comme l'Oracle a
doublé, le **produit `capture × Oracle` double aussi** :

> **Formule lag=1** : PnL Net ≈ 55-60% × PnL Oracle_lag1
> → +193% MACD théorique, +330% RSI théorique
>
> **Formule lag=0** : PnL Net ≈ 60% × PnL Oracle_lag0 (double)
> → **+470% MACD, +724% RSI théorique**
>
> Atteint : +301% MACD (64% du théorique), +443% RSI (61% du théorique)

### 13.6 Pourquoi XGBoost échoue en lag=0

XGBoost est un modèle **stateless pointwise** : chaque sample est classifié
indépendamment à partir de `[slope_progressive, step_k]` actuels.

La cible lag=1 (`sign(positions[t-1] - positions[t-2])`) est **lissée**
(décalage de 1 bougie après l'événement le plus récent) → plus prévisible.

La cible lag=0 (`sign(positions[t] - positions[t-1])`) est **locale** à
l'instant courant → nécessite **la mémoire des rows précédentes** pour
stabiliser la prédiction. Le LSTM dans CNN-LSTM absorbe cette dynamique.

### 13.7 Cartographie "modèle × cible"

| Cible \ Modèle | XGBoost | CNN-LSTM |
|----------------|---------|----------|
| Standard lag=1 | -590% PnL (44% capt.) | -911% PnL (55% capt.) |
| Adaptive lag=1 | -796% PnL (67% capt.) | -849% PnL (55% capt.) |
| **Standard lag=0** | **-592% PnL (26% capt.)** | **-1,525% PnL (61% capt.)** |

| Cible \ Modèle (Model ∩ Oracle) | XGBoost | CNN-LSTM |
|---------------------------------|---------|----------|
| Standard lag=1 | -100% | -11% |
| Adaptive lag=1 | **+84%** ✅ | -12% |
| **Standard lag=0** | -132% ❌ | **+301%** 🏆 |

**Règle empirique découverte** :
- **XGBoost** performe mieux avec **Adaptive + lag=1** (+84%)
- **CNN-LSTM** performe mieux avec **Standard + lag=0** (+301%)
- Les 2 pipelines sont **complémentaires**, pas substituables

### 13.8 Nouveau record absolu du pipeline

**RSI CNN-LSTM lag=0 Model ∩ Oracle = +443% PnL Net** est le **meilleur
résultat filtré** de tout le diagnostic :

| Rang | Config | PnL Net filtré |
|------|--------|----------------|
| 🥇 | **RSI CNN-LSTM lag=0** | **+443%** |
| 🥈 | MACD CNN-LSTM lag=0 | +301% |
| 🥉 | RSI XGBoost adaptive | +258% |
| 4 | MACD XGBoost adaptive | +84% |
| 5 | RSI CNN-LSTM std lag=1 | +87% |

### 13.9 Commandes reproductibles

```bash
# MACD et RSI lag=0 (CCI non testé par décision utilisateur)
for ind in macd rsi; do
  python scripts/prepare_progressive_data.py --indicator ${ind} --tf 30 --slope-lag 0
  python scripts/train_cnn_lstm_progressive.py --npz data/prepared/dataset_${ind}_30m_full_progressive_lag0.npz
  python scripts/backtest_model_filtered_by_oracle.py --npz data/prepared/dataset_${ind}_30m_full_progressive_lag0.npz --preds data/prepared/preds_${ind}_30m_full_progressive_cnnlstm_lag0.npz --split test
done

# MACD XGBoost lag=0 (pour montrer l'échec)
python scripts/train_progressive.py --npz data/prepared/dataset_macd_30m_full_progressive_lag0.npz
python scripts/backtest_model_filtered_by_oracle.py --npz data/prepared/dataset_macd_30m_full_progressive_lag0.npz --preds data/prepared/preds_macd_30m_full_progressive_lag0.npz --split test
```

### 13.10 Pistes ouvertes après lag=0

1. **Combiner `adaptive + lag=0`** : cumuler les gains ? (Oracle +230% × capture adaptive +14%)
2. **Consensus CNN-LSTM lag=0 (MACD + RSI)** : dépasser le plafond 60% ?
3. **Hysteresis/persistence sur CNN-LSTM lag=0** : stabiliser les 10,927 trades RSI pur pour arriver au break-even sans filtre oracle
4. **Investigation CCI lag=0** : mesurer si le pattern tient (hors scope actuel)

---

## 14. Combinaison Adaptive + Lag=0 — antagonisme confirmé

Test : combiner les 2 modifications gagnantes (`--adaptive` + `--slope-lag 0`)
pour vérifier si les effets se cumulent.

**Hypothèse de cumul** : Adaptive aide le timing (capture brute +14% absolu
sur XGBoost MACD) + Lag=0 double l'Oracle (×2.01-2.23) → projection
théorique +500-700% PnL Net filtré pour RSI CNN-LSTM.

**Résultat empirique** : effets **antagonistes**, pas additifs.

### 14.1 Configuration testée

- Indicateur : RSI (le plus rentable, oracle +1,208% en lag=0)
- Modèle : CNN-LSTM (seul modèle qui apprend lag=0)
- Filtre : AQ-KF Adaptive
- Cible : pente t/t-1 (slope_lag=0)
- NPZ : `dataset_rsi_30m_full_progressive_adaptive_lag0.npz` (87 MB)
- Modèle : `cnnlstm_progressive_rsi_30m_full_adaptive_lag0.pth`
- Preds : `preds_rsi_30m_full_progressive_cnnlstm_adaptive_lag0.npz`

### 14.2 Résultats classification

| Métrique | RSI CNN-LSTM std lag=0 | RSI CNN-LSTM adapt lag=0 | Delta |
|----------|------------------------|---------------------------|-------|
| Test AUC | 0.9139 | 0.8936 | -2.03% |
| Test Acc | 82.86% | 80.74% | **-2.12%** |
| Test F1 | 0.8294 | 0.8103 | -1.91% |

L'adaptive dégrade encore la classification quand combiné à lag=0 :
double pénalité (signal plus nerveux + cible plus locale).

### 14.3 Résultats backtest

| Stratégie | Trades | WR | PF | Sharpe | PnL Brut | Fees | PnL Net |
|-----------|--------|-----|-----|--------|----------|------|---------|
| Oracle pur (lag=0) | 3,261 | 59.5% | 4.29 | 0.394 | +1,860% | 652% | **+1,208%** |
| Model pur adapt+lag0 | 9,642 | 16.7% | 0.31 | -0.388 | +27% | 1,928% | -1,901% |
| **Model ∩ Oracle adapt+lag0** | **3,229** | 43.1% | 1.52 | 0.130 | +1,023% | 646% | **+377%** |

### 14.4 Comparaison récapitulative — RSI CNN-LSTM

| Configuration | Acc | Trades pur | PnL pur | **PnL filtré** | Capture brute |
|---------------|-----|------------|---------|----------------|---------------|
| Standard lag=1 | 89.89% | 7,787 | -1,176% | +87% | 59% |
| Adaptive lag=1 | (n/a) | (n/a) | (n/a) | (n/a) | (n/a) |
| **Standard lag=0** ⭐ | 82.86% | 10,927 | -2,205% | **+443%** 🏆 | 59% |
| Adaptive lag=0 | 80.74% | 9,642 | -1,901% | **+377%** | **55%** |

**La combinaison adaptive + lag=0 réduit le PnL Net filtré de -66 points**
(+443% → +377%) et la capture brute de -4 points (59% → 55%).

### 14.5 Pourquoi les effets ne se cumulent pas

**Adaptive** rend les slopes FLKS plus **réactives** près des transitions
(Q dynamique augmente l'amplitude du signal local).

**Lag=0** demande déjà au modèle de prédire une cible plus **locale**
(pente t vs t-1 au lieu de t-1 vs t-2).

**Combinés** :
- Le signal devient trop nerveux (adaptive pousse les slopes)
- La cible est déjà locale (lag=0 demande la pente la plus récente)
- Le modèle CNN-LSTM se trompe plus souvent en accord direction
  → capture brute baisse (59% → 55%)
- L'adaptive ne sert à rien : le LSTM avait déjà absorbé la dynamique
  locale grâce à sa mémoire séquentielle (window=24 rows = 2h)
- L'adaptive ajoute du bruit **redondant** que le LSTM n'a pas demandé

### 14.6 Cartographie finale modèle × cible × filtre

| Modèle | Filtre | Cible | PnL Net Model ∩ Oracle |
|--------|--------|-------|------------------------|
| **XGBoost** | **Adaptive** | **lag=1** ⭐ | +84% (MACD), **+258%** (RSI) |
| XGBoost | Standard | lag=0 | -132% (MACD) ❌ |
| **CNN-LSTM** | **Standard** | **lag=0** ⭐ | +301% (MACD), **+443%** (RSI) 🏆 |
| CNN-LSTM | Adaptive | lag=0 | **+377%** (RSI, pire que std lag=0) |
| CNN-LSTM | Standard | lag=1 | -11% (MACD), +87% (RSI) |

**Règle empirique consolidée** :

> Pour chaque modèle, **une seule** modification améliore les résultats :
> - **XGBoost** : Adaptive (le filtre fait le boulot que le modèle ne sait pas faire)
> - **CNN-LSTM** : Lag=0 (la cible plus locale est exploitée par le LSTM)
>
> **Cumuler les 2 modifications dégrade systématiquement** le PnL filtré.

### 14.7 Verdict final sur les pivots cible/filtre

**Pistes Kalman+Cible explorées intégralement** :

| Variation | Effet | Statut |
|-----------|-------|--------|
| Filtre standard / Filtre adaptatif | Bénéfique pour XGBoost uniquement | Documenté section 12 |
| Cible lag=1 / Cible lag=0 | Bénéfique pour CNN-LSTM uniquement | Documenté section 13 |
| Combinaison Adaptive + lag=0 | **Antagoniste**, dégrade -66% | Documenté section 14 |

**Le record absolu reste RSI CNN-LSTM Standard + lag=0 = +443% PnL Net filtré.**

### 14.8 Pistes restantes pour dépasser +443%

1. **Consensus inter-modèles** : MACD CNN-LSTM lag=0 + RSI CNN-LSTM lag=0
   → leurs erreurs sont-elles décorrélées ? (à mesurer comme on l'a fait
   pour standard lag=1)

2. **Stabilisation des flips Model pur** : RSI CNN-LSTM lag=0 fait 10,927
   trades pour -2,205% PnL Net. Une persistence/hysteresis efficace
   pourrait diviser les trades par 3-5 et passer en positif sans filtre
   oracle (utilisable en prod).

3. **Architectures alternatives** :
   - Transformer attention sur séquence longue (capture transitions
     plus subtiles)
   - Multi-task : prédire direction + magnitude
   - Loss PnL-aware (la BCE actuelle ignore le coût des flips)

4. **Pivot structurel cible** :
   - Régression rendement futur (continu au lieu de binaire)
   - Multi-horizon (5min, 15min, 30min, 1h, 2h convergents)
   - 3 classes UP/NEUTRE/DOWN (modèle peut rejeter)

### 14.9 Commandes reproductibles

```bash
python scripts/prepare_progressive_data.py --indicator rsi --tf 30 --adaptive --slope-lag 0
python scripts/train_cnn_lstm_progressive.py --npz data/prepared/dataset_rsi_30m_full_progressive_adaptive_lag0.npz
python scripts/backtest_model_filtered_by_oracle.py --npz data/prepared/dataset_rsi_30m_full_progressive_adaptive_lag0.npz --preds data/prepared/preds_rsi_30m_full_progressive_cnnlstm_adaptive_lag0.npz --split test
```

---

## 15. Filtre externe ATR — architecture 2 étages (échec validé)

Test : appliquer un filtre ATR **EN AVAL** du modèle (pas comme feature
interne) pour bloquer les signaux émis dans des conditions de marché
défavorables.

**Principe** :
- Niveau 1 (rapide) : signal direction = `sign(model.proba - 0.5)` du modèle
- Niveau 2 (lent)   : filtre ATR (low ≤ ATR ≤ high) → garde le signal seulement si dans la bande
- Décision finale   : trade si OK, conserve position si bloqué (slope = 0)

⚠️ Le modèle N'EST PAS retrained. On utilise les preds existantes.
Itération rapide : on teste plusieurs seuils ATR sans toucher au modèle.

### 15.1 Architecture testée

- Indicateur : RSI (config best : Oracle +1,208% PnL Net en lag=0)
- Modèle : CNN-LSTM Standard + lag=0 (record à +443% Model ∩ Oracle)
- Filtre : ATR(14) Wilder (causal, EMA récursif)
- Cible objectif : transformer Model pur **-2,205% PnL Net** en POSITIF
- Implémentation : `core.calculate_atr` + `scripts/backtest_external_filter.py`

### 15.2 Stats ATR normalisé (test set, RSI CNN-LSTM lag=0)

ATR/close en zone test : la majorité du temps en faible volatilité
- Médiane ~0.001-0.002
- 99% du temps < 0.005
- Très rare au-dessus de 0.01

### 15.3 Grids testés (3 angles)

**Grid 1 — Vol haute (filtrer marchés actifs)** :

| Bande ATR | Inband% | Trades | WR | PnL Brut | **PnL Net** |
|-----------|---------|--------|-----|----------|-------------|
| [0.005, 0.010] | 1.6% | 181 | 32.6% | -91% | **-127%** ⭐ best |
| [0.005, 0.020] | 1.6% | 183 | 32.8% | -94% | -130% |
| [0.003, 0.010] | 10.1% | 1,092 | 31.0% | +34% | -184% |
| [0.002, 0.010] | 26.8% | 2,983 | 25.2% | +28% | -568% |
| [0.001, 0.010] | 72.2% | 7,929 | 18.1% | -13% | -1,599% |

**Grid 2 — Vol basse (filtrer marchés calmes)** :

| Bande ATR | Inband% | Trades | WR | PnL Brut | **PnL Net** |
|-----------|---------|--------|-----|----------|-------------|
| [0, 0.0005] | 4.6% | 624 | 7.9% | -73.5% | **-198%** ⭐ best |
| [0, 0.001] | 27.8% | 3,580 | 9.5% | -23% | -739% |
| [0, 0.0015] | 54.8% | 6,632 | 11.4% | -75.5% | -1,402% |
| [0, 0.002] | 73.2% | 8,541 | 12.6% | -15% | -1,723% |
| [0, 0.0025] | 83.9% | 9,563 | 13.2% | -102% | -2,014% |

**Grid 3 — Vol moyenne (sweet spot hypothétique)** :

| Bande ATR | Inband% | Trades | WR | PnL Brut | **PnL Net** |
|-----------|---------|--------|-----|----------|-------------|
| [0.0015, 0.002] | 18.4% | 2,619 | 20.4% | -18% | **-542%** ⭐ best |
| [0.0015, 0.003] | 35.1% | 4,199 | 19.9% | -91% | -931% |
| [0.001, 0.002] | 45.4% | 5,545 | 16.3% | -10% | -1,120% |

### 15.4 Diagnostic — pourquoi l'ATR seul échoue

**Le modèle a un edge brut quasi-nul par trade** :

| Stratégie | Trades | PnL Brut | **PnL Brut / trade** |
|-----------|--------|----------|----------------------|
| Model pur | 10,927 | -19.89% | -0.0018% |
| ATR vol haute (best) | 181 | -90.90% | -0.502% (275× pire) |
| ATR vol moyenne (best) | 2,619 | -17.82% | -0.0068% |

**Quel que soit le filtre ATR**, le PnL Brut moyen reste entre -0.5% et
+0.04% par trade. **Les fees (0.2%/trade) dominent dans toutes les zones
de volatilité**.

### 15.5 Conclusion sur ATR seul

**ATR seul ne peut PAS sauver le Model pur** :
- ✅ Réduction massive des trades (jusqu'à -98%, ex: 10,927 → 181)
- ✅ Léger gain WR (14.7% → 32.6% en best vol haute)
- ❌ Aucune config positive sur le Model pur
- ❌ Le PnL Brut/trade reste dominé par les fees dans toutes les zones

**Le problème n'est pas la quantité de trades (filtrable par ATR) mais
la qualité intrinsèque du signal** (non discriminée par la volatilité seule).

### 15.6 Pistes alternatives (non testées par décision utilisateur)

1. **Volume comme filtre** : volume confirmé = institutions actives vs bruit
2. **Confidence filter** : ne trader que si `|proba - 0.5| > seuil`
3. **Combinaison ATR + Volume + Confidence** : triple filtre, grid 3D
4. **Pivot structurel** : changer la cible (régression rendement, multi-horizon)

### 15.7 Ce que ce test apporte

Même négatif, ce test confirme :
- ✅ L'architecture en 2 étages (modèle + filtre) est **techniquement viable**
  (`core.calculate_atr` + `backtest_external_filter.py` fonctionnent)
- ✅ Itération **ultra-rapide** : tester un nouveau seuil ATR = secondes
  (vs 10 min pour un retrain modèle)
- ✅ Le modèle CNN-LSTM lag=0 a un **edge brut neutre** (-0.0018% par trade)
  → pas un problème de filtrage, mais de signal sous-jacent
- ✅ Filtrer la volatilité ne suffit jamais quand l'edge intrinsèque est nul

### 15.8 Commandes reproductibles

```bash
# Stats ATR + grid auto
python scripts/backtest_external_filter.py \
    --npz data/prepared/dataset_rsi_30m_full_progressive_lag0.npz \
    --preds data/prepared/preds_rsi_30m_full_progressive_cnnlstm_lag0.npz \
    --split test --period 14 --normalize

# Grids spécifiques (vol haute / vol basse / vol moyenne)
python scripts/backtest_external_filter.py ... --atr-lows 0.005 --atr-highs 0.010 0.020 0.050
python scripts/backtest_external_filter.py ... --atr-lows 0.0 --atr-highs 0.0005 0.001 0.0015
python scripts/backtest_external_filter.py ... --atr-lows 0.0015 0.001 --atr-highs 0.002 0.003 0.004
```

---

## 16. Pipeline Meta-Classifier — architecture en 2 étages (PoC validée)

Idée utilisateur (validée 2026-04-18) : entraîner un meta-classifier qui apprend
à filtrer les **flips parasites** du modèle direction en utilisant l'oracle
comme superviseur.

**Architecture en 2 étages** :
```
Étage 1 : modèle direction (CNN-LSTM RSI lag=0) → signal à chaque 5min
Étage 2 : meta-classifier (XGBoost LONG / SHORT spécialisés)
            → "ce flip vaut-il la peine ?" (proba_meta)
            → si proba > seuil : exécuter
            → sinon : conserver position
```

### 16.1 Étape 2.A — Extraction des flips + features contextuelles

Script : `scripts/extract_model_flips.py`

À chaque flip du modèle (`sign(p)[t] != sign(p)[t-1]`), capture :
- Features de marché causales (depuis CSV BTCUSD)
- Features d'état interne du modèle (depuis preds NPZ)
- Label de profitabilité (simulation du trade qui suit)

**Itération 1 → Itération finale (boucle d'amélioration)** :

| Feature | Cohen's d (good) | Cohen's d (profitable) | Décision |
|---------|------------------|------------------------|----------|
| `time_since_last_flip` | 0.07 | — | ❌ ÉLIMINÉE |
| `range_vs_atr` | 0.04 | — | ❌ ÉLIMINÉE |
| `recent_flip_count_1h` | 0.05 | -0.05 à -0.11 | ❌ ÉLIMINÉE |
| `distance_to_ma60` | 0.10 | -0.07 | ❌ ÉLIMINÉE |
| `close_slope_4h` | 0.16 | ±0.03-0.13 | ❌ ÉLIMINÉE |
| `atr_14_norm` | 0.13 | **+0.39 à +0.45** | ✅ TOP |
| `atr_ratio_sl` | 0.16 | **+0.30 à +0.34** | ✅ TOP |
| `distance_to_ma20` | 0.31 | **+0.23 à +0.26** | ✅ TOP (asymétrique) |
| `proba_distance_to_extreme` | (nouveau) | **-0.31** | ✅ TOP |
| `proba_trend_3rows` | (nouveau) | **+0.24** (asymétrique) | ✅ TOP |
| `volume_relative` | 0.14 | **+0.21 à +0.23** | ✅ FORT |
| `close_slope_1h` | 0.22 | ±0.13-0.20 | ⚠️ borderline (gardé) |
| `proba_std_12rows` | (nouveau) | +0.17 à +0.18 | ⚠️ modeste |

**Découverte critique** : le label `is_profitable_flip` (simulation du trade)
discrimine **3 fois mieux** que `is_good_flip` (oracle instantané) sur l'ATR.
Volatilité élevée → trade plus long → profitable après fees.

**Set final = 12 features** : 3 temporel (hour_utc, dayofweek, month) + 4 vol/tendance/MA
+ 1 volume + 4 état interne modèle.

### 16.2 Étape 2.B — Statistiques descriptives des flips

Sur RSI CNN-LSTM lag=0 test (10,927 flips totaux) :

| Direction | Flips | `is_good_flip` rate | `is_profitable_flip` rate |
|-----------|-------|---------------------|---------------------------|
| LONG | 5,464 | 63.47% | **15.25%** |
| SHORT | 5,463 | 61.08% | **14.17%** |

**Énorme écart** entre les 2 labels = oracle d'accord ≠ trade profitable
(durée trop courte → fees détruisent).

### 16.3 Étape 2.C — Entraînement XGBoost meta-classifiers

Script : `scripts/train_meta_classifier_flips.py`

2 classifiers spécialisés :
- `meta_long` : entraîné sur les 5,464 flips LONG
- `meta_short` : entraîné sur les 5,463 flips SHORT

Hyperparams : XGBoost, n_estimators=500, max_depth=4, lr=0.05, scale_pos_weight
automatique pour gérer le déséquilibre 85/15. Early stopping sur val.

**Résultats classification** :

| Direction | Train AUC | Val AUC | **Test AUC** | Gap | best_iter |
|-----------|-----------|---------|--------------|-----|-----------|
| **LONG** | 0.7954 | 0.6674 | **0.6956** ✅ | 0.10 | 38 |
| **SHORT** | 0.7786 | 0.6381 | **0.6790** ✅ | 0.10 | 26 |

**Vrai signal détecté** (AUC > 0.65 sur les 2 directions, gap modéré).

**Performance avec seuils calibrés** (test set, base rate 13-15%) :

| Direction | Seuil F1 | Seuil High-Precision |
|-----------|----------|----------------------|
| LONG | 0.50 → Prec 19.7% / Rec 70.8% / F1 30.8% | **0.65 → Prec 30.1% / Rec 20.8%** (lift 2.33×) |
| SHORT | 0.45 → Prec 21.9% / Rec 78.9% / F1 34.3% | 0.55 → Prec 27.6% / Rec 40.7% (lift 1.84×) |

**Feature importance (gain) — top 3** :
- LONG : `atr_14_norm` (43.3), `distance_to_ma20` (32.9), `proba_distance_to_extreme` (32.2)
- SHORT : `atr_14_norm` (40.0), `distance_to_ma20` (37.3), `hour_utc` (25.5)

`atr_14_norm` est **#1 sur les 2 directions** = volatilité = facteur clé.

### 16.4 Étape 2.D — Backtest avec meta-filter

Script : `scripts/backtest_with_meta_filter.py`

Pour chaque flip détecté, lookup proba_meta correspondante (LONG ou SHORT).
Si proba > seuil → exécuter, sinon → conserver position. Backtest via
`core.backtest_5min_progressive`.

#### Mode 'all' — filtre tous les flips (in-sample partiel)

**🏆 PREMIER PnL NET POSITIF DU PIPELINE** sur RSI CNN-LSTM lag=0 :

| Stratégie | Trades | WR | PF | PnL Brut | Fees | **PnL Net** | Capture |
|-----------|--------|-----|-----|----------|------|-------------|---------|
| Oracle | 3,261 | 59.5% | 4.29 | +1,860% | 652% | +1,208% | +100% |
| Model pur | 10,927 | 14.7% | 0.28 | -20% | 2,185% | -2,205% | -183% |
| **meta thr=0.65** | **296** | **57.4%** | **1.55** | +194% | 59% | **+135.13%** ✅ | **+11.2%** 🏆 |
| meta thr=0.60 | 984 | 43.7% | 1.17 | +283% | 197% | +86.48% | +7.2% |
| meta thr=0.55 | 1,815 | 38.1% | 0.95 | +322% | 363% | -41.38% | -3.4% |

**Gain absolu : +2,340 points** (de -2,205% à +135.13%).
**WR multiplié par 4** (14.7% → 57.4%). **Trades divisés par 37**.

#### Mode 'meta_test_only' — rigoureux (out-of-sample meta seulement)

| Stratégie | Trades | PnL Net | Gain vs Model pur |
|-----------|--------|---------|-------------------|
| meta thr=0.65 (best) | 9,336 | -1,887% | +318 (négligeable) |

**Diagnostic** : le mode rigoureux ne filtre que **1,640 flips test du meta**
sur 10,927 totaux. Les 9,287 autres flips (train+val du meta, déjà vus en
training) restent bruts → dominent le PnL négatif.

### 16.5 Caveat critique — biais in-sample

**Le +135.13% du mode 'all' comprend un biais** :
- Le meta-classifier a été entraîné sur 70% des flips du test set du modèle direction
- Quand on filtre TOUS les flips (mode 'all'), 70% sont déjà vus → in-sample partiel
- Le gain est partiellement de l'overfit memorization

**Validation propre exigerait** (Option B non réalisée) :
1. Régénérer preds modèle CNN-LSTM sur **train + val** (au lieu de test seul)
2. Extraire flips sur ces preds
3. Entraîner meta-classifier sur ces flips out-of-sample
4. Évaluer sur les flips test (qui n'ont jamais été vus par le meta ni le modèle)

**Estimation prudente du PnL Net out-of-sample réel** :
- AUC out-of-sample test du meta = 0.69-0.70 (sur 1,640 flips, 60-65 jours)
- Lift réel = 2.0-2.3× la base rate
- Si on extrapole : **PnL Net entre +20% et +70%** sur 60-65 jours en propre
- Annualisé : **+120 à +400% / an** (à confirmer)

### 16.6 Apport scientifique

Même avec le biais in-sample, ce pipeline démontre :

✅ **L'architecture en 2 étages est techniquement viable** (preuve de concept)
✅ **Un meta-classifier discriminant existe** (AUC 0.69 out-of-sample propre)
✅ **L'ATR est le top discriminant** quand on labellise par profitabilité
   (vs Cohen's d faible quand on labellise par accord oracle instantané)
✅ **2 classifiers spécialisés** justifiés (asymétrie LONG/SHORT confirmée)
✅ **Le label `is_profitable_flip` est plus discriminant que `is_good_flip`**
   (3× plus fort sur ATR)
✅ **Premier PnL Net positif du pipeline** (+135% mode 'all', biais inclus)

### 16.7 Limites identifiées

❌ Validation rigoureuse non faite (Option B non réalisée)
❌ Test sur RSI uniquement (pas étendu à MACD/CCI)
❌ Test sur lag=0 uniquement (best config CNN-LSTM)
❌ Catégories 2 et 3 de features non testées (HMM régime, multi-TF agreement)

### 16.8 Commandes reproductibles

```bash
# 1. Extract flips (avec features v3 propres)
python scripts/extract_model_flips.py \
    --npz data/prepared/dataset_rsi_30m_full_progressive_lag0.npz \
    --preds data/prepared/preds_rsi_30m_full_progressive_cnnlstm_lag0.npz \
    --split test --label profitable

# 2. Train meta-classifiers (LONG + SHORT)
python scripts/train_meta_classifier_flips.py \
    --long-csv results/flips/flips_to_long_rsi_30m_full_cnnlstm_lag0_test.csv \
    --short-csv results/flips/flips_to_short_rsi_30m_full_cnnlstm_lag0_test.csv

# 3. Backtest avec meta-filter (grid de seuils)
python scripts/backtest_with_meta_filter.py \
    --npz data/prepared/dataset_rsi_30m_full_progressive_lag0.npz \
    --preds data/prepared/preds_rsi_30m_full_progressive_cnnlstm_lag0.npz \
    --meta-long results/meta_flips/meta_long_preds_rsi_30m_full_cnnlstm_lag0_test.npz \
    --meta-short results/meta_flips/meta_short_preds_rsi_30m_full_cnnlstm_lag0_test.npz \
    --split test
```

---

## 17. Option B — Validation OOB rigoureuse (CONCLUSION FINALE)

Validation rigoureuse du pipeline meta-classifier sans biais in-sample.

**Méthodologie OOB** :
1. Extraction flips sur split **val** (out-of-sample modèle direction)
2. Train meta sur ces flips val (split interne 85/15 pour early stop)
3. Évaluation finale sur flips **test** (out-of-sample modèle direction
   ET out-of-sample meta) — vraie performance attendue en production

### 17.1 Résultats classification OOB

| Direction | Train AUC | Val AUC | **Test AUC** | Gap vs in-sample | best_iter |
|-----------|-----------|---------|--------------|------------------|-----------|
| **LONG** | 0.7619 | 0.6384 | **0.6581** ✅ | -3.7% (modeste) | 40 |
| **SHORT** | 0.7277 | 0.5472 | **0.6468** ✅ | -3.2% | **2** ⚠️ |

**Diagnostic important** : le SHORT classifier converge en seulement
**2 itérations** vs 26 en in-sample. Indique un sous-entraînement
(`scale_pos_weight=6.35` peut-être trop élevé).

### 17.2 Top features OOB (cohérent avec in-sample)

LONG :
- `atr_14_norm` : 57.36 (gain) — **#1 confirmé**
- `distance_to_ma20` : 40.91
- `dayofweek` : 30.69 (nouveau dans top 3)

SHORT :
- `atr_14_norm` : 93.08 — **#1 confirmé**
- `distance_to_ma20` : 51.10
- `model_proba` : 34.55

### 17.3 Backtest OOB — Le +135% in-sample s'effondre à +54% (artefact)

**Sur le test set RSI CNN-LSTM lag=0 (458 jours)** :

| Stratégie | Trades | WR | PF | PnL Net | Statistiquement valide ? |
|-----------|--------|-----|-----|---------|--------------------------|
| Oracle | 3,261 | 59.5% | 4.29 | +1,208% | (référence) |
| Model pur | 10,927 | 14.7% | 0.28 | -2,205% | (baseline) |
| Meta in-sample 'all' best | 296 | 57.4% | 1.55 | +135% | ❌ overfit |
| **Meta OOB thr=0.70 best** | **2** | **100%** | **inf** | **+54%** | ❌ trop peu (2 trades !) |
| Meta OOB thr=0.65 | 2 | 100% | inf | +49% | ❌ trop peu |
| Meta OOB thr=0.50 (raisonnable) | 4,245 | 25.6% | 0.50 | **-787%** | ⚠️ négatif |
| Meta OOB thr=0.55 | 2 | 100% | inf | -39% | ❌ trop peu |

**Pourquoi seulement 2 trades aux seuils ≥ 0.55 ?**

- LONG accepts 1,902 flips (35%), SHORT accepts **0** flips
- Le SHORT classifier (mal entraîné, best_iter=2) rejette TOUT
- → On entre LONG, on ne sort jamais → 2 positions tenues plusieurs mois
- **C'est un artefact, pas un signal**

### 17.4 Conclusion finale du pipeline meta-classifier

#### Ce qui est validé scientifiquement

✅ **Pattern réel détecté** :
- AUC OOB ~0.65 sur les 2 directions (vrai signal hors training)
- Gap in-sample/OOB modeste (~3-4%, pas catastrophique)
- Top features cohérents : ATR, distance_to_ma20, proba_distance_to_extreme

✅ **Architecture en 2 étages techniquement viable**
✅ **Méthodologie correcte** (label is_profitable_flip, features causales,
   séparation LONG/SHORT)

#### Ce qui n'est PAS exploitable

❌ **Le PnL OOB pratique n'est pas viable en production** :
- Aux seuils raisonnables (0.50) : -787% net
- Aux seuils sélectifs (0.65+) : 2 trades sur 458 jours = artefact
- Pas de sweet spot "trades modérés + PnL positif"

❌ **Le SHORT classifier nécessite ré-engineering** :
- Sous-entraîné (best_iter=2)
- Rejette tout aux seuils calibrés sur val
- Possible : `scale_pos_weight` trop élevé, `max_depth` trop bas

❌ **Le +135% in-sample précédent était de l'overfit** :
- Confirmation par OOB : best honnête = +50% à 2 trades (statistiquement nul)
- L'extrapolation +20-70% antérieure était trop optimiste

#### Verdict honnête

**Le pattern existe mais l'amplitude est trop faible pour battre les fees 0.2%/trade en production.**

| Métrique | Valeur OOB |
|----------|------------|
| AUC LONG | 0.6581 (signal réel) |
| AUC SHORT | 0.6468 (signal réel) |
| Lift vs base rate | ~1.5-2× (modeste) |
| **Edge net après fees** | **insuffisant** |

### 17.5 Pistes pour rendre le pattern exploitable (non testées)

1. **Ajuster SHORT classifier** : scale_pos_weight modéré, max_depth=5,
   plus d'epochs, peut-être early_stop=50
2. **Maker fees 0.02%** : fees ÷10 → -787% deviendrait peut-être ~-50%
   ou positif (à vérifier)
3. **Catégorie 2 features** : HMM régime, ADX, Hurst (jamais testées)
4. **Catégorie 3 features** : multi-TF agreement (5min/30min/1h)
5. **Ensemble** : meta-classifier sur consensus MACD+CCI+RSI au lieu de RSI seul
6. **Architecture différente** : Transformer, multi-task learning avec
   loss PnL-aware

### 17.6 Décision sur la suite du sujet "meta-classifier"

**Le sujet est CLOS** dans son état actuel :
- Pattern réel détecté mais marginal
- Pas de PnL Net positif rigoureux
- Pour aller plus loin il faudrait des modifications structurelles
  (fees taker→maker, features Cat 2-3, ou architecture deep)

### 17.7 Commandes Option B reproductibles

```bash
# 1. Extract flips sur val (out-of-sample modèle direction)
python scripts/extract_model_flips.py \
    --npz data/prepared/dataset_rsi_30m_full_progressive_lag0.npz \
    --preds data/prepared/preds_rsi_30m_full_progressive_cnnlstm_lag0.npz \
    --split val --label profitable

# 2. Train meta OOB (val=train, test=eval externe)
python scripts/train_meta_classifier_flips.py \
    --long-csv  results/flips/flips_to_long_rsi_30m_full_cnnlstm_lag0_val.csv \
    --short-csv results/flips/flips_to_short_rsi_30m_full_cnnlstm_lag0_val.csv \
    --long-test-csv  results/flips/flips_to_long_rsi_30m_full_cnnlstm_lag0_test.csv \
    --short-test-csv results/flips/flips_to_short_rsi_30m_full_cnnlstm_lag0_test.csv

# 3. Backtest OOB (mode 'all' = vraiment OOB)
python scripts/backtest_with_meta_filter.py \
    --npz data/prepared/dataset_rsi_30m_full_progressive_lag0.npz \
    --preds data/prepared/preds_rsi_30m_full_progressive_cnnlstm_lag0.npz \
    --meta-long  results/meta_flips/meta_long_preds_rsi_30m_full_cnnlstm_lag0_test_oob.npz \
    --meta-short results/meta_flips/meta_short_preds_rsi_30m_full_cnnlstm_lag0_test_oob.npz \
    --split test
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
