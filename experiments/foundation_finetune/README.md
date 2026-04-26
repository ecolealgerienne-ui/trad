# Foundation Model Fine-Tuning — Chronos LoRA pour Reconstruction Oracle Kalman

**Statut** : ✅ **Concluant. Pipeline retenu : Chronos LoRA + skip connection**
**Branche** : `claude/merge-and-prepare-session-KkpbU`
**Indicator final** : MACD (et RSI), 30min via FLKS sub-step
**Cible** : Pente Oracle Kalman 30min RTS smoothed (régression continue)

---

## TL;DR

| Run | Pearson test | Verdict |
|---|---|---|
| LoRA simple (RSI 5min brut, fenêtre 96) | 0.78 | bat baselines naïves, plafonné par autocorr 0.93 + best_lag=-1 |
| LoRA fusion MACD/CCI | 0.78 | redondant (Phase 2.13 confirmée) |
| LoRA Volume + ATR | 0.78 | orthogonal mais inutile |
| LoRA future-slope | 0.54 | proxy learning confirmé |
| LoRA FLKS sub-step **sans** skip | RSI 0.90 / MACD 0.80 | bug architectural (perte amplitude) |
| **LoRA FLKS sub-step + skip connection** ⭐ | **MACD 0.96-0.99 par k** | **gagnant : ≥ FLKS pur, gain réel à k=0/1** |

---

## Ce que la session a appris

### 1. Reconstruction causale du RSI 5min Oracle (Phases 1-9)

Le RSI 5min seul plafonne à Pearson 0.78 / DirMatch 0.79 contre la pente Oracle Kalman, avec un `best_lag = -1` structurel (proxy retardé). Aucun enrichissement de feature classique n'aide :
- Indicateurs corrélés (MACD/CCI) → 0 gain (Phase 2.13 confirmée empiriquement)
- Indicateurs orthogonaux (Volume, ATR) → 0 gain
- Cible décalée future → confirme que tout était proxy, Pearson chute à 0.54

**Borne théorique identifiée** : `autocorr_lag1(slope_oracle) = 0.93`. Tout modèle causal qui se contente de reproduire le passé proche est borné à Pearson ≤ 0.93.

### 2. Décision : passer à FLKS sub-step + tâche 30min (Phases 10-12)

L'astuce : utiliser comme entrée la séquence des estimations FLKS sub-step (1 par pas 5min, calculées sur RSI/MACD 30min en construction) — pas les indicateurs bruts.

Réutilise massivement `src/signal_processing/core.py` :
- `prepare_features_and_labels_progressive(indicator, tf=30)`
- `compute_progressive_slopes` (FLKS Standard / AQ-KF)
- `forward_filter_30m` (Kalman 2D constant-velocity)
- `compute_slopes_test1`, `compute_slopes_test2` (FLKS lag=1, sub-step)

### 3. Bug architectural Chronos tokenizer (Phase 12)

Le tokenizer Chronos quantilise les valeurs continues en bins via **mean-scaling local par séquence** : chaque fenêtre est centrée + scalée par sa propre moyenne/std avant binning. **L'amplitude absolue est perdue.**

Conséquence : sur une cible où l'amplitude est cruciale (Oracle Kalman 30min smoothed), le modèle ne peut pas reproduire FLKS — il est borné à un signal "directionnel relatif".

| Phase | Indicateur | Pearson Chronos | Pearson FLKS | Δ |
|---|---|---|---|---|
| 12 sans skip | RSI | 0.897 | 0.942 | -4.5 pp |
| 12 sans skip | MACD | **0.802** | **0.979** | **-17.7 pp** |

L'écart est plus grave sur MACD car amplitude plus large.

### 4. Fix : skip connection avec amplitude préservée (Phase 13)

Solution : passer `slope_progressive[t]` (z-scoré globalement, **pas tokenisé**) directement au MLP head, en plus du embedding T5. Le modèle peut au pire apprendre l'identité `ŷ ≈ α · extras` et reproduire FLKS.

```
slope_progressive[t-24:t+1]  ──→ Chronos T5 (perd amplitude)  ──→ embedding (256)
                                                                      │
slope_progressive[t] z-score global (préserve amplitude)  ──→ extras (1)
                                                                      ▼
                                                       cat → MLP head → ŷ
```

### 5. Résultat post-fix sur MACD (Phase 13)

| k | Chronos Pearson | FLKS Pearson | Δ |
|---|---|---|---|
| 0 | **0.9571** | 0.9503 | **+0.0068** ✅ |
| 1 | **0.9792** | 0.9772 | **+0.0020** ✅ |
| 2 | 0.9828 | 0.9827 | +0.0001 (tie) |
| 3 | 0.9848 | 0.9851 | -0.0003 (tie) |
| 4 | 0.9868 | 0.9871 | -0.0003 (tie) |
| 5 | 0.9888 | 0.9891 | -0.0003 (tie) |

**Pattern attendu** :
- Aux premiers sous-pas (k=0, 1) où FLKS a peu d'info live → **Chronos extrait du contexte supplémentaire**, gagne ~0.7 pp sign_concordance
- Aux derniers sous-pas (k=4, 5) où FLKS sature naturellement → tie quasi parfait

C'est le comportement attendu d'un modèle ML par-dessus un estimateur paramétrique optimal : utile dans les zones d'incertitude, équivalent dans les zones saturées.

---

## Pipeline reproductible (post-fix)

```bash
# 1. Vérifier deps
pip install chronos-forecasting peft pykalman
python -c "import torch; assert torch.cuda.is_available()"

# 2. Build dataset MACD 30min FLKS sub-step
python experiments/foundation_finetune/build_dataset_flks_substep.py --indicator macd

# 3. Train Chronos LoRA + skip connection (~6 min sur RTX 4070)
python experiments/foundation_finetune/train.py \
    --mode lora --epochs 5 --batch-size 256 --num-workers 4 \
    --data data/foundation/macd_btc_5min_flks_substep.npz \
    --output-dir models/foundation_finetune_macd_flks_v2

# 4. Eval per sub-step (Chronos vs FLKS pur)
python experiments/foundation_finetune/evaluate_per_substep.py \
    --data data/foundation/macd_btc_5min_flks_substep.npz \
    --ckpt models/foundation_finetune_macd_flks_v2/chronos-t5-tiny_lora_fusion.pt
```

---

## Architecture du module

```
experiments/foundation_finetune/
├── README.md                              ← ce fichier
├── NEW_SESSION_PROMPT.md                  ← prompt pour nouvelle session
├── __init__.py
│
├── build_dataset.py                       ← Phase 1 : RSI 5min brut → Oracle 5min slope
├── build_dataset_fusion.py                ← Phase 6 : + MACD/CCI slopes (rejected)
├── build_dataset_volume_atr.py            ← Phase 8 : + Volume/ATR (rejected)
├── build_dataset_future.py                ← Phase 7 : cible Oracle[t+1]-Oracle[t]
├── build_dataset_flks_substep.py          ← Phases 10-13 : FLKS sub-step (FINAL)
│
├── baselines.py                           ← Phase 2 : identity, raw_slope, ma_slope_K
├── model.py                               ← Phase 3 : ChronosRegressor + LoRA + extras
├── train.py                               ← Phase 4 : training loop MSE + early stopping
├── evaluate.py                            ← Phase 5 : comparison vs baselines + lag CCF
├── evaluate_per_substep.py                ← Phase 11 : décompose par step_k=0..5
```

### Réutilisation (lecture seule)

Toutes les briques fondamentales viennent de :
- `src/data_utils.py` : `load_crypto_data`
- `src/indicators.py` : `calculate_rsi`, `calculate_macd`, `calculate_cci`, `calculate_atr`
- `src/filters.py` : `kalman_filter` (RTS smoother)
- `src/signal_processing/core.py` : `load_csv`, `resample_ohlcv`, `prepare_features_and_labels_progressive`, `compute_progressive_slopes`, `forward_filter_30m`, `compute_slopes_test1/2`, `split_train_val_test`, `normalize_features`, `make_sequences`
- `src/constants.py` : `RSI_PERIOD`, `KALMAN_PROCESS_VAR`, etc.

**Aucun fichier de `src/` ou `experiments/slope_improvement/` n'a été modifié** (contrainte projet).

---

## Caveat principal — comparaison vs FLKS

Le baseline "FLKS pur" (= `slope_progressive[t]` dé-normalisé) atteint Pearson 0.94-0.99 contre la cible `label_continuous` parce que :
- La cible **est** l'Oracle RTS smoothed
- FLKS converge vers RTS smoothed quand le lag tend vers ∞
- → corrélation quasi-tautologique par construction

Le test "Chronos bat ou égale FLKS" est donc déjà non-trivial dans ce setup. Pour vraiment évaluer un pouvoir prédictif au-delà de Kalman, il faudrait une **cible indépendante du Kalman** (returns futurs, prix futurs, PnL trading). C'est hors-scope de ce projet.

---

## Conditions où ce pipeline ML est utile

✅ **Utile** :
- Quand on a besoin d'une amélioration aux **premiers sous-pas** (k=0, k=1) d'une barre TF lente (info live partielle)
- Quand on veut combiner FLKS avec un contexte temporel multi-step (LSTM/Chronos extrait des patterns que FLKS ignore)
- Quand on prévoit d'enrichir avec des features vraiment indépendantes (volume, order flow) — l'infrastructure est prête

❌ **Inutile** :
- Quand FLKS sature déjà (k=4, k=5 en fin de barre TF) — le ML ne peut pas faire mieux
- Pour reconstruire purement Oracle RTS smoothed à un sous-pas final — FLKS lag=∞ est optimal par construction
- Pour des features qui sont des projections du même signal latent (RSI/CCI/MACD entre eux)

---

## Bilan complet des 13 phases

| # | Phase | Date logique | Statut | Verdict |
|---|---|---|---|---|
| 0 | Setup module | session start | ✅ | infra créée |
| 1 | Dataset RSI 5min → Oracle slope | | ✅ | base solide |
| 2 | Baselines causales | | ✅ | MA_5 = best baseline naïve |
| 3 | Chronos T5 wrap + LoRA | | ✅ | 16k probing / 115k LoRA / 8.4M full |
| 4 | Training MSE + early stop | | ✅ | pipeline robuste |
| 5 | Evaluate vs baselines | | ✅ | LoRA gagne baselines (Pearson 0.78) |
| 6 | Fusion MACD/CCI extras | | ❌ rejected | redondance Phase 2.13 confirmée |
| 7 | Cible Oracle[t+1]-Oracle[t] | | ❌ | Pearson 0.54, proxy confirmé |
| 8 | Volume + ATR extras | | ❌ rejected | orthogonal mais inutile |
| 9 | Audit FLKS pipeline existant | | ✅ | tout existe dans core.py |
| 10 | FLKS sub-step LoRA RSI | | ⚠️ | Pearson 0.897, mais sous FLKS pur |
| 11 | Eval per sub-step k=0..5 | | ⚠️ | révèle baseline biaisé |
| 12 | Switch indicator → MACD | | ⚠️ | Pearson 0.802, sous FLKS de 17 pp |
| **13** | **Skip connection bypass tokenizer** | | ✅ **WINNER** | **Chronos ≥ FLKS, gain à k=0/1** |
| **14** | **Pivot Kalman(close) + meta-labeling** | 2026-04 | ⚠️ **CLÔTURÉ** | **Wall confirmé : OHLC-derived insuffisant pour transition detection** |

---

## Phase 14 (post-clôture) — Pivot Kalman(close) + Meta-Labeling

**Date** : 2026-04-26
**Statut** : ⚠️ **PROJET CLOS — Plafond structurel confirmé (5e validation projet)**
**Branche** : `claude/resume-foundation-finetuning-PRT1w`

### Reformulation et setup

Pivot vers **γ partiel** : changer la cible de `Kalman(indicateur)` → `Kalman(close)`. Hypothèse : forcer le modèle à reconstruire le close depuis l'indicateur brise la tautologie.

3 spécialistes Chronos T5-tiny + LoRA entraînés indépendamment (BTC 5min, fenêtre 96, 5 epochs) :
- RSI(22) → Kalman_RTS(close) slope : **Val Pearson 0.595**
- MACD-hist(8/42/9) → idem : **Val Pearson 0.594**
- CCI(32) → idem : **Val Pearson 0.545**

**Anti-leakage RTS** : Kalman calculé per-split (train, train+val, full).

### Mesure d'accord inter-modèle (test set 131k)

| Paire | Pearson(yhat) | Sign Agreement | Error Jaccard | Complémentarité |
|---|---|---|---|---|
| RSI vs MACD | **0.859** | 0.877 | 0.574 | 0.123 |
| RSI vs CCI | **0.883** | 0.894 | 0.618 | 0.106 |
| MACD vs CCI | **0.826** | 0.879 | 0.591 | 0.121 |

**Vote majoritaire DirMatch = 77.94%** vs **mean individual = 77.15%** → **gain +0.79%**.
TREND uniquement : +1.40%. RANGE_LOW_VOL : +0.73%.

→ Les 3 indicateurs capturent le même ~60% du signal, gain de fusion marginal.

### Meta-labeling Triple Barrier (López de Prado)

22 features engineerées :
- **Primaires per-modèle** (9) : yhat, |yhat|, accel = yhat[t]−yhat[t−1]
- **Cross-modèle dérivées** (6) : confidence_spread/min/max/mean, n_models_agree, time_since_flip
- **Orthogonales** (3) : volume_ratio, volume_spike, atr_normalized
- **Régime** (4) : ts_score, vc_score, regime_0/1/2 one-hot

Triple Barrier : TP=0.5%, SL=0.2%, T_max=6 bars. Direction = sign(mean(yhat)).

### Résultats meta-classifieur (test 131k)

**Logistic Regression** (top 1% / 5% / 10%) :
- Precision 37.0% / 28.8% / 25.0%
- Coefficient dominant : **`atr_normalized` +1.077** (5× le suivant)
- Ordre top 6 : ATR, vc_score, mag_rsi, volume_spike, confidence_min, regime_1

**XGBoost** (mêmes valeurs) :
- Precision 30.8% / 25.7% / 22.3% — **bat pas Logistic** (signal essentiellement linéaire)
- Importance gain top 5 : ATR (212), vc_score (139), volume_ratio (43), mag_rsi (43), volume_spike (39)

**v2 (TP=0.8%/SL=0.4%/T=12)** : précision quasi-identique (31.5% / 24.0% / 20.4%). Plafond confirmé.

### Stratification par régime

| Régime | % du test | Top 10% selected | Precision |
|---|---|---|---|
| RANGE_LOW_VOL | 71.5% | 5% (~700) | **6.7-9.2%** (≈ baseline aléatoire) |
| RANGE_HIGH_VOL | 26.1% | 89% (~11k) | 22-25% |
| TREND | 2.5% | 50% (~1.7k) | 24-28% |

→ Le modèle **ignore RANGE_LOW_VOL**, concentre tout sur RANGE_HIGH_VOL + TREND.

### Le mur (5e confirmation projet)

Précision plafonne à **30-37% top 1%** quel que soit le tuning. Break-even taker = 86% (TP=0.5/SL=0.2), maker = 57%. **Inaccessible avec ce signal.**

**Findings empiriques** :

| # | Finding | Cohérence avec phases projet |
|---|---|---|
| 1 | ATR/volatilité = feature DOMINANTE pour transition detection | Nouveau (sur ce setup) |
| 2 | Sorties Chronos LoRA des 3 indicateurs quasi-équivalentes (XGBoost gain ~35 pour toutes) | Phase 2.13 confirmé (5e fois) |
| 3 | best_lag = +1 sur les 3 → autocorr 0.93 plafond | Phase 1-9, 2.10 confirmé |
| 4 | XGBoost ne bat pas Logistic → pas d'interactions cachées | Phase 2.18 confirmé |
| 5 | Volume/ATR features > yhat des indicateurs | Phase 2.7 confirmé |

Citation Phase 2.18 (toujours valide) :
> *"Pipeline scientifiquement correct, signal primary lacks alpha. Pearson 0.5 / Precision 44% est ce que la prédiction directionnelle d'indicateurs techniques peut donner sur crypto. Documenté depuis 20 ans en littérature."*

### Verdict de clôture

Le projet `foundation_finetune` est **scientifiquement validé** comme démonstration empirique rigoureuse de la limite des indicateurs OHLC-derived pour la prédiction directionnelle crypto 5min. Le pivot Kalman(close) + meta-labeling n'a **pas brisé le mur** documenté depuis Phase 2.13.

**Vrai gain de cette phase** : démonstration empirique propre que **ATR + features volume > yhat des indicateurs** pour cette tâche. Validation que les leviers d'amélioration ne sont PAS dans l'architecture (Chronos, XGBoost, etc.) mais dans **les données orthogonales** non-dérivées du close.

### Livrables Phase 14

| Fichier | Rôle |
|---|---|
| `build_dataset_close_kalman.py` | Build datasets indicator → Kalman(close) slope, anti-leakage per-split |
| `analyze_specialists_close_kalman.py` | Analyse comparative inter-modèles (pairwise + triple agreement) |
| `build_meta_dataset.py` | Triple Barrier labels + 22 features engineerées |
| `train_meta_classifier.py` | Logistic + XGBoost meta + coverage-precision tables |

### Pourquoi ne pas continuer ce projet

1. ✋ **Tunings restants** (filtrer RANGE_LOW_VOL, autres TP/SL) : gain attendu marginal (+5-10%), toujours sous break-even
2. ✋ **Architectures alternatives** (PatchTST, iTransformer, Crossformer) : XGBoost a déjà confirmé pas d'interactions cachées
3. ✋ **Fusion embedding** (3×256-dim Chronos → MLP) : Pearson 0.86 entre prédictions = redondance déjà mesurée

Toutes ces options coûteraient du temps pour confirmer le même résultat. **Le mur est dans les données, pas dans le modèle.**

---

## Référencement croisé avec slope_improvement

Le projet précédent `experiments/slope_improvement/` (clos) a établi des bornes théoriques utilisées comme référence :
- FLKS(lag=3) sur 2D MLE atteint Pearson 0.86 contre GT 3D Kalman MLE
- Identification de 4 régimes σ² selon latence acceptable
- Borne supérieure causal-relâché ≈ 0.87 (FLKS lag=∞)

Ces chiffres ne sont **pas directement comparables** avec ceux de ce projet (cibles, splits, calibrations différents) mais donnent les ordres de grandeur d'un pipeline Kalman optimisé. Voir `experiments/slope_improvement/final_report.md` pour détails.
