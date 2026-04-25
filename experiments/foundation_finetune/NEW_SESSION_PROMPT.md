# Prompt d'ouverture — Nouvelle session post-foundation_finetune

## Contexte du projet

Tu reprends une session après le projet `experiments/foundation_finetune/`
(branche `claude/merge-and-prepare-session-KkpbU`). Ce projet a testé
**Chronos LoRA** comme modèle de fondation séries temporelles pour
reconstruire en causal la pente Oracle Kalman du RSI/MACD.

## Contraintes héritées (toujours valides)

- **Lecture seule absolue** sur :
  - `src/`
  - `experiments/slope_improvement/` (projet clos précédent)
- **Toute extension** va dans `experiments/foundation_finetune/` ou un
  nouveau module dédié
- **Réutiliser l'existant** : il y a beaucoup de fonctions dans
  `src/signal_processing/core.py` (FLKS, Kalman 30min, sub-step,
  prepare_features_and_labels_progressive, etc.) — toujours auditer
  avant de coder

## Findings principaux à connaître

### 1. Le RSI 5min seul plafonne à Pearson 0.78 / lag=-1 contre Oracle Kalman 5min

- Autocorr lag-1 de la pente Oracle = **0.93** → tout modèle causal qui
  reproduit le passé proche est borné à Pearson ≤ 0.93
- Enrichir avec MACD/CCI (Phase 6) ou Volume/ATR (Phase 8) → **0 gain**
- Cible décalée Oracle[t+1] (Phase 7) → Pearson chute à 0.54 → **proxy
  learning confirmé**

### 2. FLKS sub-step + skip connection = pipeline gagnant (Phase 13)

Architecture finale Chronos LoRA :

```
slope_progressive[t-24:t+1]  ──→ Chronos T5 (perd amplitude par mean-scaling local)
                                                                  │
slope_progressive[t] z-score global (skip connection)  ──→ extras │
                                                                  ▼
                                                       cat → MLP → ŷ
```

**Sans skip connection** : Chronos atteint Pearson 0.80 sur MACD (sous
FLKS pur 0.98 de 17 pp).
**Avec skip connection** : Chronos atteint Pearson 0.96-0.99 par k,
**bat FLKS pur aux premiers sous-pas (k=0, k=1)**, égale aux derniers.

### 3. Bug architectural Chronos (à connaître pour tout futur backbone TS)

Le tokenizer Chronos applique un mean-scaling **local par séquence**
avant binning quantile → **l'amplitude absolue est perdue**. Pour
reproduire un estimateur paramétrique amplitude-dépendant (Kalman
smoothed), il faut **passer la valeur brute en skip connection** au
MLP head, pas seulement via le backbone.

Cela vaut pour Chronos, Lag-Llama, et probablement aussi MOIRAI/TimesFM
(à vérifier selon backbone choisi).

### 4. Caveat « FLKS auto-prédit »

Le baseline `slope_progressive[t]` dé-normalisé atteint Pearson 0.94-0.99
contre `label_continuous` parce que :
- La cible **est** l'Oracle Kalman RTS smoothed
- FLKS converge vers RTS smoothed quand lag → ∞
- → corrélation quasi-tautologique par construction

Pour vraiment tester un pouvoir prédictif au-delà du Kalman, il faut
une **cible indépendante de Kalman** : returns futurs, prix futurs,
PnL trading.

## Architecture du module foundation_finetune

```
experiments/foundation_finetune/
├── README.md                              # synthèse complète
├── NEW_SESSION_PROMPT.md                  # ce fichier
├── __init__.py
│
├── build_dataset.py                       # RSI 5min brut → slope past
├── build_dataset_fusion.py                # + MACD/CCI slopes (rejected)
├── build_dataset_volume_atr.py            # + Volume/ATR (rejected)
├── build_dataset_future.py                # cible Oracle[t+1]-Oracle[t]
├── build_dataset_flks_substep.py          # FLKS sub-step (FINAL, --indicator)
│
├── baselines.py                           # identity, raw_slope, ma_slope_K
├── model.py                               # ChronosRegressor + LoRA + extras
├── train.py                               # MSE loss + early stopping
├── evaluate.py                            # comparison vs baselines + lag CCF
├── evaluate_per_substep.py                # décompose par step_k=0..5
```

### Réutilisation projet (lecture seule)

| Module | Fonctions clés |
|---|---|
| `src/data_utils.py` | `load_crypto_data` |
| `src/indicators.py` | `calculate_rsi`, `calculate_macd`, `calculate_cci`, `calculate_atr` |
| `src/filters.py` | `kalman_filter` (RTS) |
| `src/signal_processing/core.py` | `load_csv`, `resample_ohlcv`, `prepare_features_and_labels_progressive`, `compute_progressive_slopes`, `forward_filter_30m`, `compute_slopes_test1/2`, `split_train_val_test`, `normalize_features`, `make_sequences` |
| `src/constants.py` | constantes du projet (RSI_PERIOD, KALMAN_*, etc.) |

## Pipeline reproductible final

```bash
# Setup deps (Python 3.13 + CUDA 12.4 testé sur RTX 4070 SUPER)
pip install chronos-forecasting peft pykalman

# 1. Build dataset MACD 30min FLKS sub-step + skip connection (~5 min)
python experiments/foundation_finetune/build_dataset_flks_substep.py --indicator macd

# 2. Train Chronos LoRA + extras (~6 min GPU)
python experiments/foundation_finetune/train.py \
    --mode lora --epochs 5 --batch-size 256 --num-workers 4 \
    --data data/foundation/macd_btc_5min_flks_substep.npz \
    --output-dir models/foundation_finetune_macd_flks_v2

# 3. Eval per sub-step Chronos vs FLKS pur
python experiments/foundation_finetune/evaluate_per_substep.py \
    --data data/foundation/macd_btc_5min_flks_substep.npz \
    --ckpt models/foundation_finetune_macd_flks_v2/chronos-t5-tiny_lora_fusion.pt
```

Durée totale : ~15-20 min GPU.

## Résultats finaux à mémoriser

**MACD 30min, test set, 131,753 samples, Chronos-t5-tiny LoRA r=8 + skip connection** :

| k | Chronos sc% | Chronos Pearson | FLKS sc% | FLKS Pearson | Δ Pearson |
|---|---|---|---|---|---|
| 0 | **90.55%** | **0.9571** | 89.92% | 0.9503 | **+0.0068** |
| 1 | **93.63%** | **0.9792** | 93.34% | 0.9772 | **+0.0020** |
| 2 | 94.13% | 0.9828 | 94.21% | 0.9827 | +0.0001 |
| 3 | 94.62% | 0.9848 | 94.64% | 0.9851 | -0.0003 |
| 4 | 95.00% | 0.9868 | 95.03% | 0.9871 | -0.0003 |
| 5 | 95.40% | 0.9888 | 95.41% | 0.9891 | -0.0003 |

→ **Chronos bat FLKS aux 2 premiers sous-pas, égale aux 4 derniers.**

## Pistes non explorées (documentées dans README §"Pour aller plus loin")

- **Cible vraiment prédictive** (returns futurs, prix futurs, PnL)
- **AQ-KF Q adaptatif** : `--adaptive` (testé partiellement)
- **Architecture multi-channel** : dual-encoder (Chronos + small CNN/Transformer pour features auxiliaires)
- **Chronos-t5-small** (46M, 5× plus gros) ou autre backbone (TimesFM, MOIRAI)
- **Sortie probabiliste** (Student-t native de Chronos) au lieu de régression scalaire

## Si la nouvelle session porte sur...

### Tester un nouveau modèle de fondation TS

L'infra `model.py` / `train.py` / `evaluate.py` est conçue spécifiquement
pour Chronos. Pour un autre backbone :
- Si HuggingFace transformers compatible (TimesFM, certaines variantes Lag-Llama)
  → adapter `model.py` (`ChronosRegressor` → `<New>Regressor`) en gardant la
  même interface (forward `(x_rsi, extras=None)`, `extra_dim` paramètre)
- **Vérifier d'abord** si le tokenizer du nouveau modèle préserve l'amplitude
  ou pas (test simple : passer une fenêtre constante × 2 et voir si la sortie
  varie). Si non préservée, **garder la skip connection** comme dans Chronos
  Phase 13.
- Réutiliser `train.py` / `evaluate.py` / `evaluate_per_substep.py` tels quels
  (formats de checkpoint et dataset compatibles)

### Tester une nouvelle cible

- Modifier `build_dataset_flks_substep.py` pour générer `y_<split>` à partir
  d'une cible custom (returns futurs, PnL, etc.)
- L'infra `train.py` / `evaluate.py` reste compatible
- Comparer vs FLKS pur (qui n'aurait plus de baseline biaisé sur cible
  indépendante du Kalman)

### Migrer en production

- Utiliser le ckpt `models/foundation_finetune_macd_flks_v2/chronos-t5-tiny_lora_fusion.pt`
- Pour l'inférence live : il faut maintenir le pipeline FLKS sub-step en
  temps réel + le ckpt Chronos
- Latence : ~10-20 ms par prédiction sur GPU, davantage sur CPU

### Nouveau projet sans rapport

- Garder cette branche close, démarrer sur une nouvelle branche
  `claude/<nom-nouveau-projet>`
- Référencer `README.md` et ce prompt comme contexte hérité

## Instructions de continuité

1. **Lire d'abord** `experiments/foundation_finetune/README.md` (synthèse 13 phases)
2. **Si nouveau test sur ce module** : ajouter à `experiments/foundation_finetune/`,
   ne JAMAIS toucher `src/` ou `experiments/slope_improvement/`
3. **Convention de commit** : message clair, `https://claude.ai/code/session_<id>` à la fin
4. **Stop hook** : commit + push à chaque ajout de fichier
5. **Auditer avant de coder** : utiliser `Agent` (Explore) pour inventorier
   les scripts/fonctions existants

## Question à clarifier au début de la nouvelle session

L'utilisateur précisera son intention. Reformuler explicitement :

- **(a) Tester un nouveau backbone TS** sur la même tâche FLKS sub-step
  → quel backbone, quel critère de succès ?
- **(b) Tester une nouvelle cible** (cible vraiment prédictive)
  → laquelle, sur quelles données ?
- **(c) Architecture multi-channel** (multi-feature input)
  → quelles features, quel backbone ?
- **(d) Migration production** (utiliser ckpt existant en live)
  → quel use case ?
- **(e) Autre projet** (foundation_finetune comme contexte uniquement)
  → quel objectif ?

Ne pas commencer à coder avant que l'objectif soit clair.
