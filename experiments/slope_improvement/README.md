# Slope Improvement Experiments — Kalman RSI

Amélioration de l'estimation de la pente du RSI filtré par Kalman,
en partant du pipeline AQ-KF comme baseline.

## Objectif

Gain mesurable de MSE out-of-sample sur l'estimation causale de la pente,
sans data leakage ni oracle non-causal dans les inputs.

## Contrainte fondamentale — Ne touche pas à l'existant

Ce module **NE modifie AUCUN fichier** de `src/`. Il importe et réutilise :
- `src.indicators.calculate_rsi` (Wilder, cohérent avec le projet)
- `src.constants.RSI_PERIOD` (22), `KALMAN_PROCESS_VAR` (0.01), `KALMAN_MEASURE_VAR` (0.1)
- `pykalman.KalmanFilter` pour le RTS smoother (même lib que `src/filters.py`)

Le KF simplifié barre-par-barre (`kf_baseline.py`) est volontairement une
implémentation parallèle (pas un import de `compute_kalman_live`) car on
doit isoler l'effet des modifications du modèle sans la logique
closure/provisional multi-TF de la prod.

## Arborescence

```
experiments/slope_improvement/
├── data_loader.py         # BTC 5min → 2022+ → RSI(22) → split 50/25/25
├── ground_truth.py        # RTS full-pass (primary) + MA centrée (secondary)
├── kf_baseline.py         # Étape 1 : KF 2D CV, σ² adaptatif scalaire
├── metrics.py             # MSE, MAE, Pearson, DirMatch, latency, Diebold-Mariano
├── diagnostics.py         # ACF, Ljung-Box (manuel), Jarque-Bera, QQ, hist
├── run_experiments.py     # Orchestrateur Étapes 0+1
├── artifacts/             # Outputs : .npy, .json, .md, plots
└── README.md              # Ce fichier
```

## Exécution — Étapes 0 + 1

Depuis la racine du projet :

```bash
python experiments/slope_improvement/run_experiments.py \
    --csv data_trad/BTCUSD_all_5m.csv \
    --start-date 2022-01-01
```

Options :
- `--end-date YYYY-MM-DD` (optionnel, défaut = fin du CSV)
- `--out-dir /chemin/custom` (défaut = `experiments/slope_improvement/artifacts`)

Durée attendue : ~2-3 min (KF sur ~420k barres + diagnostic).

## Ce que produit le run

### Dans `artifacts/`

| Fichier | Contenu |
|---------|---------|
| `meta.json` | Config, splits, verdict gate, métriques résumées |
| `ground_truth_slope_rts.npy` | Pente RTS full-pass (série complète pré-split) |
| `ground_truth_slope_ma.npy` | Pente MA centrée window=21 (série complète) |
| `baseline_slope_{train,val,test}.npy` | Estimations de pente par split |
| `baseline_innovations_train.npy` | Innovations `v_t` (pour diagnostic) |
| `baseline_innov_S_train.npy` | Variance d'innovation `S_t` |
| `baseline_sigma2_train.npy` | Trace de l'adaptation σ² |
| `baseline_metrics.json` | 5 métriques × 2 GT × 2 splits |
| `step1_diagnostic.json` | ACF, LB, JB, verdict gate |
| `step1_plots/` | Histogramme z_t, ACF(50), QQ-plot |
| `report_step1.md` | Rapport Markdown humain |

### Sur stdout

Le script imprime les métriques et **termine par un bloc SUMMARY** contenant
le verdict du gate :

```
======================================================================
SUMMARY
======================================================================
  Verdict GATE  : EXPLOITABLE  (ou MARGINAL / WHITE_NOISE)
  Baseline VAL  (vs RTS): MSE=...  Pearson=...  DirMatch=...
  Baseline TEST (vs RTS): MSE=...  Pearson=...  DirMatch=...
  ...
======================================================================
```

## Gate Étape 1 → Étape 2 (calibré par l'utilisateur)

Décision sur la base de `max|ACF(1..10)|` des innovations normalisées ET
la p-value Ljung-Box à h=10 :

| Critère | Verdict | Action |
|---------|---------|--------|
| `max|ACF| > 0.05` ET `LB p < 0.05` | **EXPLOITABLE** | Structure présente → Étape 2 (3D WNA) justifiée |
| `max|ACF| ∈ [0.02, 0.05]` ET `LB p < 0.05` | **MARGINAL** | Flag → confirmer avant Étape 2 |
| `max|ACF| < 0.02` | **WHITE_NOISE** | Skip Étape 2 quelle que soit la p-value |

Rationale : avec n ~ 200k sur train, un LB p-value seul rejette quasi toujours
H0 même pour une auto-corrélation négligeable. La magnitude de l'ACF est le
critère discriminant pratique.

## Ce que le script NE fait PAS (encore)

- Étape 2 (KF 3D WNA) : à implémenter après validation du gate
- Étape 3 (R adaptatif) : conditionnel au meilleur de Étapes 1/2
- Étape 4 (FLKS lag sweep) : optionnel
- Étape 5 (IMM 2 modèles) : conditionnel à Δ MSE > 1% sur test après Étapes 2+3
- Étape 6 (rapport final synthétique) : livré après les 5 étapes

## Dépendances

- numpy, pandas, scipy (standard)
- pykalman (déjà dans le projet, utilisé par `src/filters.py`) — avec fallback hand-rolled
- matplotlib (optionnel, plots automatiquement skippés si absent)

Aucune installation supplémentaire requise par rapport au projet existant.

## Méthodologie (rappel)

- **Causalité** : toutes les estimations à l'instant t dépendent uniquement
  de y_{1:t}. Exception documentée : le RTS ground truth est non-causal
  par design — utilisé **uniquement comme référence d'évaluation**, jamais
  en input.
- **Split temporel strict** : train=50%, val=25%, test=25%. Pas de shuffle,
  pas de k-fold. Test jamais touché avant la comparaison finale.
- **Ground truth calculé une fois** sur la série complète, puis split
  (contrainte utilisateur).
- **Tuning d'hyperparamètres** (pour les Étapes 3+) uniquement sur `val`.
