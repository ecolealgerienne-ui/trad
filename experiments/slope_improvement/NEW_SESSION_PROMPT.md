# Prompt d'ouverture — Nouvelle session post-slope_improvement

## Contexte du projet

Tu reprends une session après le projet `slope_improvement` (branche `claude/improve-kalman-filter-bMDBP`). Ce projet a été conclu avec une synthèse à 4 régimes opérationnels pour l'estimation causale de la pente du RSI filtré par Kalman.

## État actuel — projet conclu, branche prête

- **Branche** : `claude/improve-kalman-filter-bMDBP`
- **Working tree** : clean (tout est committé et poussé)
- **Doc principale** : `experiments/slope_improvement/final_report.md` (8 sections, ~600 lignes)
- **Doc projet** : `CLAUDE.md` § "Consolidation AQ-KF + MLE (slope_improvement, 2026-04-24)"

## Findings principaux à connaître

### 1. Le `KALMAN_PROCESS_VAR = 0.01` du projet est structurellement trop bas

Sur RSI 30min BTC, le régime σ²=0.01 produit :
- **45.34% concordance transitions à k=0 (FLKS Test 1)** — anti-prédictif (pire que hasard)
- 55.52% à T0 (forward pur strict)
- Cause probable du gap Phase 2.10 (CNN-LSTM transition accuracy ~58%)

### 2. Quatre régimes σ² complémentaires (pas un seul gagnant universel)

| Régime opérationnel | Latence | Pipeline recommandé | σ² effectif |
|---|---|---|---|
| **HFT / signaux instantanés** | 0 min | **C2 — AQ-KF unlocked** (clip [0.001, 10]) | adaptive, mean=0.11, P95=0.17 |
| **Quasi temps-réel** | 5-30 min | **C1 — AQ-KF historique** (clip [0.001, 0.1]) | adaptive, ~0.10 |
| **Latence acceptable** | 30 min | C1 ≈ B (équivalents) | — |
| **Latence longue** | ≥30 min | **B — MLE fixed** | 1.155 |
| **Labels ML** | — | **B — MLE fixed** (σ²=1.155, R=3.27) | 1.155 |

### 3. Myers-Tapley ≠ MLE (théorique important)

L'AQ-KF Myers-Tapley converge naturellement à `σ² ≈ 0.11` (method-of-moments local optimum), même sans clipping. Le MLE trouve `σ² = 1.155` (likelihood global optimum). **Deux estimateurs différents, deux optima distincts.**

### 4. Anomalie courbe en U du `σ²=0.01` historique

Pour A (historique fixe), la concordance transitions présente un MIN à k=0 (45.34%) — pire que T0 (55.52%) ou k≥1. Le RTS backward sur un filtre déjà sur-lissé l'amplifie encore. **Pathologie spécifique au régime σ²=0.01**.

### 5. Comparaison labels OLD vs NEW (test compare_label_calibration)

- Agreement rate global : **90.19%** (10% labels différents)
- σ²=0.01 vs σ²=1.155 → labels Phase 2.15 different sur les transitions principalement
- Verdict : MARGINAL (à la frontière 10%)
- Implication : **TIMING des prédictions change** (~30 min de moins de lag), **pas forcément accuracy**

## Architecture du module `experiments/slope_improvement/`

### Scripts pipelining (ordre méthodologique)

```
experiments/slope_improvement/
├── data_loader.py            # BTC 5min → 2022+ → RSI(22) → split 50/25/25
├── kf_baseline.py            # KF 2D adaptive σ² Myers-Tapley scalaire (Étape 1)
├── kf_nd.py                  # Toolkit N-D générique (forward, RTS, NLL, MLE, AIC/BIC)
├── gt_3d.py                  # 3D WNA pour GT (forward, RTS, NLL, MLE)
├── gt_4d.py                  # 4D constant-jerk (rejeté mais conservé)
├── ground_truth.py           # RTS 2D fixed + MA centrée (réutilisé)
├── flks.py                   # Fixed-Lag Smoother vectorisé N-D (Étape 4)
├── estimate_R.py             # 4 estimateurs empiriques R + sélection
├── diagnostics.py            # ACF, Ljung-Box (manuel), Jarque-Bera, plots
├── metrics.py                # MSE, MAE, Pearson, DirMatch, latency, Diebold-Mariano
├── run_experiments.py        # Étape 0+1 orchestrateur
├── investigate_bounds.py     # Étape B.1 (sigma bounds sweep)
├── validate_gt_and_R.py      # Étape B.4 (GT 3D officiel + recalibration R via MLE)
├── validate_gt_4d.py         # Étape B.4b (GT 4D validation, rejeté via BIC)
├── sanity_mle_4d.py          # 18 runs MLE 4D pour exclure minimum local
├── finalize_2d_baseline.py   # Étape B.5 (baseline 2D MLE fixed)
├── etape2_multi_ref.py       # Étape 2 (2D vs 3D MLE multi-refs)
├── etape4_flks_sweep.py      # Étape 4 (FLKS sweep → FLKS(lag=3))
├── flks_substep_mle.py       # Sub-step convergence A/B/C1/C2 (T0 + k=0..6)
├── compare_label_calibration.py  # OLD vs NEW labels Phase 2.15 agreement
├── final_report.md           # 8 sections, synthèse complète
└── artifacts/                # JSON + .npy + plots, 25-30 min pour reproduire
```

### Pipeline complet reproductible

```bash
cd experiments/slope_improvement/
python run_experiments.py                    # Étape 0+1 (baseline + diagnostic)
python investigate_bounds.py                 # Étape B.1
python validate_gt_and_R.py                  # Étape B.4 (GT 3D + R MLE)
python validate_gt_4d.py                     # Étape B.4b (GT 4D rejeté)
python sanity_mle_4d.py                      # Sanity MLE 4D
python finalize_2d_baseline.py               # Étape B.5 (baseline 2D MLE)
python etape2_multi_ref.py                   # Étape 2 (2D vs 3D)
python etape4_flks_sweep.py                  # Étape 4 (FLKS sweep)
python flks_substep_mle.py                   # AQ-KF + MLE 4-way comparison
python compare_label_calibration.py          # OLD vs NEW labels test
```

Durée totale : ~30-40 min sur machine GPU.

## Findings chiffrés à TEST set (récapitulatif)

| Variante | MSE val | Pearson val | DirMatch val | Latence |
|---|---|---|---|---|
| Baseline original (adaptive R=0.1) | 2.0496 | 0.5990 | 0.7029 | +1 |
| Baseline 2D MLE fixed | 1.3427 | 0.6428 | 0.7229 | +2 |
| Baseline 3D MLE fixed (rejeté) | 2.06 | 0.65 | 0.73 | +1 |
| **FLKS(lag=3) sur 2D MLE** | **0.4040** | **0.8608** | **0.8300** | +3 |
| FLKS(lag=∞) sur 2D MLE | 0.398 | 0.870 | 0.836 | non-causal |

**Pour latence ≥15 min** : FLKS(lag=3) = pipeline retenu, gain MSE −80% vs baseline original.

## Pistes non explorées (documentées dans final_report.md §7)

- R adaptatif Myers-Tapley sur top de FLKS (Étape 3 originale, sautée)
- IMM (Interacting Multiple Models) avec mean-reverting (Étape 5 originale, sautée)
- Multi-timeframe (Clock-Injected fashion)
- Transfert à autres indicateurs (CCI, MACD)
- UKF pour RSI borné [0, 100]
- Réentraînement CNN-LSTM avec labels MLE-calibrés (gap estimé +2-3pp accuracy ou ~30 min de moins de lag selon la métrique)

## Caveats méthodologiques connus

1. **MLE 2D fitté avec Q diagonale** mais baseline utilise Q rank-1 (G=[1,1]ᵀ) — std(z)=1.10 confirme décalage marginal acceptable
2. **GT 3D circulaire partielle** : le baseline 3D MLE partage la famille avec le GT 3D — atténué par GT 4D + MA51 comme refs secondaires
3. **MLE statique** : refit sur 20k premiers samples train, pas de ré-estimation périodique
4. **Test sur RSI 5min BTC 2022+ uniquement** : pas validé sur autres timeframes/assets/périodes

## Si la nouvelle session porte sur...

### "Compléter le projet slope_improvement"
- Possible : tester Étape 3 (R adaptatif sur 2D MLE) ou Étape 5 (IMM)
- Possible : refit MLE par fenêtre glissante (3-6 mois) pour capturer les drifts
- Possible : étendre le test sub-step à CCI et MACD pour confirmer généralisation

### "Migrer le finding vers la production"
- Modifier `src/constants.py` pour ajouter `KALMAN_PROCESS_VAR_MLE = 1.155`, `KALMAN_MEASURE_VAR_MLE = 3.27` (sans toucher les anciens)
- Régénérer les datasets meta-labels Phase 2.15 avec calibration MLE
- Réentraîner RSI Kalman dual-binary CNN-LSTM avec nouveaux labels
- Mesurer gap accuracy + gap PnL backtest

### "Nouveau projet"
- Garder cette branche close, démarrer sur une autre branche `claude/<nom-nouveau-projet>`
- Référencer `final_report.md` et CLAUDE.md section AQ-KF/MLE pour le contexte hérité

## Instructions de continuité

1. **Lire d'abord** `experiments/slope_improvement/final_report.md` (synthèse complète à jour)
2. **Si nouveau test sur le module** : ajouter à `experiments/slope_improvement/`, ne JAMAIS toucher aux scripts existants ni à `src/`
3. **Convention de commit** : message clair, lien `https://claude.ai/code/session_<id>` à la fin
4. **Convention de doc** : update `final_report.md` (§ correspondante) ET `CLAUDE.md` (section AQ-KF/MLE) si finding majeur
5. **Stop hook** : commit + push à chaque ajout de fichier (le repo bloque sur untracked)

## Question à clarifier au début de la nouvelle session

- L'objectif de la nouvelle session est-il :
  - (a) **étendre slope_improvement** (Étape 3, IMM, multi-indicateur, refit glissant) ?
  - (b) **migrer en production** (modifier constants, regénérer datasets, réentraîner) ?
  - (c) **un autre projet** (et juste utiliser slope_improvement comme contexte) ?

L'utilisateur peut donner cet objectif explicitement, ou le déduire en posant la question.
