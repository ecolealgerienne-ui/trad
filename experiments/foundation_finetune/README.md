# Foundation Model Fine-Tuning — RSI → Oracle Slope (causal)

Fine-tuning d'un modèle de fondation pour séries temporelles (Lag-Llama) afin de
reconstruire en **causal** la pente de l'Oracle Kalman du RSI.

## Spécifications

- **Asset / TF** : BTC 5min (`data_trad/BTCUSD_all_5m.csv`)
- **Indicateur** : RSI(22) (`RSI_PERIOD` du projet)
- **Window d'entrée** : 96 valeurs `RSI[t-95:t]`
- **Cible (régression scalaire)** : `slope[t] = Oracle[t] - Oracle[t-1]`
- **Oracle** : Kalman 1D + RTS smoother (`src.filters.kalman_filter`,
  `Q=0.01`, `R=0.1` par défaut, paramétrables)
- **Anti-leakage RTS** : Oracle calculé séparément par split
  - `oracle_train` ne voit que `rsi[:n_train]`
  - `oracle_val` voit `rsi[:n_train+n_val]`
  - `oracle_test` voit `rsi[:n]` (l'Oracle "vérité ultime")
- **Split temporel** : 70/15/15
- **Backbone** : Lag-Llama (HuggingFace), fine-tune via LoRA
- **Loss** : MSE sur la pente
- **Métriques** : MSE, MAE, DirMatch (signe pente), Pearson, lag CCF

## Structure

```
experiments/foundation_finetune/
├── README.md                # ce fichier
├── __init__.py
├── build_dataset.py         # Phase 1 : RSI → Oracle slope, .npz par split
├── baselines.py             # Phase 2 : identité, pente RSI brut, MA causale
├── model.py                 # Phase 3 : Lag-Llama + tête régression + LoRA
├── train.py                 # Phase 4 : fine-tuning MSE
└── evaluate.py              # Phase 5 : métriques + comparaison baselines
```

Sortie data : `data/foundation/rsi_btc_5min_slope.npz`

## Réutilisation projet

Le script `build_dataset.py` réutilise (lecture seule) :
- `src.constants` — `BTC_DATA_FILE_5M`, `RSI_PERIOD`, `KALMAN_PROCESS_VAR`,
  `KALMAN_MEASURE_VAR`
- `src.data_utils.load_crypto_data` — charge le CSV
- `src.indicators.calculate_rsi` — calcul RSI
- `src.filters.kalman_filter` — KF 1D random walk + RTS smoother (pykalman)

## Pipeline reproductible

```bash
# Phase 1 : préparer le dataset (~5-10 min)
python experiments/foundation_finetune/build_dataset.py

# Phase 2 : baselines (référence avant fine-tune)
python experiments/foundation_finetune/baselines.py

# Phase 3-4 : fine-tune Lag-Llama (LoRA, ~2-4h GPU)
python experiments/foundation_finetune/train.py

# Phase 5 : évaluation
python experiments/foundation_finetune/evaluate.py
```

## Caveat — calibration σ²

Q=0.01 (défaut `src/filters.py`) est documenté dans `CLAUDE.md` (section
slope_improvement) comme structurellement bas pour le RSI. Le projet
slope_improvement a identifié σ²=1.155 (MLE) comme optimum labels-ML.

Pour cette session : on part sur **Q=0.01 (défaut historique)** par décision
explicite de l'utilisateur ("oublie nos précédents travaux"), paramétrable
via `--process-var` si besoin de tester d'autres valeurs.
