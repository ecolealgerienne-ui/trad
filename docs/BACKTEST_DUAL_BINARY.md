# Guide Backtest Dual-Binary

## Vue d'ensemble

Le script `tests/test_dual_binary_trading.py` teste la stratégie de trading simple basée sur l'architecture Dual-Binary (Direction + Force).

## Architecture Dual-Binary

Chaque indicateur prédit **2 outputs binaires** :
- **Direction** : UP (1) ou DOWN (0)
- **Force** : STRONG (1) ou WEAK (0)

## Stratégie Simple (Decision Matrix)

```python
if Direction == UP and Force == STRONG:
    → LONG
elif Direction == DOWN and Force == STRONG:
    → SHORT
else:
    → HOLD (filtrer signaux WEAK)
```

## Prérequis

### 1. Données Préparées

```bash
# Générer les 3 datasets dual-binary
python src/prepare_data_purified_dual_binary.py --assets BTC ETH BNB ADA LTC
```

**Outputs** :
- `data/prepared/dataset_btc_eth_bnb_ada_ltc_rsi_dual_binary_kalman.npz`
- `data/prepared/dataset_btc_eth_bnb_ada_ltc_macd_dual_binary_kalman.npz`
- `data/prepared/dataset_btc_eth_bnb_ada_ltc_cci_dual_binary_kalman.npz`

### 2. Modèles Entraînés (Optionnel)

Pour tester avec les **prédictions modèle** (au lieu des labels Oracle) :

```bash
# Entraîner les 3 modèles
python src/train.py --data data/prepared/dataset_btc_eth_bnb_ada_ltc_macd_dual_binary_kalman.npz --epochs 50
python src/train.py --data data/prepared/dataset_btc_eth_bnb_ada_ltc_cci_dual_binary_kalman.npz --epochs 50
python src/train.py --data data/prepared/dataset_btc_eth_bnb_ada_ltc_rsi_dual_binary_kalman.npz --epochs 50

# Générer les prédictions (sauvegarde Y_pred dans .npz)
python src/evaluate.py --data data/prepared/dataset_btc_eth_bnb_ada_ltc_macd_dual_binary_kalman.npz
python src/evaluate.py --data data/prepared/dataset_btc_eth_bnb_ada_ltc_cci_dual_binary_kalman.npz
python src/evaluate.py --data data/prepared/dataset_btc_eth_bnb_ada_ltc_rsi_dual_binary_kalman.npz
```

## Usage

### Test 1 : Labels Oracle (Monde Parfait)

Tester avec les **labels réels** (monde parfait, accuracy 100%) :

```bash
# MACD (recommandé: meilleur indicateur 86.9%)
python tests/test_dual_binary_trading.py --indicator macd --split test --fees 0.1

# CCI (2ème meilleur: 83.3%)
python tests/test_dual_binary_trading.py --indicator cci --split test --fees 0.1

# RSI (3ème: 80.7%)
python tests/test_dual_binary_trading.py --indicator rsi --split test --fees 0.1
```

### Test 2 : Prédictions Modèle (Réaliste)

Tester avec les **prédictions du modèle** (accuracy réelle ~80-87%) :

```bash
# MACD avec prédictions
python tests/test_dual_binary_trading.py \\
    --indicator macd \\
    --split test \\
    --use-predictions \\
    --fees 0.1

# CCI avec prédictions
python tests/test_dual_binary_trading.py \\
    --indicator cci \\
    --split test \\
    --use-predictions \\
    --fees 0.1
```

### Paramètres

| Paramètre | Description | Valeurs | Défaut |
|-----------|-------------|---------|--------|
| `--indicator` | Indicateur à tester | `rsi`, `macd`, `cci` | **Requis** |
| `--split` | Split à tester | `train`, `val`, `test` | `test` |
| `--fees` | Frais par trade (%) | Float | `0.1` (0.1%) |
| `--use-predictions` | Utiliser prédictions modèle | Flag | `False` (Oracle) |

## Interprétation des Résultats

### Métriques Clés

| Métrique | Description | Objectif |
|----------|-------------|----------|
| **Total Trades** | Nombre de trades fermés | Optimal: ~20-40k (vs ~100k sans filtrage Force) |
| **Win Rate** | % de trades gagnants | Objectif: >50% (vs ~42% sans Force) |
| **Profit Factor** | sum(wins) / abs(sum(losses)) | Objectif: >1.1 |
| **PnL Net** | Rendement après frais | Positif si stratégie rentable |
| **Avg Duration** | Périodes par trade | Plus long = moins de frais |

### Exemple Output

```
📊 RÉSULTATS BACKTEST - MACD (Oracle)
======================================================================

📈 Trades:
  Total Trades:     22,000
  LONG:             11,200
  SHORT:            10,800
  HOLD (filtered):  ~70,000 (70% des signaux filtrés)
  Avg Duration:     33.3 périodes

💰 Performance:
  Win Rate:         55.00%
  Profit Factor:    1.15
  Avg Win:          +0.450%
  Avg Loss:         -0.300%

💵 PnL:
  PnL Brut:         +1348.00%
  Frais Totaux:     -44.00%
  PnL Net:          +1304.00%
======================================================================
```

### Comparaison Attendue Oracle vs Modèle

| Mode | Total Trades | Win Rate | PF | PnL Net | Notes |
|------|--------------|----------|-----|---------|-------|
| **Oracle MACD** | ~22k | ~55% | ~1.15 | **+1300%** | Monde parfait (accuracy 100%) |
| **Modèle MACD** | ~22k | ~48% | ~1.08 | **+500-800%** | Accuracy réelle 86.9% |
| **Oracle CCI** | ~25k | ~52% | ~1.12 | **+1100%** | Monde parfait |
| **Modèle CCI** | ~25k | ~46% | ~1.06 | **+400-600%** | Accuracy réelle 83.3% |

**Gain attendu du Force Filtering** :
- Trades: **-60% à -80%** (filtrage des signaux WEAK)
- Win Rate: **+8% à +13%** (qualité > quantité)
- Profit Factor: **+0.08 à +0.12**

## Impact du Force Filtering

### Sans Force (Direction seule)

```
Total Trades: ~100,000
Win Rate: ~42%
Profit Factor: ~1.03
PnL Net: Négatif (frais > edge)
```

### Avec Force (Decision Matrix)

```
Total Trades: ~22,000 (-78%)
Win Rate: ~55% (+13%)
Profit Factor: ~1.15 (+0.12)
PnL Net: +1300% (positif!)
```

**Le filtrage Force réduit les trades de 78% mais améliore la qualité de 13%.**

## Tests Recommandés

### Séquence de Tests

1. **Test Oracle MACD** (baseline performance maximale)
   ```bash
   python tests/test_dual_binary_trading.py --indicator macd --split test --fees 0.1
   ```

2. **Test Modèle MACD** (performance réaliste)
   ```bash
   python tests/test_dual_binary_trading.py --indicator macd --split test --use-predictions --fees 0.1
   ```

3. **Comparer Oracle vs Modèle** (gap = marge d'amélioration)

4. **Tester CCI et RSI** (comparaison indicateurs)

### Analyse de Sensibilité

Tester différents niveaux de frais :

```bash
# Frais faibles (Maker fees)
python tests/test_dual_binary_trading.py --indicator macd --split test --fees 0.02

# Frais standards (Taker fees)
python tests/test_dual_binary_trading.py --indicator macd --split test --fees 0.1

# Frais élevés (slippage inclus)
python tests/test_dual_binary_trading.py --indicator macd --split test --fees 0.2
```

## Prochaines Étapes

Si les résultats Oracle sont positifs :

1. ✅ **Valider MACD** comme indicateur principal
2. ✅ **Mesurer gap Oracle/Modèle** (marge d'amélioration)
3. 🔄 **Implémenter State Machine** avec règles combinées (MACD + CCI + RSI)
4. 🔄 **Optimiser Hysteresis** pour réduire micro-sorties
5. 🔄 **Tester timeframes** (15min, 30min)

## Notes Importantes

### Limitations

- **PnL non composé** : Calcul en rendement simple (somme)
- **Pas de slippage** : Prix d'exécution = Close exact
- **Frais fixes** : Pas de variation selon taille position
- **1 asset à la fois** : Pas de diversification

### Améliorations Futures

- [ ] Calcul PnL composé
- [ ] Simulation slippage
- [ ] Sizing dynamique (% capital)
- [ ] Multi-asset portfolio
- [ ] Drawdown analysis
- [ ] Sharpe ratio
- [ ] Sortino ratio

## Références

- **CLAUDE.md** : Architecture Dual-Binary complète
- **docs/SPEC_ARCHITECTURE_IA.md** : Spécifications modèle
- **src/prepare_data_purified_dual_binary.py** : Génération datasets
- **src/train.py** : Entraînement modèles (auto-détection config)
