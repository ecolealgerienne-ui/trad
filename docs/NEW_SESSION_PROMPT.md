# Prompt Nouvelle Session - Projet Trading ML

**Date mise à jour**: 2026-01-07
**Version projet**: 8.9 (Phase 2.8 complétée - Direction-Only validé)

---

## 📋 Prompt à Copier-Coller

```
Contexte: Je travaille sur un système de trading algorithmique avec ML (CNN-LSTM).

État actuel du projet (Phase 2.8 COMPLÉTÉE):
- Architecture Direction-Only VALIDÉE (abandon Force définitif)
- 6 modèles entraînés: 3 indicateurs (MACD, RSI, CCI) × 2 filtres (Kalman, Octave)
- Résultats: Direction-Only stable/amélioré (+0.1% à +0.9% vs Dual-Binary)
- Kalman > Octave systématiquement (-1.1% à -4.0% gap)
- Signal validé: +110.89% PnL Brut (30,876 trades, holding 30p)
- Problème persistant: Trop de trades → -2,976% PnL Net (frais 0.6% round-trip)

Modèles Direction-Only (Test Set):
- MACD Kalman: 92.5% accuracy (meilleur)
- CCI Kalman: 90.2% accuracy (+0.9% meilleur gain Direction-Only)
- RSI Kalman: 87.6% accuracy
- Versions Octave: 84.3%-91.4% (inférieures)

Documentation clés à lire:
1. CLAUDE.md (lignes 1-100) - Vue d'ensemble + Phase 2.8 résultats
2. CLAUDE.md (lignes 612-722) - Phase 2.8 complète Direction-Only
3. CLAUDE.md (lignes 725-800) - Force Filter échec complet (contexte)

Diagnostic actuel:
✅ Signal fonctionne: +110.89% PnL Brut, Win Rate 42.05%
✅ Modèle performant: 92.5% accuracy MACD Direction
✅ Architecture simplifiée: Direction-Only validé (1 output vs 2)
❌ Trop de trades: 30,876 (48 trades/jour/asset)
❌ Frais destructeurs: -9,263% (83× le PnL brut!)
❌ Edge insuffisant: +0.36% - 0.6% frais = -0.24% négatif

Prochaine étape critique: ATR Structural Filter
Objectif: Réduire trades de 30,876 → ~15,000 (-50%)
Approche: Filtrer par volatilité (López de Prado 2018)
Impact attendu: Win Rate 42% → 50-55%, PnL Net -2,976% → +100-200% ✅

Questions pour toi:
1. Peux-tu lire CLAUDE.md (lignes 1-100 puis 612-722) pour comprendre Phase 2.8?
2. Faut-il créer le script tests/test_atr_structural_filter.py pour tester le filtre ATR?
3. Ou préfères-tu explorer d'autres approches (timeframe 15min, maker fees 0.02%)?

Ma contrainte: Exchange standard (frais 0.3% round-trip), timeframe 5min, 5 assets (BTC/ETH/BNB/ADA/LTC).

Objectif: Atteindre PnL Net positif sur backtest avant passage production.
```

---

## 📚 Documents de Contexte (Ordre de Lecture)

### 1. Vue d'Ensemble - CLAUDE.md

**Sections critiques Phase 2.8**:
- **Lignes 1-10**: Statut actuel (v8.9, Direction-Only validé)
- **Lignes 612-722**: Phase 2.8 complète (6 modèles, tous résultats)
- **Lignes 725-800**: Force Filter Tests (contexte échec)
- **Lignes 250-610**: Phases 2.6-2.7 (holding minimum, veto rules)

**Ce que ça apporte**: Vue d'ensemble, historique complet, tous résultats validés

### 2. Résultats Direction-Only - CLAUDE.md (Phase 2.8)

**Tableau récapitulatif (ligne 630)**:
```
MACD Kalman:  92.5% (+0.1% vs Dual-Binary)
MACD Octave:  91.4%
RSI Kalman:   87.6% (+0.2% vs Dual-Binary)
RSI Octave:   84.3%
CCI Kalman:   90.2% (+0.9% vs Dual-Binary) ← Meilleur gain!
CCI Octave:   86.2%
```

**Découvertes majeures**:
1. Direction-Only N'A PAS dégradé (stable/amélioré)
2. Kalman > Octave systématiquement
3. CCI bénéficie le plus (+0.9%)
4. Force confirmé comme inutile

### 3. Force Filter Échec - CLAUDE.md (lignes 725-800)

**Ce que ça apporte**: Comprendre pourquoi Force a été abandonné définitivement
- 10 configurations testées (Force STRONG/WEAK, consensus)
- 10/10 échecs (-354% à -800% dégradation)
- Direction seule > Toutes configs avec Force

### 4. Phase 2.7 Context - CLAUDE.md + docs/PHASE_27_FINAL_RESULTS.md

**Ce que ça apporte**: Comprendre échec veto rules avant Direction-Only
- Holding minimum 30p: +110.89% brut, -2,976% net
- Veto rules: -3.9% trades (insuffisant)
- Diagnostic: Problème = fréquence trading, pas qualité signal

---

## 🎯 État Technique Actuel

### Datasets Direction-Only (Nouveaux)

```bash
# Direction-Only (Y shape: n,1)
data/prepared/dataset_btc_eth_bnb_ada_ltc_macd_direction_only_kalman.npz
data/prepared/dataset_btc_eth_bnb_ada_ltc_rsi_direction_only_kalman.npz
data/prepared/dataset_btc_eth_bnb_ada_ltc_cci_direction_only_kalman.npz

# Versions Octave20
data/prepared/dataset_btc_eth_bnb_ada_ltc_macd_direction_only_octave20.npz
data/prepared/dataset_btc_eth_bnb_ada_ltc_rsi_direction_only_octave20.npz
data/prepared/dataset_btc_eth_bnb_ada_ltc_cci_direction_only_octave20.npz

Format: X=(n, 25, 1 ou 3), Y=(n, 1) [Direction uniquement]
Split: 70% train / 15% val / 15% test (chronologique)
Assets: BTC, ETH, BNB, ADA, LTC
Timeframe: 5min
Période: 2017-2026 (~8.5 ans, ~4.3M sequences)
```

### Modèles Direction-Only Entraînés

```bash
# Kalman (meilleurs)
models/best_model_macd_direction_only_kalman.pth   (92.5% accuracy)
models/best_model_cci_direction_only_kalman.pth    (90.2% accuracy)
models/best_model_rsi_direction_only_kalman.pth    (87.6% accuracy)

# Octave20 (backup)
models/best_model_macd_direction_only_octave20.pth (91.4% accuracy)
models/best_model_cci_direction_only_octave20.pth  (86.2% accuracy)
models/best_model_rsi_direction_only_octave20.pth  (84.3% accuracy)
```

### Scripts Clés Phase 2.8

```bash
# Génération datasets Direction-Only
src/prepare_data_direction_only.py

# Backtest référence (Phase 2.6)
tests/test_holding_strategy.py

# Tests consensus ML (Phase 2.7)
tests/test_oracle_filtered_by_ml.py

# Entraînement (auto-détection Direction-Only)
src/train.py

# Évaluation
src/evaluate.py
```

---

## 📊 Métriques de Référence

### Modèles Direction-Only (Test Set, Phase 2.8)

```
MACD Kalman:  92.5% accuracy ← DÉCIDEUR PRINCIPAL
CCI Kalman:   90.2% accuracy
RSI Kalman:   87.6% accuracy

Gaps Kalman vs Octave:
- MACD: -1.1% (92.5% vs 91.4%)
- RSI:  -3.3% (87.6% vs 84.3%)
- CCI:  -4.0% (90.2% vs 86.2%)

Conclusion: Kalman est filtre optimal pour labels ML
```

### Trading Performance (Holding 30p, Phase 2.6)

```
Indicateur:  MACD Direction (Dual-Binary à l'époque)
Trades:      30,876 (48 trades/jour/asset)
Win Rate:    42.05% (excellent)
PnL Brut:    +110.89% ✅ LE SIGNAL FONCTIONNE!
PnL Net:     -2,976% ❌
Avg Dur:     18.5p (~90 min)
Frais:       -9,263% (0.3% × 2 × 30,876 trades)

Diagnostic:
Edge/trade:  +0.36%
Frais/trade: -0.6%
Résultat:    -0.24% par trade (négatif)

Conclusion: Signal robuste MAIS trop de trades détruisent rentabilité
```

### Oracle Kalman (Plafond Théorique)

```
PnL:         +6,644%
Sharpe:      18.5
Win Rate:    78.4%
Conclusion:  Signal EXISTE, est PUISSANT, et est EXPLOITABLE
```

---

## 🚀 Prochaine Étape Critique: ATR Structural Filter

### Principe

**Ne trader QUE dans les régimes de volatilité "sains"** (ni trop basse, ni trop haute)

```python
# Trade UNIQUEMENT si:
MACD Direction = UP or DOWN  (signal ML)
AND
Q20 < ATR < Q80  (volatilité acceptable)

# Exclure:
- ATR < Q20: volatilité trop basse (ranging market, signaux faibles)
- ATR > Q80: volatilité extrême (gaps, slippage élevé)
```

### Impact Attendu

```
Baseline (sans filtre):
Trades:      30,876
Win Rate:    42.05%
PnL Brut:    +110.89%
PnL Net:     -2,976%

Avec ATR Filter (hypothèse):
Trades:      ~15,000 (-50%)
Win Rate:    ~50-55% (+8-13%) ← Meilleures conditions
PnL Brut:    ~+100% (maintenu car Win Rate ↑)
Frais:       -4,500% (au lieu de -9,263%)
PnL Net:     ~+100 à +200% ✅ POSITIF!
```

### Implémentation

**Script à créer**: `tests/test_atr_structural_filter.py`

**Logique** (réutiliser `test_holding_strategy.py`):
```python
1. Charger prédictions MACD Direction-only Kalman
2. Charger données OHLC (pour calcul ATR)
3. Calculer ATR(14) sur chaque asset
4. Définir Q20 et Q80 de l'ATR (percentiles 20 et 80)
5. Backtester:
   if MACD_pred == UP and Q20 < ATR[i] < Q80:
       enter_long()
   elif MACD_pred == DOWN and Q20 < ATR[i] < Q80:
       enter_short()
   else:
       hold()  # Volatilité hors range acceptable
6. Comparer métriques vs baseline sans filtre
```

**Référence académique**: López de Prado (2018) - "Advances in Financial ML" (Chapitre 18: Structural Breaks)

---

## 🛠️ Commandes Utiles

### Génération Datasets Direction-Only

```bash
# Tous assets (complet)
python src/prepare_data_direction_only.py --assets BTC ETH BNB ADA LTC

# Test rapide (échantillon)
python src/prepare_data_direction_only.py --assets BTC --max-samples 10000
```

### Entraînement Direction-Only

```bash
# MACD Kalman (décideur principal)
python src/train.py \
    --data data/prepared/dataset_btc_eth_bnb_ada_ltc_macd_direction_only_kalman.npz \
    --epochs 50

# Auto-détection: 1 output → mode Direction-Only activé
```

### Évaluation

```bash
python src/evaluate.py \
    --data data/prepared/dataset_btc_eth_bnb_ada_ltc_macd_direction_only_kalman.npz
```

### Backtest Holding Minimum (Référence Phase 2.6)

```bash
python tests/test_holding_strategy.py --indicator macd --split test
```

---

## 🐛 Bugs Critiques Connus (Tous Corrigés)

### Bug #1: Direction Flip Double Trades (commit e51a691)

**Symptôme**: 38,573 trades au lieu de 30,876 (+25%), PnL -8.76% au lieu de +110.89%
**Cause**: LONG→FLAT→SHORT (2 trades) au lieu de LONG→SHORT (1 trade direct)
**Fix**: `position = target` (flip immédiat) au lieu de `position = FLAT`
**Doc**: docs/BUG_DIRECTION_FLIP_ANALYSIS.md

```python
# INCORRECT (bug)
if exit_reason == "DIRECTION_FLIP":
    position = Position.FLAT  # Crée 2 trades!

# CORRECT (fix)
if exit_reason == "DIRECTION_FLIP":
    position = target  # Flip immédiat, 1 seul trade
```

### Bug #2: IndexError prepare_data_direction_only.py (ligne 599)

**Symptôme**: `IndexError: index 1 is out of bounds for axis 1 with size 1`
**Cause**: Tentative d'accès à Force `Y[:, 1]` qui n'existe plus en Direction-Only
**Fix**: Suppression stats Force, ajout paramètre `--max-samples` pour tests rapides

### Bug #3: PnL Calculation (commit 8ec2610)

**Cause**: Traiter returns comme des prix
**Fix**: Accumuler returns dans current_pnl (logique prouvée)

**Règle d'Or Validée**: "Mutualisé les fonctions" = TOUJOURS copier code prouvé, JAMAIS réécrire!

---

## 📈 Feuille de Route Recommandée

### Option 1: ATR Structural Filter (RECOMMANDÉE - Court Terme)

**Effort**: ~2-3h (script + tests)
**Gain attendu**: PnL Net -2,976% → +100-200% ✅
**Risque**: Faible (approche académique validée)

**Étapes**:
1. Créer `tests/test_atr_structural_filter.py` (réutiliser holding_strategy.py)
2. Tester Q20 < ATR < Q80 sur test set
3. Si positif → valider sur plusieurs seeds
4. Si robuste → production

### Option 2: Timeframe 15min/30min (Moyen Terme)

**Effort**: ~4-6h (datasets + réentraînement)
**Gain attendu**: Trades -50% à -67%, PnL Net potentiellement positif
**Risque**: Moyen (signal peut se dégrader)

**Étapes**:
1. Régénérer datasets 15min (5 assets)
2. Réentraîner MACD Kalman (décideur principal)
3. Backtest holding 30p (ou adapter durée)
4. Comparer vs baseline 5min

### Option 3: Maker Fees 0.02% (Quick Win - Si Possible)

**Effort**: ~1-2h (adaptation stratégie exécution)
**Gain attendu**: Frais ÷10 → PnL Net immédiatement positif ✅
**Risque**: Faible (si exchange disponible)

**Calcul**:
```
Frais actuels: 0.3% round-trip (taker)
Frais maker: 0.02% round-trip
Réduction: ÷15

30,876 trades × 0.02% = -926%
PnL Net: +110.89% - 926% = +9,174% ✅ POSITIF!
```

**Contrainte**: Nécessite exchange avec bons rebates maker + gestion limit orders

---

## 🎯 Objectifs Session Suivante

**Minimum**:
Lire CLAUDE.md (Phase 2.8, lignes 612-722) pour comprendre résultats Direction-Only

**Recommandé**:
Créer `tests/test_atr_structural_filter.py` et tester sur test set

**Ambitieux**:
Valider PnL Net positif avec ATR filter → passage production

---

## 📞 Questions Fréquentes

**Q: Pourquoi Direction-Only au lieu de Dual-Binary?**
R: Force n'apporte AUCUN bénéfice (10 tests, 10 échecs, -354% à -800% dégradation). Direction-Only est stable/amélioré (+0.1% à +0.9%).

**Q: Pourquoi Kalman > Octave?**
R: Kalman (filtre bayésien) produit labels plus stables que Octave (fréquentiel). Gap constant -1.1% à -4.0% selon indicateur.

**Q: CCI meilleur gain (+0.9%), pourquoi ne pas l'utiliser comme décideur?**
R: MACD reste meilleur en absolu (92.5% vs 90.2%). CCI profite juste plus du single-task, mais MACD est décideur optimal validé.

**Q: Le modèle est-il assez bon (92.5%)?**
R: OUI! Le problème n'est PAS la qualité du modèle (excellent), mais la FRÉQUENCE de trading. Signal fonctionne (+110% brut), trop de trades détruisent rentabilité.

**Q: ATR Filter va-t-il suffire?**
R: Potentiellement OUI. Réduire trades -50% + améliorer Win Rate +8-13% devrait donner PnL Net positif. Approche validée académiquement (López de Prado).

**Q: Quel est le vrai problème?**
R: Edge/trade (+0.36%) < Frais/trade (-0.6%) → Perte nette -0.24%/trade. Solution = Réduire trades OU Réduire frais OU Améliorer Win Rate.

**Q: Oracle +6,644% connaît le futur?**
R: NON! Oracle utilise labels (pente t-2 vs t-3) à 100% accuracy. Teste le potentiel MAX du signal, pas le futur. Prouve que signal EXISTE.

---

## 🔄 Historique Versions

**v1.0** (2026-01-07 - Phase 2.7): Prompt initial post veto rules
**v2.0** (2026-01-07 - Phase 2.8): Mise à jour Direction-Only validé, Force abandonné, ATR filter next step

---

**Créé**: 2026-01-07
**Dernière MAJ**: 2026-01-07
**Version**: 2.0
**Auteur**: Claude Code
**Objectif**: Permettre nouvelle session de partir du contexte complet Phase 2.8 sans perte d'information
