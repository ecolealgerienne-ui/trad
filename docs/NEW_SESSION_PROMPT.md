# 🚀 Prompt Nouvelle Session - Meta-Labeling Phase 2.18

**Date**: 2026-01-10
**Version**: 10.1 - Phase 2.18: Meta-Model Backtest et Aligned Labels
**Branch Git**: `claude/review-context-update-main-844S0`
**Commit Actuel**: `a74abec` - Script aligned meta-labels créé

---

## 📋 Contexte à Charger

Bonjour Claude,

Je continue le projet **CNN-LSTM Direction-Only avec Meta-Labeling**. **Lis d'abord `/home/user/trad/CLAUDE.md`** pour le contexte complet, particulièrement les sections:
- Phase 2.15 (Nouvelle formule labels t vs t-1 - SUCCÈS)
- Phase 2.17 (Meta-model baseline - Logistic Regression)
- **Phase 2.18 (IMPORTANT - Diagnostic problème architecture)**

---

## 🎯 État Actuel - Phase 2.18 Meta-Model Backtest

### Situation Critique Identifiée

**PROBLÈME FONDAMENTAL**: Meta-model prédit selon Triple Barrier, backtest calcule selon Signal Reversal

#### Résultats Backtest Après Corrections Bugs

| Stratégie | Trades | Filtrés | Win Rate | PnL Net | Observation |
|-----------|--------|---------|----------|---------|-------------|
| **Baseline (no filter)** | 108,702 | 0 | 22.49% | **-21,382%** | Référence catastrophique |
| **Meta-Filter (0.5)** | 76,881 | 210,115 | 22.32% | -14,924% | -29% trades, WR stable |
| **Meta-Filter (0.6)** | 40,315 | 476,449 | **20.34%** ❌ | -7,790% | Win Rate **BAISSE** |
| **Meta-Filter (0.7)** | 16,277 | 602,131 | **19.22%** ❌ | -3,034% | Win Rate **BAISSE** encore |

**OBSERVATION CRITIQUE**: Plus on filtre, plus le Win Rate **EMPIRE** au lieu de s'améliorer!

#### Bugs Corrigés (Commits Précédents)

1. **✅ Bug Fees ×100** (Commit `4815ba9`):
   ```python
   # AVANT (bug)
   total_fees = 2 * fees * 100  # 0.001 * 100 = 0.1 = 10%!

   # APRÈS (corrigé)
   total_fees = 2 * fees  # 0.001 = 0.1%
   ```

2. **✅ Bug Trading Logic Fatal** (Commit `ea672e8`):
   ```python
   # AVANT (bug - ne sortait JAMAIS)
   if position != Position.FLAT and meta_prob <= threshold:
       continue  # ❌ Bloque exit quand signal change

   # APRÈS (corrigé - Option B: FLAT autorisé)
   if position == Position.FLAT:
       if meta_prob > threshold:
           position = target
   elif position != target:
       # TOUJOURS sortir si signal change
       exit_trade()
       position = target  # Flip immédiat
   ```

### Diagnostic Expert - Mismatch Architecture Fondamental

**Citation Expert**:
> "Le problème NE vient pas du méta-modèle. Il vient AVANT."
>
> "Un meta-model ne transforme jamais un modèle perdant en modèle gagnant."
> — López de Prado

#### Le Meta-Modèle Fonctionne Techniquement

| Métrique | Valeur | Interprétation |
|----------|--------|----------------|
| **Precision** | **68.41%** | ✅ Niveau institutionnel |
| **ROC AUC** | 0.5846 | ✅ Signal détectable |
| **F1-Score** | 0.5703 | ✅ Balance OK |
| **confidence_spread** | **+2.6584** | ✅ 10× autres features (valide théorie) |

**Découverte Majeure Validée**:
> "Le meilleur trade n'est PAS celui où les modèles sont d'accord, mais celui où ils sont en conflit."

#### Mais Il Prédit la Mauvaise Chose!

**Le Mismatch**:

| Aspect | Triple Barrier (meta-labels) | Backtest Réel |
|--------|------------------------------|---------------|
| **Sortie** | Barrières prix + duration | Changement signal |
| **PnL** | (exit - entry) avec barrières | (exit - entry) au signal change |
| **Duration** | Contrainte min_duration=5 | Variable selon signal |
| **Exits** | 3 conditions (TP, SL, time) | 1 condition (signal flip) |

**Explication du Problème**:
```
Meta-modèle apprend:
  "Ce trade sera profitable selon Triple Barrier"
  (avec barrières fixes et contraintes de durée)

Backtest calcule:
  "Ce trade est profitable selon signal reversal"
  (sortie immédiate quand direction change)

→ Le meta-modèle filtre les "mauvais" trades selon Triple Barrier
→ Mais ces trades peuvent être BONS selon la vraie stratégie
→ Résultat: Filtrage INVERSE (Win Rate baisse au lieu de monter)
```

#### Pourquoi le Win Rate Diminue

Le meta-modèle avec Precision 68.41% dit:
- "68% des trades que je recommande sont profitables... **selon Triple Barrier**"

Mais le backtest utilise une logique différente:
- Trades recommandés peuvent être **perdants dans le backtest réel**
- Trades rejetés peuvent être **gagnants dans le backtest réel**

**Résultat**: Le filtrage sélectionne les MAUVAIS trades du point de vue du backtest.

---

## ✅ Solution Créée - Aligned Meta-Labels

### Script: `src/create_meta_labels_aligned.py` (CRÉÉ)

**Commit**: `a74abec` - "feat: Create aligned meta-labels script matching real backtest strategy"

**Principe**: Créer des meta-labels qui correspondent **EXACTEMENT** au calcul PnL du backtest.

#### Pipeline Aligned

```python
# Au lieu de Triple Barrier:
direction = modèle_primaire[i]
entry_price = open[i+1]

# Trouver quand direction change
j = prochain_index_où_direction_change

exit_price = open[j+1]

# Calculer PnL exactement comme dans le backtest
if direction == UP:
    pnl = (exit_price - entry_price) / entry_price
else:  # SHORT
    pnl = (entry_price - exit_price) / entry_price

pnl_after_fees = pnl - (2 * fees)

# Label meta simple et aligné
label_meta = 1 if pnl_after_fees > 0 else 0
```

**Avantages**:
- Labels correspondent **EXACTEMENT** au calcul PnL du backtest
- Pas de barrières artificielles
- Pas de contraintes de durée arbitraires
- Le meta-modèle apprend à prédire la profitabilité **RÉELLE**

### Différences Clés vs Triple Barrier

| Aspect | Triple Barrier | Aligned |
|--------|----------------|---------|
| **Exit Logic** | 3 conditions (TP, SL, time) | 1 condition (signal flip) |
| **PnL Calc** | Avec barrières | Sans barrières |
| **Duration** | min_duration=5 imposé | Variable naturelle |
| **Alignment** | ❌ Différent du backtest | ✅ **IDENTIQUE au backtest** |

---

## 🎯 Prochaines Étapes EXACTES

### Étape 1: Générer Aligned Meta-Labels (À FAIRE)

```bash
# Train split
python src/create_meta_labels_aligned.py \
    --indicator macd \
    --filter kalman \
    --split train \
    --fees 0.001

# Validation split
python src/create_meta_labels_aligned.py \
    --indicator macd \
    --filter kalman \
    --split val \
    --fees 0.001

# Test split
python src/create_meta_labels_aligned.py \
    --indicator macd \
    --filter kalman \
    --split test \
    --fees 0.001
```

**Outputs attendus**:
```
data/prepared/meta_labels_macd_kalman_train_aligned.npz
data/prepared/meta_labels_macd_kalman_val_aligned.npz
data/prepared/meta_labels_macd_kalman_test_aligned.npz
```

### Étape 2: Modifier train_meta_model_phase217.py (À FAIRE)

**Ajout requis**:
```python
# Ligne ~30
parser.add_argument('--aligned', action='store_true',
                    help='Utiliser labels aligned au lieu de Triple Barrier')

# Ligne ~45
if args.aligned:
    # Charger datasets aligned
    train_data = np.load('data/prepared/meta_labels_macd_kalman_train_aligned.npz')
    # ...
else:
    # Charger datasets Triple Barrier (ancien)
    train_data = np.load('data/prepared/meta_labels_macd_kalman_train.npz')
    # ...
```

### Étape 3: Réentraîner Meta-Model avec Aligned Labels (À FAIRE)

```bash
python src/train_meta_model_phase217.py --filter kalman --aligned
```

**Modèle sauvegardé**:
```
models/meta_model/meta_model_baseline_kalman_aligned.pkl
models/meta_model/meta_model_results_kalman_aligned.json
```

### Étape 4: Modifier test_meta_model_backtest.py (À FAIRE)

**Ajout requis**:
```python
# Ligne ~30
parser.add_argument('--aligned', action='store_true',
                    help='Utiliser meta-model aligned')

# Ligne ~100
if args.aligned:
    model_path = 'models/meta_model/meta_model_baseline_kalman_aligned.pkl'
else:
    model_path = 'models/meta_model/meta_model_baseline_kalman.pkl'
```

### Étape 5: Re-Backtest avec Aligned Meta-Model (À FAIRE)

```bash
# Test avec aligned meta-model
python tests/test_meta_model_backtest.py \
    --indicator macd \
    --split test \
    --aligned \
    --compare-thresholds
```

**Résultats Attendus**:

| Stratégie | Trades | Win Rate | PnL Net | Verdict |
|-----------|--------|----------|---------|---------|
| Baseline | 108,702 | 22.49% | -21,382% | Référence |
| **Aligned (0.6)** | ~40,000 | **≥35%** ✅ | **Positif** ✅ | Win Rate **AUGMENTE** |

**Critères de Succès**:
- ✅ Win Rate **AUGMENTE** avec filtrage (pas de diminution)
- ✅ PnL Net devient **positif** ou nettement amélioré
- ✅ Trades réduits de ~60-70%

---

## 📊 Contexte Phase 2.15 (Rappel)

### Oracle Results - Nouvelle Formule (t vs t-1)

| Indicateur | PnL Net | Win Rate | Profit Factor | Sharpe |
|------------|---------|----------|---------------|--------|
| **RSI** 🥇 | **+23,039%** | 57.3% | 4.02 | 102.67 |
| **CCI** 🥈 | **+17,335%** | 56.4% | 3.16 | 87.55 |
| **MACD** 🥉 | **+14,359%** | 53.4% | 2.79 | 85.44 |

**Le signal EXISTE et fonctionne!** Oracle prouve +14k-23k% PnL Net.

### ML Baseline (Sans Meta-Model)

| Indicateur | Trades | Win Rate | PnL Net | Problème |
|------------|--------|----------|---------|----------|
| MACD | 108,702 | 22.49% | **-21,382%** | ❌ Trop de trades |
| RSI | 96,886 | - | - | ❌ (non testé mais similaire) |

**L'objectif du meta-model**: Filtrer pour passer de 22% Win Rate → 35-40%+ Win Rate.

---

## 🚫 Ce Qui a ÉCHOUÉ (Ne Pas Retester)

| Approche | Résultat | Raison |
|----------|----------|--------|
| **Triple Barrier Meta-Labels** | Win Rate ↓ | ❌ Mismatch avec backtest |
| Fusion multi-indicateurs | -15% à -43% | Corrélation 100% |
| Vote majoritaire | 0% gain | Mêmes erreurs |
| Force filter | -354% à -800% | Non prédictif |
| ATR filters | Neutre | Flickering bypass |
| Kalman/Octave sliding window | -19% à -116% | Lag détruit signal |

---

## 📁 Fichiers Clés du Projet

### Scripts Meta-Labeling

| Script | Status | Usage |
|--------|--------|-------|
| `src/create_meta_labels_phase215.py` | ✅ Existant | Triple Barrier (ANCIEN) |
| **`src/create_meta_labels_aligned.py`** | ✅ **CRÉÉ** | **Aligned labels (NOUVEAU)** |
| `src/train_meta_model_phase217.py` | ⏳ À modifier | Ajout --aligned flag |
| `tests/test_meta_model_backtest.py` | ⏳ À modifier | Ajout --aligned flag |

### Datasets Direction-Only

```
data/prepared/dataset_btc_eth_bnb_ada_ltc_macd_direction_only_kalman.npz
data/prepared/dataset_btc_eth_bnb_ada_ltc_rsi_direction_only_kalman.npz
data/prepared/dataset_btc_eth_bnb_ada_ltc_cci_direction_only_kalman.npz
```

### Modèles Primaires Entraînés

```
models/best_model_macd_kalman_dual_binary.pth  (92.4% Direction, 81.5% Force)
models/best_model_rsi_kalman_dual_binary.pth   (87.4% Direction, 74.0% Force)
models/best_model_cci_kalman_dual_binary.pth   (89.3% Direction, 77.4% Force)
```

---

## ⚠️ Règles Critiques

### 1. Ne JAMAIS exécuter de scripts
Claude n'a PAS les données. Fournir commandes, utilisateur exécute.

### 2. Réutiliser l'existant
- Logique backtest → `tests/test_meta_model_backtest.py`
- Calcul PnL → Copier exactement, ne pas réinventer

### 3. MACD = Indicateur Pivot
Focus sur MACD pour Phase 2.18 (meilleur pour trading réel).

### 4. Alignement = Clé du Succès
**Les labels de meta-labeling doivent correspondre EXACTEMENT à la stratégie de trading.**

---

## 💡 Ce Que Tu Dois Faire

### Tâche Immédiate

1. **Lire** `/home/user/trad/CLAUDE.md` section Phase 2.18 pour contexte complet
2. **Vérifier** que tu comprends le mismatch Triple Barrier vs Backtest
3. **Proposer** les modifications exactes pour étapes 2 et 4 ci-dessus
4. **Fournir** les commandes complètes pour tester

### Questions à Anticiper

- "Comment modifier train_meta_model_phase217.py pour support --aligned?"
- "Comment modifier test_meta_model_backtest.py pour charger aligned model?"
- "Que faire si aligned meta-model ne fonctionne pas mieux?"

### Approche Attendue

1. Lire le code des scripts à modifier
2. Proposer les modifications précises (diff-style)
3. Expliquer pourquoi c'est aligné maintenant
4. Donner commandes de test et critères de validation

---

## 📌 Résumé Exécutif

| Aspect | État |
|--------|------|
| **Phase** | 2.18 Meta-Model Backtest |
| **Problème identifié** | ✅ Triple Barrier ≠ Backtest (mismatch) |
| **Solution créée** | ✅ Script aligned meta-labels |
| **Next step** | ⏳ Générer labels + réentraîner + re-backtest |
| **Critère succès** | Win Rate ↑ avec filtrage (pas ↓) |
| **Commit actuel** | `a74abec` |

---

## 🔗 Références Critiques

**Expert Diagnosis** (CLAUDE.md Phase 2.18):
> "Le problème NE vient pas du méta-modèle. Il vient AVANT."
>
> "Triple Barrier labels ≠ Backtest PnL calculation"

**López de Prado (Advances in Financial ML)**:
> "Meta-labeling improves profitable primary models. It cannot invert the sign of a losing model."

**Leçon Critique**:
> Les labels de meta-labeling doivent correspondre EXACTEMENT à la stratégie de trading utilisée en backtest. Toute différence créera un mismatch qui rendra le filtrage inefficace ou inverse.

---

**Dis-moi que tu as bien compris le contexte et je te donne la première tâche!**
