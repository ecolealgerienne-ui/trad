# 🚀 Prompt de Démarrage Nouvelle Session

**Date de dernière session**: 2026-01-09
**État du projet**: Overfitting détecté sur modèle large - Diagnostic en cours
**Branch Git**: `claude/review-project-context-oZWBw`

---

## 📋 Contexte à Charger

Bonjour Claude,

Je continue le développement du projet **CNN-LSTM Direction-Only** pour prédiction de tendance crypto (MACD). Voici le contexte essentiel de la dernière session:

## 🎯 Situation Actuelle

### Modèle Baseline (SUCCÈS) ✅

- **Architecture**: 64 CNN filters / 64 LSTM hidden / 2 LSTM layers
- **Test Accuracy**: **90.3%** (excellent)
- **F1 Score**: 0.903
- **Gap train/val**: ~4% (acceptable)
- **Dataset**: `dataset_btc_eth_bnb_ada_ltc_macd_direction_only_kalman.npz`
- **Format**: Direction-Only (1 output, 1 feature c_ret uniquement)

### Modèle Large (ÉCHEC) ❌

- **Architecture testée**: 128 CNN filters / 128 LSTM hidden / 3 LSTM layers
- **Résultats**:
  - Train Acc: 89.9% ✅
  - **Val Acc: 69.9%** ❌ (gap -20% = overfitting sévère)
  - **Test Acc: 88.3%** 📉 (perte de -2% vs baseline)
- **Diagnostic**: Modèle trop grand pour la quantité de données → overfitting massif

### Anomalie Détectée ⚠️

**Val Acc (69.9%) << Test Acc (88.3%)**
→ Écart de +18.4% entre val et test (très inhabituel!)

**Hypothèses**:
1. Val set d'une période exceptionnellement difficile
2. Weighted transitions (_wt) cause l'overfitting
3. Bug dans le calcul de val accuracy pendant training

## 🔧 Script de Diagnostic Créé

**Fichier**: `tests/diagnose_overfitting.py`
**Objectif**: Comprendre l'anomalie val/test et identifier la cause de l'overfitting

**Ce qu'il analyse**:
- Distribution labels train/val/test
- Périodes temporelles de chaque split
- Volatilité (difficulté) de chaque période
- Recalcul accuracy pour vérifier les métriques
- Transitions (si weighted loss utilisé)

**Commande**:
```bash
python tests/diagnose_overfitting.py --data data/prepared/dataset_btc_eth_bnb_ada_ltc_macd_direction_only_kalman_wt.npz
```

## 📁 Format Direction-Only (IMPORTANT)

**Structure du dataset**:
- **X**: (n, 25, 3) → colonnes [timestamp, asset_id, c_ret]
- **Y**: (n, 3) → colonnes [timestamp, asset_id, label]
- **T**: (n, 3) → colonnes [timestamp, asset_id, is_transition]

**Extraction dans `load_prepared_data()`**:
```python
X_train = data['X_train'][:, :, 2:3]  # (n, 25, 3) → (n, 25, 1) = c_ret uniquement
Y_train = data['Y_train'][:, 2:3]     # (n, 3) → (n, 1) = label uniquement
T_train = data['T_train'][:, 2:3]     # (n, 3) → (n, 1) = is_transition uniquement
```

**Asset ID Mapping** (0-indexed):
- BTC=0, ETH=1, BNB=2, ADA=3, LTC=4

**Filtrage par asset**: Utilise `OHLCV[:, 1]` (pas X car X n'a qu'1 colonne après extraction)

## 🐛 Bugs Critiques Déjà Fixés (Sessions Précédentes)

| Bug | Impact | Fix | Commit |
|-----|--------|-----|--------|
| Asset ID 1-indexed | 20.6% perte données | `enumerate(start=0)` | a5faaff |
| X contient timestamp/asset_id | Model apprend du bruit (50% acc) | Extract col 2 uniquement | ffdb61c |
| Filtering après extraction | IndexError | Use OHLCV[:, 1] | 990ba36 |

**Résultat**: 4-pass verification ✅ COMPLÈTE (commit 687ca96)

## 🎯 Prochaines Étapes Recommandées

### Option 1: Diagnostic (PRIORITÉ) 🔍

```bash
python tests/diagnose_overfitting.py --data data/prepared/dataset_btc_eth_bnb_ada_ltc_macd_direction_only_kalman_wt.npz
```

**Objectif**: Comprendre pourquoi val=69.9% mais test=88.3%

### Option 2: Revenir au Baseline (RECOMMANDÉ) ✅

```bash
python src/train.py \
    --data data/prepared/dataset_btc_eth_bnb_ada_ltc_macd_direction_only_kalman.npz \
    --epochs 50 \
    --patience 15
```

**Attendu**: Retrouver 90.3% test accuracy

### Option 3: Taille Intermédiaire (ALTERNATIF) ⚖️

```bash
python src/train.py \
    --data data/prepared/dataset_btc_eth_bnb_ada_ltc_macd_direction_only_kalman.npz \
    --cnn-filters 96 \
    --lstm-hidden 96 \
    --lstm-layers 2 \
    --lstm-dropout 0.35 \
    --dense-dropout 0.4 \
    --epochs 50
```

**Attendu**: 89-90% test accuracy, gap train/val < 10%

## 📊 Données du Projet

**Assets**: BTC, ETH, BNB, ADA, LTC
**Timeframe**: 5 minutes
**Total sequences**: ~4.3M (2.4M train après filtrage)
**Période**: 2017-08 → 2026-01 (~8.5 ans)

**Splits**:
- Train: 70% (~13 mois)
- Val: 15% (~2.8 mois, échantillonné)
- Test: 15% (~2.8 mois, toujours à la FIN)

## 🔑 Informations Clés

1. **Modèle baseline (64/64/2) fonctionne parfaitement**: 90.3% test accuracy
2. **Ne PAS augmenter la taille**: Overfitting confirmé avec 128/128/3
3. **Direction-Only format validé**: 4-pass verification complète
4. **Asset filtering fonctionne**: 0-indexed mapping corrigé
5. **Weighted transitions (_wt)**: Potentiellement cause de l'overfitting (à investiguer)

## ❓ Questions à Résoudre

1. **Pourquoi val=69.9% mais test=88.3%?** (anomalie majeure)
2. **Weighted transitions cause-t-il l'overfitting?** (fichier _wt.npz)
3. **Le val set vient-il d'une période exceptionnellement difficile?** (volatilité?)
4. **Faut-il désactiver weighted transitions?** (loss standard vs weighted)

## 🛠️ Fichiers Importants

**Scripts de diagnostic**:
- `tests/diagnose_overfitting.py` (créé session actuelle - commit baa393d)
- `tests/verify_pipeline.py` (4-pass verification)
- `tests/diagnose_dataset.py` (analyse raw data)

**Scripts de training**:
- `src/train.py` (avec --assets, --cnn-filters, --lstm-hidden, etc.)
- `src/evaluate.py` (avec --assets)
- `src/prepare_data.py` (avec Direction-Only extraction)

**Documentation**:
- `CLAUDE.md` (règles critiques et historique complet)
- `docs/ADAPTATION_DIRECTION_ONLY.md` (format Direction-Only)

## 🚀 Comment Démarrer

**Si tu veux continuer immédiatement**, exécute:

```bash
python tests/diagnose_overfitting.py --data data/prepared/dataset_btc_eth_bnb_ada_ltc_macd_direction_only_kalman_wt.npz
```

**Sinon, demande-moi**:
- "Lance le diagnostic" → je lance `diagnose_overfitting.py`
- "Retour au baseline" → je réentraîne avec 64/64/2
- "Teste taille intermédiaire" → je teste 96/96/2 avec plus de dropout
- "Explique les résultats" → j'analyse les logs fournis

## 📌 Git Status

**Branch actuelle**: `claude/review-project-context-oZWBw`
**Dernier commit**: `baa393d` - feat: Add overfitting diagnostic script
**Status**: Clean (tous changements committés et pushés)

---

## 📚 Historique Session Précédente (Context)

### Session 1: Adaptation Direction-Only (Bugs Fixés)

**3 bugs critiques corrigés**:
1. **Asset ID mapping**: 1-indexed → 0-indexed (20.6% data loss)
2. **Feature extraction**: X contenait timestamp/asset_id → extraire c_ret uniquement
3. **Filtering mechanism**: IndexError après extraction → utiliser OHLCV

**Résultat**: Modèle passe de 50% (bruit) à 90.3% accuracy (signal)

### Session 2 (actuelle): Test Modèle Large

**Tentative**: Augmenter capacité modèle (64/64/2 → 128/128/3)
**Résultat**: Overfitting sévère (gap train/val 20%)
**Action**: Diagnostic créé, retour au baseline recommandé

---

## 💡 Ce Que Tu Dois Savoir

### ✅ Ce Qui Fonctionne

- **Pipeline de données**: Extraction Direction-Only validée (4-pass ✅)
- **Asset filtering**: 0-indexed mapping correct
- **Modèle baseline**: 90.3% test accuracy excellent
- **Format Direction-Only**: Plus simple et performant

### ❌ Ce Qui Ne Fonctionne PAS

- **Modèle large (128/128/3)**: Overfitting massif, -2% performance
- **Weighted transitions (_wt)**: Potentiellement cause de l'overfitting

### 🤔 Ce Qu'on Doit Investiguer

- **Anomalie val/test**: Pourquoi val=69.9% mais test=88.3%?
- **Périodes temporelles**: Val set exceptionnellement difficile?
- **Weighted transitions**: Impact sur overfitting?

---

**Commence par me dire ce que tu veux faire** et je t'aiderai à continuer exactement où on en était! 🎯

**Suggestions**:
1. 🔍 Lance le diagnostic pour comprendre l'anomalie
2. ✅ Retour au baseline (safe)
3. ⚖️ Test taille intermédiaire (compromis)
