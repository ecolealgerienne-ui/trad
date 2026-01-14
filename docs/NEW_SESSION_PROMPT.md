# 🎯 Prompt Nouvelle Session - Classificateur de Régimes avec Labels Futurs

**Date**: 2026-01-14
**Version**: 11.0 - Régimes avec prédiction FUTURE
**Branch Git**: `claude/review-and-sync-main-g5Oqq`
**Dernier Commit**: `755d870` - "feat: Implement future regime labels with Any TREND logic"

---

## ⚠️ RÈGLES CRITIQUES À RESPECTER (IMPÉRATIF)

### 1. 🔁 UTILISER L'EXISTANT

**Principe Fondamental**: **"Je regarde l'existant et je reparte de l'existant"**

Avant d'écrire du nouveau code, TOUJOURS:
1. Chercher un script similaire existant
2. Le **COPIER** comme base
3. Modifier **UNIQUEMENT** ce qui doit changer

**Exemples validés**:
- ✅ `create_meta_labels_aligned.py`: Copié de `create_meta_labels_phase215.py`, modifié SEULEMENT la fonction de labeling → Succès
- ❌ Réécrire from scratch → ImportError, bugs, perte de temps

**Coût d'une violation**: Bug critique, incompatibilités, perte de temps (2h vs 5min)

### 2. ❌ PAS D'INITIATIVES SANS RAISON

- **NE PAS** prendre d'initiatives sans demander
- **NE PAS** ajouter de features ou optimisations non demandées
- **DEMANDER** en cas de doute

### 3. 📚 SCRIPTS D'INSPIRATION

| Script | Usage |
|--------|-------|
| `src/train.py` | Pattern d'entraînement |
| `src/evaluate.py` | Pattern d'évaluation |
| `src/prepare_data_regime.py` | Préparation données régime |
| `src/train_regime_classifier.py` | CNN-LSTM régime |

### 4. 📝 MISE À JOUR DOCUMENTATION

Après chaque changement majeur:
- **METTRE À JOUR** `CLAUDE.md` (section Classificateur de Régimes)
- **CRÉER/METTRE À JOUR** `docs/status.md` pour le suivi

### 5. 🚫 NE JAMAIS LANCER DE SCRIPTS

Claude Code ne possède PAS les datasets locaux.

**INTERDIT**: `python src/train.py`, `python tests/test_*.py`
**AUTORISÉ**: Lire/modifier code .py, fournir commandes à exécuter

---

## 📋 CONTEXTE ACTUEL DU PROJET

### Problème Initial : Data Leakage (RÉSOLU)

**XGBoost 98.95% = INVALIDE** car:
- Les 20 features (adx, atr, bb_width, etc.) sont les MÊMES que celles utilisées pour calculer les labels
- Le modèle reconstruit la formule au lieu de prédire

**Solution validée**:
- **CNN-LSTM avec raw returns uniquement** (h_ret, l_ret, c_ret) → 86.33% accuracy
- Ces features ne sont PAS dans la formule de labeling → PAS de leakage
- **C'est le SEUL modèle valide**

### Évolution Majeure : Labels FUTURS (2026-01-14)

**Insight critique de l'utilisateur**:
> "Les labels des régimes doivent être calculés en fonction du FUTUR pour que le modèle apprenne à prédire quelque chose d'inconnu, sinon on peut utiliser directement les features pour estimer le régime actuel."

**Paramètres décidés**:
- **N = 6** (lookahead de 30 minutes sur données 5min)
- **Logique "Any TREND"**: Si TREND apparaît dans [t+1, t+6], label = TREND

---

## 🔧 IMPLÉMENTATION RÉALISÉE

### 1. `src/constants.py` - Nouvelle constante

```python
SEQUENCE_LENGTH = 25
REGIME_LOOKAHEAD = 6  # Horizon prédiction: 6 × 5min = 30 min
```

### 2. `src/prepare_data_regime.py` - Fonction helper

```python
def compute_future_regime_label(regime_labels: np.ndarray,
                                 lookahead: int = REGIME_LOOKAHEAD) -> np.ndarray:
    """
    Calcule le label de régime FUTUR avec logique "Any TREND".

    Pour chaque position t, regarde les régimes dans [t+1, t+N]:
    - Si ANY régime est TREND (2): label = TREND
    - Sinon: vote majoritaire entre RANGE_LOW_VOL (0) et RANGE_HIGH_VOL (1)
    """
    N = len(regime_labels)
    n_valid = N - lookahead

    future_labels = np.zeros(n_valid, dtype=np.float32)

    for i in range(n_valid):
        future_window = regime_labels[i+1:i+1+lookahead]

        if 2 in future_window:  # Any TREND
            future_labels[i] = 2.0
        else:
            counts = np.bincount(future_window.astype(int), minlength=3)
            if counts[1] > counts[0]:
                future_labels[i] = 1.0  # RANGE_HIGH_VOL
            else:
                future_labels[i] = 0.0  # RANGE_LOW_VOL

    return future_labels
```

### 3. `create_sequences_for_regime()` - Mise à jour

- Ajout paramètre `lookahead` (défaut: 6)
- Le label régime est sur fenêtre FUTURE `[t+1, t+N]`
- Les labels direction (macd, rsi, cci) restent à temps `t`
- On perd `lookahead` séquences à la fin (pas de données futures)

---

## 📊 ALIGNEMENT DES DONNÉES

| Élément | Position | Description |
|---------|----------|-------------|
| **Features X** | `[t-24, t]` | 25 timesteps de features |
| **Label direction** | `t` | macd_dir, rsi_dir, cci_dir à la fin de séquence |
| **Label régime** | `[t+1, t+6]` | Fenêtre FUTURE avec logique "Any TREND" |

**Perte de séquences**: On perd 6 séquences à la fin de chaque split (pas de données futures disponibles)

---

## 🏷️ Labels de Régime (3 classes)

| Classe | ID | Description | Seuils |
|--------|-----|-------------|--------|
| **RANGE_LOW_VOL** | 0 | Range, faible volatilité | TS < 0.45 ET Vol < P50 |
| **RANGE_HIGH_VOL** | 1 | Range, haute volatilité | TS < 0.45 ET Vol >= P50 |
| **TREND** | 2 | Tendance claire | TS >= 0.45 |

**Distribution observée**: TREND très rare (3-6%), RANGE domine (94-97%)

---

## 📁 FICHIERS CLÉS

| Fichier | Rôle |
|---------|------|
| `src/constants.py` | Constantes (REGIME_LOOKAHEAD=6) |
| `src/prepare_data_regime.py` | Préparation dataset avec labels FUTURS |
| `src/train_regime_classifier.py` | Entraînement CNN-LSTM |
| `src/regime_model.py` | Architecture CNN-LSTM (3 classes) |
| `src/regime_labeler.py` | Calcul labels régime (TS, VC scores) |
| `data/prepared/regime_*.npz` | Datasets préparés |
| `models/regime_cnn_lstm/` | Modèles sauvegardés |

---

## ✅ COMMITS RÉCENTS

```
755d870 feat: Implement future regime labels with "Any TREND" logic
0873a70 feat: Add REGIME_LOOKAHEAD constant for future regime prediction
f8e3ec9 docs: Document data leakage in XGBoost regime classifier
eeb998f docs: Document regime label distribution and class imbalance
```

---

## 🎯 TÂCHES À FAIRE

### Priorité 1 : Régénérer le Dataset

```bash
python src/prepare_data_regime.py --assets BTC ETH BNB ADA LTC
```

Le nouveau dataset aura:
- X: Features aux temps `[t-24, t]`
- Y: Labels avec régime FUTUR (calculé sur `[t+1, t+6]`)
- ~6 séquences de moins par asset (perdues à cause du lookahead)

### Priorité 2 : Réentraîner le CNN-LSTM

```bash
python src/train_regime_classifier.py \
    --data data/prepared/regime_train.npz \
    --val-data data/prepared/regime_val.npz \
    --epochs 50 \
    --batch-size 512
```

### Priorité 3 : Évaluer les Résultats

- Comparer accuracy avant/après labels futurs
- L'accuracy sera plus basse (prédiction réelle, pas reconstruction)
- Vérifier F1 par classe, surtout TREND (classe rare)

### Priorité 4 : Mettre à Jour Documentation

- Section "Classificateur de Régimes" dans `CLAUDE.md`
- Créer/màj `docs/status.md`

---

## ❓ FAQ RAPIDE

**Q: Pourquoi XGBoost est invalide?**
R: Les features sont les MÊMES que celles du labeling → data leakage

**Q: Pourquoi CNN-LSTM est valide?**
R: Utilise UNIQUEMENT raw returns (pas dans la formule de labeling)

**Q: Pourquoi prédire le régime FUTUR?**
R: Prédire le présent avec features du présent = pas de prédiction utile

**Q: C'est quoi "Any TREND"?**
R: Si TREND apparaît au moins 1 fois dans les 6 prochaines périodes → label = TREND

**Q: Pourquoi N=6?**
R: 6 × 5min = 30 min de lookahead (horizon raisonnable pour trading)

---

## 🚀 COMMANDES POUR DÉMARRER

```bash
# 1. Vérifier la branche
git checkout claude/review-and-sync-main-g5Oqq
git pull

# 2. Régénérer le dataset avec labels futurs
python src/prepare_data_regime.py --assets BTC ETH BNB ADA LTC

# 3. Entraîner le CNN-LSTM
python src/train_regime_classifier.py \
    --data data/prepared/regime_train.npz \
    --val-data data/prepared/regime_val.npz \
    --epochs 50

# 4. Évaluer sur test set
python src/train_regime_classifier.py \
    --eval-only \
    --data data/prepared/regime_test.npz
```

---

## 📌 RÉSUMÉ EXÉCUTIF

| Aspect | État |
|--------|------|
| **Data Leakage** | ✅ Résolu (CNN-LSTM valide, XGBoost invalide) |
| **Labels Futurs** | ✅ Implémenté (lookahead N=6, Any TREND) |
| **Code** | ✅ Commité et poussé |
| **Prochaine étape** | ⏳ Régénérer dataset + Réentraîner |
| **Accuracy attendue** | Plus basse (prédiction réelle vs reconstruction) |

---

## 💬 INSTRUCTION POUR CLAUDE

**Dis-moi que tu as compris le contexte et attends mes instructions.**

**Rappel des règles**:
1. Utiliser l'existant
2. Pas d'initiatives sans raison
3. S'inspirer des scripts existants
4. Mettre à jour CLAUDE.md et status.md
