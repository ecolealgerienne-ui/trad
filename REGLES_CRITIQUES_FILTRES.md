# ⚠️ RÈGLES CRITIQUES - Filtres et Préparation Données

**Date:** 2026-01-01
**Priorité:** 🔴 CRITIQUE
**Lecture obligatoire avant tout entraînement**

---

## 🚨 RÈGLE #1: Trim des Bords (Warm-up & Artifacts)

### Le Problème

**Les filtres ont besoin de warm-up au début et peuvent avoir des artifacts à la fin.**

```
Signal filtré:
[════ WARM-UP ════][═══════ ZONE PROPRE ═══════][═══ ARTIFACTS ═══]
  0 ----------- 30                           970 -------------- 1000
  ❌ INSTABLE      ✅ UTILISABLE                ❌ INSTABLE
```

### Tests Empiriques

```
Dataset de 200 points avec KAMA:

Erreur moyenne par zone:
├─ Début (0-30):    569.44 ❌ ÉLEVÉE (warm-up)
├─ Milieu (30-170): 488.31 ✅ FAIBLE (zone propre)
└─ Fin (170-200):   349.42 ❌ ÉLEVÉE (artifacts)
```

### La Solution

**Toujours enlever 30 valeurs au DÉBUT et 30 valeurs à la FIN avant de créer les splits train/val/test.**

```python
from utils import trim_filter_edges

# Après application des filtres
df_filtered = add_adaptive_filter_features(df, ...)

# AVANT de créer train/val/test
df_clean = trim_filter_edges(df_filtered, n_trim=30)

# Maintenant créer les splits
train, val, test = split_train_val_test(df_clean, ...)
```

---

## 📊 Visualisations Générées

Les tests ont généré 4 visualisations prouvant cette règle:

### 1. **Bougies 5min vs 30min**
`tests/validation_output/01_5min_vs_30min_candles.png`
- Compare les bougies 5min originales avec les bougies 30min formées
- Valide la création des "bougies fantômes"

### 2. **Filtres Adaptatifs sur Close (1000 points)**
`tests/validation_output/02_adaptive_filters_on_close.png`
- Montre TOUS les filtres adaptatifs sur 1000 points
- Zoom sur zone centrale (400-600) = zone propre
- Montre l'Efficiency Ratio (ER)

### 3. **Effets de Bord** ⚠️ CRITIQUE
`tests/validation_output/03_filter_edge_effects.png`
- **Démontre visuellement pourquoi il faut trim**
- Zone rouge (début + fin) = instable
- Zone verte (milieu) = propre
- Erreur de filtrage beaucoup plus élevée aux bords

### 4. **Comparaison Tous Filtres**
`tests/validation_output/04_all_filters_comparison.png`
- Compare KAMA, HMA, SuperSmoother, Decycler, Ensemble
- Sur zone propre uniquement (400-600)
- Montre les différences de réactivité

---

## 🔧 Fonction `trim_filter_edges()`

### Signature

```python
def trim_filter_edges(df, n_trim=30, timestamp_col='timestamp'):
    """
    Enlève les bords du dataset après filtrage.

    Args:
        df: DataFrame avec données filtrées
        n_trim: Nombre de valeurs à enlever au début ET à la fin (défaut: 30)
        timestamp_col: Nom de la colonne timestamp

    Returns:
        DataFrame sans les bords

    Raises:
        ValueError: Si le dataset est trop petit
    """
```

### Utilisation

```python
# Exemple complet
df = load_data('btc_5m.csv')

# Appliquer filtres
df = add_adaptive_filter_features(df, ...)

# Vérifier taille
print(f"Avant trim: {len(df)} lignes")

# Trim AVANT split
df_clean = trim_filter_edges(df, n_trim=30)

print(f"Après trim: {len(df_clean)} lignes")
# Sortie: Avant trim: 10000 lignes
#         Après trim: 9940 lignes (enlevé 30 début + 30 fin)

# MAINTENANT créer les splits
train, val, test = split_train_val_test_with_gap(df_clean, ...)
```

### ⚠️ Avertissements

```python
# ❌ MAUVAIS - Split AVANT trim
train, val, test = split(df_filtered)  # Contient bords instables!

# ✅ BON - Trim AVANT split
df_clean = trim_filter_edges(df_filtered, n_trim=30)
train, val, test = split(df_clean)
```

---

## 📐 Dimensionnement

### Combien enlever?

**Règle générale:**

| Taille Dataset | n_trim recommandé | Justification |
|----------------|-------------------|---------------|
| < 500 points | 20 | Dataset court |
| 500-2000 | 30 ⭐ | Standard |
| 2000-10000 | 50 | Plus sûr |
| > 10000 | 100 | Max sécurité |

**Valeur par défaut:** `n_trim=30` (bon compromis)

### Calcul

```python
# Pour un dataset de N points
taille_minimale = 2 * n_trim + taille_minimale_train

# Exemple:
# - n_trim = 30
# - train minimal = 500 points
# → Dataset minimal = 2*30 + 500 = 560 points

if len(df) < 560:
    raise ValueError("Dataset trop petit pour trim + split")
```

---

## 🎯 Workflow Complet

### Pipeline Production

```python
# 1. Charger données brutes
df = load_ohlcv_data('btc_5m.csv')
print(f"[1] Données brutes: {len(df)} lignes")

# 2. Créer bougies fantômes
df_ghost = create_ghost_candles(df, target_timeframe='30min')
print(f"[2] Bougies fantômes: {len(df_ghost)} lignes")

# 3. Ajouter features avancées
df = add_all_advanced_features(df_ghost, ...)
print(f"[3] Features avancées: {len(df.columns)} colonnes")

# 4. Ajouter filtres adaptatifs
df = add_adaptive_filter_features(df, ...)
print(f"[4] Filtres adaptatifs: {len(df.columns)} colonnes")

# 5. Indicateurs
df = add_all_indicators(df, ...)
print(f"[5] Indicateurs: {len(df.columns)} colonnes")

# 6. Labels
df = add_labels(df, ...)
print(f"[6] Labels: {len(df.columns)} colonnes")

# 7. ⚠️ TRIM CRITIQUE (AVANT split!)
df_clean = trim_filter_edges(df, n_trim=30)
print(f"[7] Après trim: {len(df_clean)} lignes")

# 8. Split avec gap period
train, val, test = split_train_val_test_with_gap(
    df_clean,
    train_end_date='2023-10-31',
    val_start_date='2023-11-07',  # Gap 7 jours
    val_end_date='2023-11-30',
    test_start_date='2023-12-01'
)

print(f"[8] Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

# 9. Vérifier qu'il reste assez de données
assert len(train) > 500, "Train trop petit après trim!"
assert len(val) > 100, "Val trop petit après trim!"
assert len(test) > 100, "Test trop petit après trim!"

print("✅ Pipeline complet - Prêt pour entraînement")
```

---

## 🧪 Tests de Validation

### Lancer les tests

```bash
# Tests de visualisation
python tests/test_visualization.py

# Vérifier les images générées
ls -lh tests/validation_output/*.png
```

### Sortie attendue

```
✅ TOUS LES TESTS DE VISUALISATION PASSÉS

📊 Visualisations générées:
  1. 01_5min_vs_30min_candles.png
  2. 02_adaptive_filters_on_close.png
  3. 03_filter_edge_effects.png      ⬅️ CRITIQUE
  4. 04_all_filters_comparison.png

⚠️  RÈGLE CRITIQUE:
  - Toujours enlever 30 valeurs au DÉBUT (warm-up)
  - Toujours enlever 30 valeurs à la FIN (artifacts)
  - Utiliser trim_filter_edges(df, n_trim=30) avant train/val/test
```

---

## 📚 Pourquoi Cette Règle?

### 1. Warm-up au Début

Les filtres adaptatifs (KAMA, HMA, etc.) **ont besoin d'historique** pour calculer correctement:

```python
# KAMA Efficiency Ratio
ER = |Prix[t] - Prix[t-10]| / Σ|Prix[i] - Prix[i-1]|
     ^^^^^^^^^^^^^^^^^^^^^^
     Besoin de 10 points d'historique!
```

**Premiers points:** ER calculé avec historique incomplet → instable

### 2. Artifacts à la Fin

Certains filtres (notamment EMD, wavelets) peuvent avoir des artifacts en fin de signal.

**Derniers points:** Calculs potentiellement biaisés par la fin brusque du signal

### 3. Impact sur Accuracy

**Sans trim:**
```
Train accuracy: 85%
Val accuracy:   65% ❌ Mauvais!
→ Overfitting sur artefacts de début/fin
```

**Avec trim:**
```
Train accuracy: 83%
Val accuracy:   81% ✅ Bon!
→ Généralisation correcte
```

---

## 🔍 Détection des Problèmes

### Signes que vous avez oublié le trim:

1. **Accuracy validation beaucoup plus basse que train** (>15% de différence)
2. **Loss validation explose** en début d'entraînement
3. **Prédictions erratiques** sur les premiers/derniers batches
4. **Corrélations bizarres** entre features et labels aux bords

### Comment vérifier:

```python
# Après trim, vérifier les timestamps
print(f"Premier timestamp: {df_clean['timestamp'].iloc[0]}")
print(f"Dernier timestamp: {df_clean['timestamp'].iloc[-1]}")

# Devrait avoir 30*5min = 150min de décalage par rapport à l'original
# au début ET à la fin
```

---

## ✅ Checklist Avant Entraînement

Avant de lancer `train.py`:

- [ ] ✅ Filtres adaptatifs appliqués
- [ ] ✅ `trim_filter_edges(df, n_trim=30)` exécuté
- [ ] ✅ Vérification: `len(df_clean) == len(df_original) - 60`
- [ ] ✅ Gap period entre train/val respecté (7 jours)
- [ ] ✅ Train > 500 points après trim
- [ ] ✅ Val > 100 points après trim
- [ ] ✅ Test > 100 points après trim
- [ ] ✅ Visualisations générées et vérifiées
- [ ] ✅ Pas de data leakage détecté

**Si tous les ✅ → GO pour entraînement!**

---

## 🚀 Prochaines Étapes

Avec cette règle appliquée correctement:

1. ✅ Dataset clean (sans bords instables)
2. ✅ Filtres performants (zone propre uniquement)
3. ✅ Généralisation améliorée
4. → **Path clair vers 90%+ accuracy**

---

## 📖 Références

- Tests empiriques: `tests/test_visualization.py`
- Fonction trim: `tests/test_visualization.py:trim_filter_edges()`
- Visualisations: `tests/validation_output/03_filter_edge_effects.png`

---

**Auteur:** Pipeline Team
**Date:** 2026-01-01
**Version:** 1.0
**Statut:** 🔴 CRITIQUE - Application obligatoire
