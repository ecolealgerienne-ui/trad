# Tests et Validation du Pipeline

Ce dossier contient les tests rigoureux pour valider chaque étape du pipeline de données.

**RÈGLE D'OR:** Toujours valider les données entre chaque étape avant de continuer. Un seul bug dans les données peut ruiner des semaines d'entraînement GPU!

---

## 📋 Scripts de Test

### 1. `quick_validation.py` - Validation Rapide ⚡

**Usage:**
```bash
python tests/quick_validation.py
```

**Ce qu'il teste:**
- ✅ Création des bougies fantômes (6 steps par bougie 30min)
- ✅ Intégrité OHLC (H≥L, H≥O, H≥C, etc.)
- ✅ Présence de toutes les features avancées
- ✅ Ranges corrects (step_index_norm [0.0-1.0], amplitude>0, etc.)
- ✅ Pas de NaN inattendus

**Durée:** ~5 secondes

**Quand l'utiliser:**
- Après chaque modification du code
- Avant de commit
- Pour vérifier rapidement que le pipeline fonctionne

---

### 2. `test_pipeline_validation.py` - Validation Complète 📊

**Usage:**
```bash
python tests/test_pipeline_validation.py
```

**Ce qu'il teste:**
- ✅ Bougies fantômes avec visualisations
- ✅ Features avancées (velocity, log returns, Z-Score) + distributions
- ✅ Data leakage (corrélations futures)
- ✅ Génération de graphiques de validation

**Sortie:**
- `tests/validation_output/01_ghost_candle_evolution.png`
- `tests/validation_output/02_advanced_features_distributions.png`
- `tests/validation_output/03_feature_evolution_per_step.png`
- `tests/validation_output/04_data_leakage_check.png`
- `tests/validation_output/validation_report.txt`

**Durée:** ~30 secondes

**Quand l'utiliser:**
- Avant un entraînement GPU
- Après intégration de nouvelles features
- Pour débugger visuellement les données

---

## 🎯 Checklist de Validation

Avant de lancer un entraînement, vérifier:

### Données Brutes (5min)
- [ ] Pas de NaN dans OHLC
- [ ] Timestamps consécutifs (pas de trous)
- [ ] OHLC integrity: H≥max(O,C), L≤min(O,C)
- [ ] Volume > 0

### Bougies Fantômes (30min)
- [ ] Exactement 6 steps par bougie complète
- [ ] Open constant dans une bougie
- [ ] High monotone croissant (ou constant)
- [ ] Low monotone décroissant (ou constant)
- [ ] Close = dernier close 5min

### Features Avancées
- [ ] `velocity`: range raisonnable (pas >1.0)
- [ ] `amplitude`: toujours positive
- [ ] `acceleration`: mean proche de 0
- [ ] `ghost_high/low/close_log`: mean proche de 0
- [ ] `ghost_open_zscore`: mean~0, std~1
- [ ] `step_index_norm`: exactement [0.0, 1.0]

### Indicateurs Techniques
- [ ] RSI: range [0, 100]
- [ ] CCI: range raisonnable [-300, +300]
- [ ] MACD: pas de valeurs aberrantes
- [ ] Bollinger: upper > middle > lower

### Labels
- [ ] Distribution équilibrée (40-60% de chaque classe)
- [ ] Pas de NaN (sauf warm-up du filtre)
- [ ] Pas de data leakage (|corr| < 0.7 avec future)
- [ ] Corrélation idéale: 0.1-0.3

### Multi-Actifs (si applicable)
- [ ] Colonne `asset` présente
- [ ] Normalisation séparée par actif
- [ ] Pas de fuite inter-actifs
- [ ] Distribution équilibrée entre actifs

### Split Train/Val/Test
- [ ] Gap period respecté (7 jours)
- [ ] Pas de chevauchement temporel
- [ ] Distribution labels similaire entre splits
- [ ] Taille suffisante (train>60%, val~20%, test~20%)

---

## 🚀 Exemples d'Usage

### Test Rapide après Modification
```bash
# Modifier le code
vim src/advanced_features.py

# Valider rapidement
python tests/quick_validation.py

# Si OK, commit
git add src/advanced_features.py
git commit -m "Add new feature"
```

### Validation Complète avant GPU
```bash
# Valider avec visualisations
python tests/test_pipeline_validation.py

# Vérifier les graphiques
ls -lh tests/validation_output/*.png

# Lire le rapport
cat tests/validation_output/validation_report.txt

# Si tout est OK, lancer l'entraînement
python train.py --config config/model_v1.yaml
```

### Test sur Vraies Données
```bash
# Créer dataset BTC
python src/data_pipeline.py \
  --input ../data_trad/BTCUSD_all_5m.csv \
  --output data/processed/btc_test.csv

# Valider le dataset
python tests/validate_dataset.py data/processed/btc_test.csv

# Si OK, créer multi-asset
python example_multiasset_run.py
```

---

## 📊 Interprétation des Résultats

### ✅ Succès
```
============================================================
✅ VALIDATION RÉUSSIE - Pipeline fonctionnel!
============================================================
```
**Action:** Continuer au prochain test ou lancer l'entraînement.

### ❌ Erreurs Critiques
```
❌ VALIDATION ÉCHOUÉE - 3 erreurs:
  - Certaines bougies n'ont pas 6 steps!
  - Amplitude négative détectée
  - Data leakage détecté: rsi_14 (corr=0.89)
```
**Action:**
1. Corriger les erreurs une par une
2. Relancer le test après chaque correction
3. NE PAS continuer tant qu'il reste des erreurs

### ⚠️  Warnings
```
⚠️  2 WARNINGS:
  - Open Z-Score mean=0.35 (devrait être ~0)
  - 10% de NaN dans RSI (acceptable pour warm-up)
```
**Action:** Vérifier manuellement, acceptable si expliqué.

---

## 🔧 Dépannage

### Erreur: "Bougies sans 6 steps"
**Cause:** Données 5min incomplètes (début/fin de période)
**Solution:**
- C'est normal pour les bougies en bordure
- Au moins 80% des bougies doivent être complètes
- Utiliser un dataset plus long si nécessaire

### Erreur: "Amplitude négative"
**Cause:** Bug dans le calcul ou OHLC invalide
**Solution:**
- Vérifier le calcul: `amplitude = (H - L) / O`
- Vérifier OHLC integrity en amont

### Erreur: "Data leakage détecté"
**Cause:** Feature utilise des données futures
**Solution:**
- Vérifier que la feature est causale
- Utiliser décalage temporel si nécessaire
- Ne PAS utiliser de données après timestamp[t]

### Erreur: "Open Z-Score mean éloigné de 0"
**Cause:** Dataset trop petit ou biais dans les données
**Solution:**
- Acceptable pour petits datasets (<500 bougies)
- Pour production: utiliser >5000 bougies
- Vérifier absence de trend fort dans les données

---

## 📈 Métriques de Qualité

### Dataset Production
- **Lignes:** >10,000 (pour training stable)
- **Features:** 15-30 (pas trop pour éviter overfitting)
- **Labels balance:** 45-55% (équilibré)
- **NaN:** <5% (sauf warm-up filtres)
- **Leakage:** Toutes features |corr| < 0.5
- **Corrélation idéale:** 10-20 features dans [0.1, 0.3]

### Dataset Multi-Actifs
- **Actifs:** 2-5 (BTC+ETH minimum)
- **Distribution:** 30-70% par actif (équilibré)
- **Normalisation:** PAR ACTIF (critique!)
- **Période commune:** Au moins 6 mois de données

---

## 💡 Bonnes Pratiques

1. **Toujours visualiser** les données avant entraînement
2. **Tester sur échantillon** avant dataset complet
3. **Valider chaque étape** séparément
4. **Sauvegarder les graphiques** pour référence
5. **Documenter les anomalies** dans le rapport
6. **Versionner les datasets** (btc_v1.csv, btc_v2.csv, etc.)
7. **Comparer les statistiques** entre versions

---

## 🎓 Pour Aller Plus Loin

### Créer un Nouveau Test
```python
# tests/test_my_feature.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_my_feature():
    """Test de ma nouvelle feature."""
    from my_module import my_function

    # Test
    result = my_function(input_data)

    # Assertions
    assert result is not None, "Result shouldn't be None"
    assert len(result) > 0, "Result shouldn't be empty"

    print("✅ Test passed!")

if __name__ == '__main__':
    test_my_feature()
```

### Ajouter une Visualisation
```python
import matplotlib.pyplot as plt

def visualize_feature(df, feature_name):
    """Visualise une feature."""
    plt.figure(figsize=(12, 6))

    # Histogram
    plt.subplot(1, 2, 1)
    df[feature_name].hist(bins=50)
    plt.title(f'Distribution: {feature_name}')

    # Time series
    plt.subplot(1, 2, 2)
    plt.plot(df[feature_name])
    plt.title(f'Evolution: {feature_name}')

    plt.savefig(f'tests/validation_output/{feature_name}.png')
    plt.close()
```

---

## 📞 Support

Si les tests échouent de manière persistante:
1. Lire le rapport complet: `tests/validation_output/validation_report.txt`
2. Examiner les visualisations
3. Vérifier les logs détaillés
4. Tester avec données synthétiques
5. Consulter la documentation du pipeline

---

**Dernière mise à jour:** 2026-01-01
**Version tests:** 1.0
**Auteur:** Pipeline Validation Team
