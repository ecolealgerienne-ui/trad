# ⚠️ RÈGLE CRITIQUE : Éviter le Data Leakage Temporel

## 🚨 Le Problème : Data Leakage sur Séries Temporelles

### Qu'est-ce que le Data Leakage ?

**Data leakage** = Le modèle voit des informations du futur pendant l'entraînement

Sur séries temporelles, cela arrive quand :
1. On **shuffle** les données AVANT de split train/val/test
2. Des séquences temporellement **proches** se retrouvent dans train ET test
3. Le modèle "triche" en reconnaissant le contexte immédiat

### Exemple Concret du Problème

```python
# Données : 200k bougies BTC+ETH
# On crée des séquences de 12 timesteps

# Séquence 1 : bougies [0-11]   → X1
# Séquence 2 : bougies [1-12]   → X2  ← Chevauche avec X1 !
# Séquence 3 : bougies [2-13]   → X3  ← Chevauche avec X1 et X2 !
# ...

# ❌ SI ON SHUFFLE AVANT SPLIT :
all_sequences = [X1, X2, X3, ..., X200000]
shuffle(all_sequences)  # ← ERREUR ICI !

train, val, test = split(all_sequences, [0.7, 0.15, 0.15])

# Résultat catastrophique :
# - X1 dans train
# - X2 dans test  ← 11 timesteps en commun avec X1 !
# - X3 dans val   ← 10 timesteps en commun avec X1 !

# L'IA "voit" le futur !
```

### Impact sur l'Accuracy

**Avec shuffle global (DATA LEAKAGE)** :
- Test accuracy : **90-95%** ✅ (trop beau pour être vrai!)
- Production accuracy : **50-55%** ❌ (hasard!)

**Raison** : Le modèle a vu des séquences quasi-identiques en train

**Avec split temporel (CORRECT)** :
- Test accuracy : **65-75%** ✓ (réaliste)
- Production accuracy : **65-75%** ✓ (cohérent!)

---

## ✅ La Solution : Split Temporel STRICT

### Principe

```
|<-------- Train (70%) ------>|<- Val (15%) ->|<- Test (15%) ->|
[bougies 0 → 140k]            [140k → 170k]   [170k → 200k]
     Passé                      Présent          Futur
```

**Règle d'or** : Train sur le PASSÉ, valide sur le FUTUR

### Implémentation Correcte

```python
import pandas as pd
import numpy as np
from constants import TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, RANDOM_SEED

def temporal_split(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split temporel STRICT sans data leakage.

    Args:
        data : DataFrame de séries temporelles
        train_ratio : Proportion de données pour train
        val_ratio : Proportion pour validation
        test_ratio : Proportion pour test

    Returns:
        train, val, test : DataFrames splittés temporellement

    ⚠️ IMPORTANT : PAS de shuffle avant split!
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 0.001, \
        "Les ratios doivent sommer à 1.0"

    n_total = len(data)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    # Split TEMPOREL (ordre chronologique préservé)
    train_data = data.iloc[:n_train].copy()
    val_data = data.iloc[n_train:n_train+n_val].copy()
    test_data = data.iloc[n_train+n_val:].copy()

    print(f"📊 Split temporel:")
    print(f"  Train: {len(train_data):,} bougies ({train_ratio:.0%})")
    print(f"  Val:   {len(val_data):,} bougies ({val_ratio:.0%})")
    print(f"  Test:  {len(test_data):,} bougies ({test_ratio:.0%})")

    # ✅ Shuffle APRÈS split (uniquement train)
    # Cela mélange l'ordre des batches SANS introduire de leakage
    train_data = train_data.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"  ✅ Train shuffled (évite biais d'ordre)")

    return train_data, val_data, test_data
```

### Exemple d'Utilisation

```python
# Charger et combiner BTC + ETH
btc_data = pd.read_csv('../data_trad/BTCUSD_all_5m.csv')
eth_data = pd.read_csv('../data_trad/ETHUSD_all_5m.csv')

# Prendre 100k de chaque
btc_data = btc_data.tail(100000).reset_index(drop=True)
eth_data = eth_data.tail(100000).reset_index(drop=True)

# Concaténer (ordre chronologique préservé)
all_data = pd.concat([btc_data, eth_data], ignore_index=True)
# Total : 200k bougies

# Split TEMPOREL (pas de shuffle global!)
train, val, test = temporal_split(all_data,
                                   train_ratio=0.7,
                                   val_ratio=0.15,
                                   test_ratio=0.15)

# Résultat :
# Train : 140k bougies (shuffled internement)
# Val   : 30k bougies (ordre chrono)
# Test  : 30k bougies (ordre chrono)
```

---

## 🔬 Validation : Comment Détecter le Leakage

### Test Simple

Si vous avez un data leakage, vous observerez :

1. **Test accuracy >> Train accuracy**
   - Normal : Test ≈ Train (± 2-5%)
   - Leakage : Test > Train (signe de triche)

2. **Test loss << Train loss**
   - Normal : Test ≥ Train
   - Leakage : Test < Train (trop facile!)

3. **Production accuracy << Test accuracy**
   - Normal : Prod ≈ Test (± 5%)
   - Leakage : Prod << Test (échec en prod!)

### Exemple de Détection

```python
# Pendant l'entraînement
train_acc = 0.70  # 70%
val_acc = 0.72    # 72%
test_acc = 0.94   # 94% ← ⚠️ SUSPECT !

# Si test >> train : probablement du leakage !

# Vérification en production
prod_acc = 0.52   # 52% ← ❌ CONFIRME LE LEAKAGE !
```

---

## 📊 Comparaison : Shuffle vs Temporel

| Aspect | Shuffle Global ❌ | Split Temporel ✅ |
|--------|------------------|-------------------|
| **Data Leakage** | OUI (massif) | NON |
| **Test Accuracy** | 90-95% (faux) | 65-75% (réel) |
| **Prod Accuracy** | 50-55% (hasard) | 65-75% (cohérent) |
| **Réalisme** | Non (triche) | Oui (futur inconnu) |
| **Train/Test gap** | Test > Train | Test ≈ Train |
| **Robustesse** | Faible | Forte |

---

## 🎯 Règles d'Or

### ✅ À FAIRE

1. **Split temporel STRICT** : Train sur passé, test sur futur
2. **Shuffle APRÈS split** : Uniquement dans train
3. **Valider la cohérence** : Test accuracy ≈ Prod accuracy
4. **Surveiller les métriques** : Test pas >> Train

### ❌ À NE JAMAIS FAIRE

1. ❌ **Shuffle avant split** : Data leakage garanti
2. ❌ **Shuffle val/test** : Détruit l'ordre temporel
3. ❌ **Utiliser K-fold cross-validation** : Inapproprié pour séries temporelles
4. ❌ **Ignorer l'ordre chronologique** : Perd la structure temporelle

---

## 🧪 Exemple de Code Production

```python
from constants import (
    BTC_DATA_FILE, ETH_DATA_FILE,
    BTC_CANDLES, ETH_CANDLES,
    TRIM_EDGES,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT
)

def load_and_split_data():
    """
    Charge BTC+ETH et fait un split temporel correct.

    Returns:
        train, val, test : DataFrames prêts pour l'entraînement
    """
    # Charger données
    print("📂 Chargement des données...")
    btc = pd.read_csv(BTC_DATA_FILE)
    eth = pd.read_csv(ETH_DATA_FILE)

    # Prendre les dernières N bougies
    btc = btc.tail(BTC_CANDLES).reset_index(drop=True)
    eth = eth.tail(ETH_CANDLES).reset_index(drop=True)

    print(f"  BTC: {len(btc):,} bougies")
    print(f"  ETH: {len(eth):,} bougies")

    # Trim edges (warm-up + artifacts)
    btc = btc.iloc[TRIM_EDGES:-TRIM_EDGES].reset_index(drop=True)
    eth = eth.iloc[TRIM_EDGES:-TRIM_EDGES].reset_index(drop=True)

    print(f"  Après trim ({TRIM_EDGES} début+fin):")
    print(f"    BTC: {len(btc):,}")
    print(f"    ETH: {len(eth):,}")

    # Combiner
    all_data = pd.concat([btc, eth], ignore_index=True)
    print(f"  Total: {len(all_data):,} bougies")

    # ⚠️ CRITIQUE : Split TEMPOREL (pas shuffle!)
    train, val, test = temporal_split(
        all_data,
        train_ratio=TRAIN_SPLIT,
        val_ratio=VAL_SPLIT,
        test_ratio=TEST_SPLIT
    )

    return train, val, test

# Utilisation
train_data, val_data, test_data = load_and_split_data()
```

---

## 📚 Références

### Articles Académiques

1. **"Time Series Forecasting: Preventing Data Leakage"** - Google Research
2. **"Common Pitfalls in Time Series Analysis"** - Forecasting Journal
3. **"Walk-Forward Analysis in Trading Systems"** - Journal of Trading

### Points Clefs de la Littérature

- **Principe de causalité** : Le futur ne peut pas influencer le passé
- **Walk-forward validation** : Train sur passé, test sur futur (comme trading réel)
- **K-fold inapproprié** : Les folds mélangent passé et futur

---

## ⚠️ Cas Particulier : Normalisation

### Attention au Leakage dans la Normalisation !

```python
# ❌ MAUVAIS (Data leakage via normalisation)
# Calculer mean/std sur TOUTES les données
mean = all_data.mean()
std = all_data.std()

train_normalized = (train_data - mean) / std  # ← Utilise info du test !
test_normalized = (test_data - mean) / std

# ✅ CORRECT (Fit sur train uniquement)
# Calculer mean/std sur TRAIN seulement
mean = train_data.mean()
std = train_data.std()

train_normalized = (train_data - mean) / std  # ← OK
val_normalized = (val_data - mean) / std      # ← OK (utilise stats du train)
test_normalized = (test_data - mean) / std    # ← OK (utilise stats du train)
```

**Règle** : Les statistiques (mean, std, min, max) viennent TOUJOURS du train uniquement!

---

## ✅ Checklist de Validation

Avant de lancer l'entraînement, vérifier :

- [ ] Split fait TEMPORELLEMENT (ordre chrono préservé)
- [ ] Train = 70% premiers, Val = 15% milieu, Test = 15% derniers
- [ ] Shuffle fait APRÈS split (uniquement sur train)
- [ ] Normalisation calculée sur TRAIN seulement
- [ ] Pas de K-fold cross-validation
- [ ] Test accuracy ≈ Train accuracy (± 5%)
- [ ] Métriques cohérentes (Test loss ≥ Train loss)

---

**Date** : 2026-01-01
**Version** : 1.0
**Status** : RÈGLE CRITIQUE - Non négociable
**Impact** : Différence entre 50% (hasard) et 70%+ (réel) en production
