# Prochaines Étapes - Profitability Relabeling

**Date**: 2026-01-06
**Statut**: Scripts prêts, nécessite mise à jour du pipeline de données

---

## ✅ Ce Qui a Été Créé

### 1. Scripts de Test

**Proposition A - Smart Hybrid**:
- `tests/test_smart_hybrid_relabeling.py`
- Règles: Durée 3 TOUT supprimé, Durée 4-5 SI Vol Q4

**Proposition B - Profitability** 🏆:
- `tests/test_profitability_relabeling.py`
- Règles: Si Max Return < Frais → Relabeler WEAK

**Script comparatif**:
- `tests/test_both_relabeling_proposals.sh`
- Teste les 2 propositions + variantes

### 2. Documentation

- `docs/PROFITABILITY_RELABELING_GUIDE.md` - Guide complet
- Explications théoriques, littérature ML, workflow

---

## ⚠️ PROBLÈME IDENTIFIÉ

**Les datasets actuels ne contiennent pas les métadonnées nécessaires.**

### Métadonnées Requises

| Métadonnée | Usage | Script |
|------------|-------|--------|
| **prices** | Calculer PnL futur (Max Return) | Profitability (B) ✅ |
| **duration** | Identifier durées STRONG courtes | Smart Hybrid (A) |
| **vol_rolling** | Identifier Q4 volatilité | Smart Hybrid (A) |

**Actuellement sauvegardé** dans `dataset_*_dual_binary_kalman.npz`:
```python
np.savez_compressed(
    X_train, Y_train,
    X_val, Y_val,
    X_test, Y_test,
    metadata=json.dumps(...)
)
```

**Manquant**: prices, duration, vol_rolling pour chaque split.

---

## 🚀 SOLUTION 1: Mise à Jour du Script de Préparation (RECOMMANDÉ)

### Modifier `src/prepare_data_purified_dual_binary.py`

**Étape 1**: Ajouter calcul des métadonnées dans `prepare_indicator_dataset`:

```python
def prepare_indicator_dataset(df: pd.DataFrame, asset_name: str, indicator: str,
                              feature_cols: list, clip_value: float = 0.10) -> tuple:
    """
    ...
    Returns:
        (X, Y, indices, metadata) pour cet indicateur
          metadata = {'prices': array, 'duration': array, 'vol_rolling': array}
    """
    # ... code existant ...

    # AJOUTER: Calculer métadonnées
    metadata = {}

    # 1. Prices (Close)
    metadata['prices'] = df['close'].values[indices[:, 1]]  # Prix aux indices de labels

    # 2. Duration (nombre de périodes consécutives STRONG)
    force_col = df[f'{indicator}_force'].values
    duration = calculate_strong_duration(force_col)
    metadata['duration'] = duration[indices[:, 1]]

    # 3. Vol Rolling (écart-type des returns sur window=20)
    c_ret = df['c_ret'].values
    vol_rolling = pd.Series(c_ret).rolling(window=20).std().values
    metadata['vol_rolling'] = vol_rolling[indices[:, 1]]

    return X, Y, indices, metadata
```

**Étape 2**: Fonction helper pour calculer Duration:

```python
def calculate_strong_duration(force_labels: np.ndarray) -> np.ndarray:
    """
    Calcule le nombre de périodes consécutives STRONG pour chaque position.

    Returns:
        duration: array de même taille que force_labels
          duration[i] = nombre de périodes consécutives STRONG jusqu'à i
    """
    duration = np.zeros(len(force_labels), dtype=int)
    count = 0

    for i in range(len(force_labels)):
        if force_labels[i] == 1:  # STRONG
            count += 1
            duration[i] = count
        else:  # WEAK
            count = 0
            duration[i] = 0

    return duration
```

**Étape 3**: Modifier `split_chronological` pour gérer les métadonnées:

```python
def split_chronological(X, Y, indices, metadata):
    """
    Split chronologique avec métadonnées.

    Returns:
        {
            'train': (X_train, Y_train, metadata_train),
            'val': (X_val, Y_val, metadata_val),
            'test': (X_test, Y_test, metadata_test)
        }
    """
    # ... code existant pour split X, Y ...

    # Split métadonnées
    metadata_train = {k: v[:train_size] for k, v in metadata.items()}
    metadata_val = {k: v[train_size:train_size+val_size] for k, v in metadata.items()}
    metadata_test = {k: v[train_size+val_size:] for k, v in metadata.items()}

    return {
        'train': (X_train, Y_train, metadata_train),
        'val': (X_val, Y_val, metadata_val),
        'test': (X_test, Y_test, metadata_test)
    }
```

**Étape 4**: Modifier `prepare_and_save_all` pour concaténer et sauvegarder:

```python
# Ligne ~500: Concaténation
datasets = {
    'rsi': {
        'train': {'X': [], 'Y': [], 'prices': [], 'duration': [], 'vol_rolling': []},
        'val': {...},
        'test': {...}
    },
    ...
}

# Après préparation de chaque asset:
for split_name in ['train', 'val', 'test']:
    X, Y, meta = splits[split_name]
    datasets[indicator][split_name]['X'].append(X)
    datasets[indicator][split_name]['Y'].append(Y)
    datasets[indicator][split_name]['prices'].append(meta['prices'])
    datasets[indicator][split_name]['duration'].append(meta['duration'])
    datasets[indicator][split_name]['vol_rolling'].append(meta['vol_rolling'])

# Ligne ~580: Sauvegarde
prices_train = np.concatenate(datasets[indicator]['train']['prices'])
prices_val = np.concatenate(datasets[indicator]['val']['prices'])
prices_test = np.concatenate(datasets[indicator]['test']['prices'])

duration_train = np.concatenate(datasets[indicator]['train']['duration'])
duration_val = np.concatenate(datasets[indicator]['val']['duration'])
duration_test = np.concatenate(datasets[indicator]['test']['duration'])

vol_rolling_train = np.concatenate(datasets[indicator]['train']['vol_rolling'])
vol_rolling_val = np.concatenate(datasets[indicator]['val']['vol_rolling'])
vol_rolling_test = np.concatenate(datasets[indicator]['test']['vol_rolling'])

np.savez_compressed(
    output_path,
    X_train=X_train, Y_train=Y_train,
    X_val=X_val, Y_val=Y_val,
    X_test=X_test, Y_test=Y_test,

    # AJOUTER: Métadonnées
    prices_train=prices_train,
    prices_val=prices_val,
    prices_test=prices_test,

    duration_train=duration_train,
    duration_val=duration_val,
    duration_test=duration_test,

    vol_rolling_train=vol_rolling_train,
    vol_rolling_val=vol_rolling_val,
    vol_rolling_test=vol_rolling_test,

    metadata=json.dumps(metadata)
)
```

**Ensuite régénérer datasets**:
```bash
python src/prepare_data_purified_dual_binary.py --assets BTC ETH BNB ADA LTC
```

---

## 🏃 SOLUTION 2: Script Wrapper (RAPIDE mais moins propre)

Créer `tests/test_profitability_with_reload.py` qui:
1. Charge le dataset .npz
2. Recharge les CSV bruts pour récupérer les prix
3. Recalcule duration et vol_rolling à la volée
4. Applique Profitability Relabeling
5. Compare Oracle AVANT vs APRÈS

**Avantage**: Pas besoin de modifier prepare_data
**Inconvénient**: Plus lent, code dupliqué

---

## 📋 RECOMMANDATION

### Option A: Mise à Jour du Pipeline (RECOMMANDÉ) 🏆

**Avantages**:
- ✅ Métadonnées sauvegardées une fois pour toutes
- ✅ Réutilisables pour tous futurs tests
- ✅ Pipeline propre et complet
- ✅ Permet tests rapides ensuite

**Temps requis**: ~30 min (modifications) + 5 min (régénération datasets)

**Workflow**:
1. Je modifie `prepare_data_purified_dual_binary.py` (ajouter fonctions)
2. Vous exécutez régénération datasets
3. Vous testez Proposition B
4. Analyse résultats → GO/NO-GO

---

### Option B: Script Wrapper (SI URGENT)

**Avantages**:
- ✅ Pas de modifications du pipeline
- ✅ Test immédiat possible

**Inconvénients**:
- ❌ Plus lent à chaque exécution
- ❌ Code moins propre
- ❌ Dépendance aux CSV bruts

**Temps requis**: ~15 min (script wrapper) + 2 min (test)

---

## 🎯 MA RECOMMANDATION

**Choisir Option A (mise à jour pipeline)** car:

1. **Proposition B est l'approche finale** - Vous allez l'utiliser pour de bon
2. **Métadonnées utiles pour d'autres analyses** - Duration/Vol pour stats
3. **Une fois fait, c'est fait** - Tous futurs tests en bénéficient
4. **Temps total équivalent** - 30 min setup vs 15×N min à chaque test

**Plan d'action**:
1. ✅ Je modifie `prepare_data_purified_dual_binary.py` (FAIT dans le prochain message)
2. ⏳ Vous régénérez datasets (~5 min)
3. ⏳ Vous testez Proposition B (~10 secondes)
4. ⏳ Analyse résultats → Décision finale

**Si résultats Proposition B positifs**:
```
ΔWin Rate > +3%
ΔPrédictivité > +40%
ΔPnL Total > -30%
```

**Alors**: GO pour relabeling complet + réentraînement
**Gain attendu IA**: Win Rate 14% → 22-25% (+8-11%)

---

**Voulez-vous que je procède à Option A (modification du script) ?**

