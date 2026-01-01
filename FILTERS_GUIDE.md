# Guide des Filtres de Signal

Ce document compare les différents filtres disponibles et donne des recommandations d'utilisation.

## 📊 Filtres Disponibles

### 1. **signal_filtfilt** (PRINCIPAL) ⭐

**Méthode de référence du projet**

```python
from filters import signal_filtfilt
filtered = signal_filtfilt(rsi, step=0.25, order=3)
```

- **Type**: Butterworth lowpass + filtfilt
- **Causalité**: NON-CAUSAL (utilise le futur)
- **Usage**: UNIQUEMENT pour labels
- **Paramètres**:
  - `step`: 0.2 (fort), 0.25 (recommandé), 0.3 (léger)
  - `order`: 3 (défaut)

**Avantages**:
- ✅ Pas de déphasage (filtfilt bidirectionnel)
- ✅ Lissage précis et contrôlable
- ✅ Rapide

**Inconvénients**:
- ❌ Utilise le futur (non-causal)

---

### 2. **kalman_filter** (AVANCÉ)

```python
from filters import kalman_filter
filtered = kalman_filter(rsi, process_variance=0.01, measurement_variance=0.1)
```

- **Type**: Filtre de Kalman (smoother)
- **Causalité**: NON-CAUSAL avec smoother
- **Usage**: Labels ou combinaison avec autres filtres

**Avantages**:
- ✅ Optimal pour signaux bruités
- ✅ Modèle probabiliste
- ✅ Très utilisé en finance

**Inconvénients**:
- ❌ Plus lent que Butterworth
- ❌ Tuning des variances nécessaire

**Recommandation paramètres**:
- `process_variance`: 0.001 (conservatif) à 0.1 (agressif)
- `measurement_variance`: 0.1 (typique)

---

### 3. **hp_filter** (Hodrick-Prescott)

```python
from filters import hp_filter
result = hp_filter(close_prices, lamb=400)
trend = result['trend']
cycle = result['cycle']
```

- **Type**: Séparation tendance/cycle
- **Causalité**: NON-CAUSAL
- **Usage**: Extraction de tendance long-terme

**Avantages**:
- ✅ Sépare tendance et cycle
- ✅ Très utilisé en économétrie
- ✅ Pas de déphasage

**Inconvénients**:
- ❌ Sensible aux endpoints
- ❌ Pas adapté pour signaux haute fréquence

**Recommandation lambda**:
- Crypto intraday: 100-400
- Données journalières: 1600
- Données mensuelles: 129600

---

### 4. **wavelet_denoise** (EXCELLENT POUR CRYPTO)

```python
from filters import wavelet_denoise
denoised = wavelet_denoise(rsi, wavelet='db4', level=3)
```

- **Type**: Décomposition en ondelettes
- **Causalité**: NON-CAUSAL
- **Usage**: Débruitage multi-échelle

**Avantages**:
- ✅ Multi-échelle (adaptatif)
- ✅ Excellent pour signaux non-stationnaires
- ✅ Préserve les discontinuités (pics de prix)
- ✅ Très adapté aux cryptomonnaies

**Inconvénients**:
- ❌ Plus complexe à paramétrer
- ❌ Plus lent

**Ondelettes recommandées**:
- `db4`, `db8`: Daubechies (bon compromis)
- `sym4`: Symlets (symétrique)
- `coif3`: Coiflets (régulier)

**Niveaux**:
- Level 2-3: Bruit haute fréquence
- Level 4-5: Tendances moyennes

---

### 5. **loess_smoothing** (ROBUSTE AUX OUTLIERS)

```python
from filters import loess_smoothing
smoothed = loess_smoothing(rsi, frac=0.1)
```

- **Type**: Locally Weighted Regression
- **Causalité**: NON-CAUSAL
- **Usage**: Lissage robuste

**Avantages**:
- ✅ Très robuste aux outliers
- ✅ Adaptable localement
- ✅ Pas de distribution assumée

**Inconvénients**:
- ❌ Très lent (O(n²))
- ❌ Pas adapté pour gros datasets

**Recommandation frac**:
- 0.05-0.1: Lissage léger
- 0.2-0.3: Lissage moyen

---

### 6. **emd_filter** (EXPERIMENTAL)

```python
from filters import emd_filter
result = emd_filter(close_prices, n_imfs=3)
filtered = result['filtered']
```

- **Type**: Empirical Mode Decomposition
- **Causalité**: NON-CAUSAL
- **Usage**: Décomposition en modes intrinsèques

**Avantages**:
- ✅ Adaptatif (sans paramètres prédéfinis)
- ✅ Sépare les fréquences naturellement
- ✅ Très puissant pour signaux complexes

**Inconvénients**:
- ❌ Très lent
- ❌ Instable (mode mixing)
- ❌ Non-déterministe

**Quand l'utiliser**:
- Signaux très complexes multi-fréquences
- Analyse exploratoire
- Pas pour production (trop lent)

---

### 7. **ensemble_filter** (ROBUSTE) ⭐

```python
from filters import ensemble_filter
filtered = ensemble_filter(
    rsi,
    filters=['signal_filtfilt', 'kalman', 'wavelet'],
    weights=[0.5, 0.3, 0.2]
)
```

- **Type**: Combinaison de plusieurs filtres
- **Causalité**: NON-CAUSAL
- **Usage**: Maximiser la robustesse

**Avantages**:
- ✅ Plus robuste qu'un seul filtre
- ✅ Réduit le risque de sur-lissage
- ✅ Combine les forces de chaque filtre

**Inconvénients**:
- ❌ Plus lent (calcule N filtres)
- ❌ Complexité accrue

**Combinaisons recommandées**:

```python
# Combinaison équilibrée
filters=['signal_filtfilt', 'kalman']
weights=[0.6, 0.4]

# Combinaison robuste
filters=['signal_filtfilt', 'kalman', 'wavelet']
weights=[0.5, 0.3, 0.2]

# Combinaison ultra-robuste
filters=['signal_filtfilt', 'kalman', 'hp']
weights=[0.4, 0.3, 0.3]
```

---

## 🎯 Recommandations d'Utilisation

### Pour les Labels (Cible de Prédiction)

**Option 1: Simple et Rapide** (RECOMMANDÉ)
```python
filtered = signal_filtfilt(rsi, step=0.25)
```

**Option 2: Plus Robuste**
```python
filtered = ensemble_filter(
    rsi,
    filters=['signal_filtfilt', 'kalman'],
    weights=[0.6, 0.4],
    step=0.25,
    process_variance=0.01
)
```

**Option 3: Maximum de Qualité** (lent)
```python
filtered = ensemble_filter(
    rsi,
    filters=['signal_filtfilt', 'kalman', 'wavelet'],
    weights=[0.5, 0.3, 0.2],
    step=0.25,
    process_variance=0.01,
    wavelet='db4',
    level=3
)
```

---

## 📈 Comparaison des Performances

| Filtre | Vitesse | Qualité | Robustesse | Complexité |
|--------|---------|---------|------------|------------|
| signal_filtfilt | ⚡⚡⚡⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 🔧 |
| kalman_filter | ⚡⚡⚡⚡ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 🔧🔧 |
| hp_filter | ⚡⚡⚡⚡ | ⭐⭐⭐ | ⭐⭐ | 🔧🔧 |
| wavelet_denoise | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 🔧🔧🔧 |
| loess_smoothing | ⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 🔧 |
| emd_filter | ⚡ | ⭐⭐⭐⭐ | ⭐⭐ | 🔧🔧🔧🔧 |
| ensemble_filter | ⚡⚡ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 🔧🔧🔧 |

---

## 🔬 Tests et Expérimentations

### Comparer les Filtres

```python
from filters import apply_filter_with_validation

# Test signal_filtfilt
result1 = apply_filter_with_validation(rsi, filter_type='signal_filtfilt', step=0.25)
print(f"Smoothing ratio: {result1['stats']['smoothing_ratio']:.3f}")

# Test kalman
result2 = apply_filter_with_validation(rsi, filter_type='kalman')
print(f"Smoothing ratio: {result2['stats']['smoothing_ratio']:.3f}")

# Test wavelet
result3 = apply_filter_with_validation(rsi, filter_type='wavelet', wavelet='db4')
print(f"Smoothing ratio: {result3['stats']['smoothing_ratio']:.3f}")
```

### Optimiser les Paramètres

```python
import numpy as np
import matplotlib.pyplot as plt

steps = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
smoothing_ratios = []

for step in steps:
    filtered = signal_filtfilt(rsi, step=step)
    ratio = np.std(filtered) / np.std(rsi)
    smoothing_ratios.append(ratio)

plt.plot(steps, smoothing_ratios, marker='o')
plt.xlabel('Step parameter')
plt.ylabel('Smoothing ratio (std_filtered / std_original)')
plt.title('Impact du paramètre step sur le lissage')
plt.grid(True)
plt.show()
```

---

## 💡 Suggestions Avancées

### 1. **Filtre Adaptatif**

Ajuster le paramètre de lissage en fonction de la volatilité:

```python
from filters import signal_filtfilt
import numpy as np

# Calculer la volatilité locale
volatility = rsi.rolling(window=20).std()

# Ajuster step en fonction de la volatilité
# Haute volatilité = lissage fort (step faible)
# Basse volatilité = lissage léger (step élevé)
step_adaptive = 0.15 + 0.15 * (1 / (1 + volatility / volatility.mean()))

# Appliquer le filtre par segments
# (simplifié, implémentation complète nécessaire)
```

### 2. **Double Filtrage**

Pour maximiser la qualité (lent):

```python
# 1er pass: Wavelet denoising
denoised = wavelet_denoise(rsi, wavelet='db4', level=3)

# 2ème pass: signal_filtfilt
filtered = signal_filtfilt(denoised, step=0.3)
```

### 3. **Détection de Régime**

Utiliser différents filtres selon le régime de marché:

```python
from filters import hp_filter, signal_filtfilt

# Détecter le régime avec HP filter
hp_result = hp_filter(close_prices, lamb=400)
cycle = hp_result['cycle']

# Marché en tendance: utiliser signal_filtfilt
# Marché en range: utiliser kalman (plus conservatif)
```

---

## ✅ Checklist de Validation

Après avoir appliqué un filtre:

- [ ] Visualiser signal original vs filtré
- [ ] Vérifier que smoothing_ratio ∈ [0.3, 0.8]
- [ ] Calculer la pente et vérifier la distribution
- [ ] Tester sur différentes périodes de marché
- [ ] Comparer avec d'autres filtres
- [ ] Backtest avec les labels générés

---

## 📚 Ressources

- **signal_filtfilt**: scipy.signal.butter + scipy.signal.filtfilt
- **Kalman**: [pykalman documentation](https://pykalman.github.io/)
- **HP Filter**: [statsmodels hp_filter](https://www.statsmodels.org/stable/generated/statsmodels.tsa.filters.hp_filter.hpfilter.html)
- **Wavelet**: [PyWavelets](https://pywavelets.readthedocs.io/)
- **EMD**: [PyEMD](https://pyemd.readthedocs.io/)

---

**Note**: Tous les filtres listés sont NON-CAUSAUX (utilisent le futur). Ils doivent être utilisés UNIQUEMENT pour générer les labels, jamais pour les features.
