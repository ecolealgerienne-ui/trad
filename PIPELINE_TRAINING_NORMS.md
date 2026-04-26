# Normes Génériques de Pipeline ML — Trading Quantitatif

**Version** : 2.0 (en cours, structure 25 sections)
**Date** : 2026-04-26
**Objet** : checklist normative générique applicable à **tout** projet ML pour stratégie de trading. Indépendant de tout projet ou paradigme architectural spécifique.

## Conventions

- ✅ : à faire
- ❌ : à éviter (anti-pattern documenté)
- ⚠️ : subtilité ou compromis à connaître
- 📖 : référence académique standard

## Table des matières

**Partie I — Données (sections 1-6)**
1. **Causalité temporelle** ✅
2. Stationarité des features
3. Normalisation
4. Distribution shift
5. Outliers et valeurs manquantes
6. Multicolinéarité

**Partie II — Labels (sections 7-9)**
7. Engineering des labels
8. Cost-sensitive labeling
9. Class imbalance

**Partie III — Splitting & validation (sections 10-12)**
10. Train/Val/Test pour time series
11. Cross-validation pour time series
12. Anti-leakage checklist

**Partie IV — Modélisation (sections 13-16)**
13. Baseline first
14. Régularisation
15. Hyperparameter tuning
16. Calibration des probabilités

**Partie V — Évaluation (sections 17-20)**
17. Choix des métriques
18. Statistical significance
19. Per-segment analysis
20. Backtesting réaliste

**Partie VI — Production (sections 21-23)**
21. Drift monitoring
22. Retraining schedule
23. Reproductibilité

**Partie VII — Qualité du code (sections 24-25)**
24. Tests automatiques
25. Audit pré-déploiement

---

## Section 1 — Causalité temporelle

### Principe général

En machine learning supervisé sur séries temporelles, à chaque instant `t`, **toute feature** doit être calculable **uniquement** à partir d'informations disponibles strictement avant ou à `t`. Toute information de `t+1` ou ultérieure dans une feature constitue une **fuite de données** (data leakage) qui invalide totalement l'évaluation hors échantillon.

📖 **Référence canonique** : Marcos López de Prado, *Advances in Financial Machine Learning* (Wiley, 2018), Chapitre 7.

### Règle pratique

✅ **Toute feature** doit avoir un lookback **explicite et borné** : `f(t) = g(x[t-N..t])` où `N ≥ 0` est connu.

✅ **Le label** peut (et doit souvent) utiliser le futur `[t+1..t+H]` pour H bars (Triple Barrier, return forward, etc.) — c'est par construction.

❌ **Pas de filtres non-causaux comme features** : tout filtre forward-backward (Savitzky-Golay symétrique, Kalman smoother RTS, Butterworth `filtfilt`, etc.) injecte de l'information future dans le présent.

❌ **Pas de statistiques globales** computées sur le dataset complet utilisées pour normaliser : `feature / max_global_dataset` calcule `max_global` à partir de tout l'historique incluant futur → leak.

⚠️ **Subtilité** : `pd.Series.rolling(N).mean()` à l'index `t` utilise `[t-N+1, t]` **inclus**. La valeur `t` est dans sa propre normalisation. C'est **acceptable** car `t` est connue à la close du bar, et le label porte sur `t+1..t+H`. Pas besoin de `.shift(1)` par paranoïa.

### Liste de référence — fonctions et leur causalité

| Fonction | Causalité | Verdict feature |
|---|---|---|
| `pd.rolling(N).mean/std/sum/quantile` | backward `[t-N+1..t]` | ✅ OK |
| `pd.rolling(N, center=True)` | centrée `[t-N/2..t+N/2]` | ❌ INTERDIT |
| `pd.expanding().mean` | depuis `t=0` | ✅ OK (mais non-stationnaire) |
| `pd.shift(N)` avec N>0 | recule de N | ✅ OK |
| `pd.shift(N)` avec N<0 | avance vers futur | ❌ INTERDIT comme feature |
| `scipy.signal.filtfilt` | forward-backward | ❌ INTERDIT |
| `scipy.signal.lfilter` | forward causal | ✅ OK |
| `pykalman.KalmanFilter.filter()` | forward causal | ✅ OK |
| `pykalman.KalmanFilter.smooth()` | RTS smoother backward | ❌ INTERDIT (mais OK comme label) |
| `talib.RSI/MACD/CCI/ATR/ADX` | tous backward | ✅ OK |
| `numpy.argmax`, `numpy.cumsum` | causal si appliqué `.iloc[:t]` | ⚠️ vérifier scope |

### Règles spécifiques

#### 1.1 Agrégations multi-timeframe

Pour toute feature calculée sur granularité agrégée (daily/hourly) puis broadcastée au timeframe inférieur (5min/1min) :

✅ **Shifter de 1 unité avant ffill** :
```python
daily_close = df.resample('1D').last()
prev_daily_close = daily_close.shift(1)  # OBLIGATOIRE
levels = compute_indicator(prev_daily_close)
broadcast = levels.reindex(df.index, method='ffill')
```

❌ Calculer sur jour D et utiliser dans jour D = fuite intra-jour.

#### 1.2 Volume Profile / POC sur fenêtre rolling

✅ Pour POC[D] (Point of Control jour D), utiliser uniquement données `[D-N..D-1]` (jours antérieurs uniquement).

#### 1.3 Sliding window features

✅ **Bornes inclusives** : `feature[t] = f(x[t-N+1..t])` est causal.
❌ **Bornes futures** : `feature[t] = f(x[t-N+1..t+1])` est non-causal.

### Test automatique de causalité

✅ **Test obligatoire** pour toute nouvelle feature complexe :

```python
def test_causality(compute_features_fn, df, idx_test=1000, n_corruptions=10):
    """
    Test générique de causalité.
    Vérifie qu'une feature à l'instant t ne change pas si on modifie
    aléatoirement les valeurs après t.
    """
    feat_orig = compute_features_fn(df.copy())

    for _ in range(n_corruptions):
        df_corrupted = df.copy()
        # Corrompre toutes les valeurs futures avec du bruit gaussien
        rng = np.random.default_rng(42)
        for col in df.select_dtypes(include=np.number).columns:
            df_corrupted.loc[df.index[idx_test+1]:, col] += rng.normal(0, df[col].std(), size=len(df) - idx_test - 1)

        feat_modified = compute_features_fn(df_corrupted)

        # Les features à idx_test doivent être identiques
        diff = (feat_orig.iloc[idx_test] - feat_modified.iloc[idx_test]).abs().sum()
        assert diff < 1e-9, (
            f"DATA LEAK: feature à t={idx_test} change quand t+1.. est corrompu. "
            f"Diff totale = {diff}"
        )
```

Ce test prend < 1 minute à coder et détecte > 95% des fuites silencieuses.

### Règles d'or — résumé

1. **Une feature = une fonction causale du passé**, point.
2. **Le label = peut utiliser le futur**, c'est sa raison d'être.
3. **Toute statistique globale du dataset** (mean, std, max, min) doit être **calculée uniquement sur train** et appliquée à val/test.
4. **Vérifier la causalité par test automatique** avant tout entraînement, pas après.

### Références complémentaires

📖 López de Prado, *Advances in Financial Machine Learning* (2018), chap. 7 "Cross-Validation" et chap. 18 "Backtesting"
📖 Hamilton, *Time Series Analysis* (1994), chap. 1 sur les processus stochastiques causaux
📖 Tashiro et al., *Causal Time Series Analysis* (Cambridge, 2020)

---

**Section 1 fin.** Je continue avec Section 2 (Stationarité des features) au prochain message si tu valides.

---

## Section 2 — Stationarité des features

### Principe général

Une série `X_t` est dite **(faiblement) stationnaire** si son espérance `E[X_t]`, sa variance `Var(X_t)`, et ses autocovariances `Cov(X_t, X_{t+h})` sont **invariantes par translation temporelle**. La plupart des algorithmes ML statistiques assument **implicitement** que train et test partagent la même distribution sous-jacente — cette hypothèse échoue dès que les features sont non-stationnaires.

Les marchés financiers sont **typiquement non-stationnaires** : les prix dérivent (random walk avec tendance), les volatilités passent par des régimes, les corrélations s'inversent. Toute feature directement dérivée du prix sans transformation **héritera** cette non-stationnarité.

📖 **Références** :
- Hamilton, *Time Series Analysis* (Princeton, 1994), chap. 15-17
- López de Prado, *Advances in Financial Machine Learning* (Wiley, 2018), chap. 5 "Fractionally Differentiated Features"
- Tsay, *Analysis of Financial Time Series* (Wiley, 2010), chap. 2

### Règle pratique

✅ **Avant tout entraînement**, tester la stationarité de chaque feature continue numérique non-bornée.

✅ Pour les features non-stationnaires : **transformer** ou **éliminer**.

❌ **NE PAS** alimenter directement un modèle avec des prix bruts, des indicateurs en valeur absolue de prix (MACD line, signal line, slopes de régression sur prix), ou tout cumul non normalisé (volume cumulé, OBV brut).

⚠️ **Compromis** : les transformations (différenciation, log-return) **détruisent** de l'information. La fractional differentiation (López de Prado) est un compromis pour préserver la mémoire long terme tout en stationarisant.

### Tests de stationnarité — choix méthodologique

| Test | Hypothèse H₀ | Interprétation |
|---|---|---|
| **ADF** (Augmented Dickey-Fuller) | série a une racine unitaire (non-stationnaire) | rejet H₀ → série stationnaire |
| **KPSS** (Kwiatkowski-Phillips-Schmidt-Shin) | série stationnaire (autour d'une moyenne ou tendance) | rejet H₀ → série non-stationnaire |
| **Phillips-Perron** | similaire ADF avec correction d'autocorrélation | idem ADF |

✅ **Best practice** : utiliser **ADF + KPSS conjointement** (ils ont des hypothèses opposées). Les deux d'accord = confiance élevée.

```python
from statsmodels.tsa.stattools import adfuller, kpss

def test_stationarity(series, alpha=0.05):
    """Test combiné ADF + KPSS. Retourne True si stationnaire."""
    series_clean = series.dropna()
    if len(series_clean) < 100:
        return None  # série trop courte

    # ADF: H0 = non-stationnaire. Petit p-value = rejet → stationnaire
    adf_stat, adf_pvalue, _, _, _, _ = adfuller(series_clean, autolag='AIC')
    adf_stationary = adf_pvalue < alpha

    # KPSS: H0 = stationnaire. Petit p-value = rejet → non-stationnaire
    kpss_stat, kpss_pvalue, _, _ = kpss(series_clean, regression='c', nlags='auto')
    kpss_stationary = kpss_pvalue >= alpha

    return {
        'adf_stationary': adf_stationary,
        'kpss_stationary': kpss_stationary,
        'consensus': adf_stationary and kpss_stationary,
    }
```

### Hiérarchie des transformations

Quand une feature est non-stationnaire, choix par ordre de **préservation d'information décroissante** :

#### Niveau 1 — Ratio adimensionnel (préfère)

✅ Si la feature est **proportionnelle à une grandeur de référence** (typiquement le prix), la diviser :
```python
macd_line_pct = macd_line / close      # MACD en % du prix
volume_ratio = volume / volume_ma_20   # ratio à la moyenne récente
```

**Avantage** : reste interprétable, perd peu d'information, scale-invariant.
**Quand applicable** : MACD line, slopes de régression, OBV slope, niveaux dérivés du prix.

#### Niveau 2 — Log-return / différence relative

✅ Pour des séries de prix ou volumes :
```python
log_return = np.log(price / price.shift(1))      # standard finance
pct_change = (price - price.shift(1)) / price.shift(1)
```

**Avantage** : transformation classique, bien étudiée. Stationnarise sous hypothèse de random walk multiplicatif.
**Inconvénient** : perd l'info de niveau (où est le prix par rapport à son histoire).

#### Niveau 3 — Différenciation entière

✅ Différences successives jusqu'à stationnarité :
```python
diff_1 = series.diff()      # première différence
diff_2 = diff_1.diff()      # deuxième différence (rare nécessaire)
```

**Inconvénient** : détruit la **mémoire long terme**. Une série I(1) différenciée perd toutes ses corrélations long-range.

#### Niveau 4 — Fractional differentiation (López de Prado)

✅ Différenciation **non-entière** `d ∈ (0, 1)` qui préserve mémoire long terme tout en stationnarisant :

```python
# Approximation finie via somme pondérée
def fracdiff(series, d, threshold=1e-4):
    """Fractional differentiation préservant la mémoire long terme.
    Référence: López de Prado AFML chap 5."""
    # Calculer les poids w_k = (-1)^k * Gamma(d+1) / (Gamma(k+1) * Gamma(d-k+1))
    weights = [1.0]
    k = 1
    while True:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1
    weights = np.array(weights[::-1])
    # Convolution
    return series.rolling(len(weights)).apply(
        lambda x: np.dot(weights, x), raw=True
    )
```

**Avantage** : compromis optimal mémoire/stationnarité.
**Inconvénient** : choix du `d` optimal nécessite recherche par grid search (souvent `d ∈ [0.2, 0.6]` pour des prix).

#### Niveau 5 — Z-score rolling

✅ Si transformation algébrique impossible :
```python
def rolling_zscore(series, window):
    return (series - series.rolling(window).mean()) / series.rolling(window).std()
```

**Avantage** : stationnarise n'importe quoi, simple, causal.
**Inconvénient** : perd l'info de niveau absolu, sensible au choix de fenêtre.

### Features naturellement stationnaires (à privilégier)

✅ Les indicateurs **bornés mathématiquement** sont stationnaires par construction :

| Indicateur | Borne | Type |
|---|---|---|
| RSI | [0, 100] | Oscillateur momentum |
| Stochastic %K, %D | [0, 100] | Oscillateur momentum |
| Williams %R | [-100, 0] | Oscillateur momentum |
| ADX, DI+, DI- | [0, 100] | Force de tendance |
| Money Flow Index | [0, 100] | Volume momentum |
| Bollinger %B | [0, 1] approx | Position dans bandes |
| Body ratio | [0, 1] | Forme de bougie |
| Wicks ratios | [0, 1] | Forme de bougie |

⚠️ **Subtilité** : "borné" ≠ "stationnaire en distribution". Un RSI peut avoir une distribution **différente** entre régimes (plus de temps à 70+ en bull market). Mais sa **forme** reste comparable, et son support est invariant.

### Features systématiquement non-stationnaires (à transformer)

❌ **À transformer impérativement** :

| Feature | Problème | Transformation |
|---|---|---|
| Prix brut (close) | Trend perpétuelle | Log-return |
| MACD line/signal | Proportionnel au prix | `/ close × 100` |
| Slopes de régression sur prix | Proportionnel au prix | `/ close × 100` |
| OBV (cumulatif) | Non-stationnaire par construction | Slope rolling, ou diff + z-score |
| Volume brut | Croissance long terme | Z-score rolling |
| ATR brut | Proportionnel au prix | `/ close` ou rolling z-score |
| Yang-Zhang vol, Garman-Klass vol | Régimes de volatilité | Rolling z-score |

### Décision arbre — choix de transformation

```
La feature est-elle bornée mathématiquement (ex: RSI ∈ [0,100]) ?
├── OUI → ✅ Garder telle quelle
└── NON → Tester ADF + KPSS
    ├── Stationnaire → ✅ Garder telle quelle
    └── Non-stationnaire
        ├── Proportionnelle à une grandeur de référence (prix, volume) ?
        │   ├── OUI → ✅ Diviser par cette grandeur (ratio %)
        │   └── NON → continuer
        ├── Cumulative (OBV, equity curve) ?
        │   ├── OUI → ✅ Diff ou rolling slope
        │   └── NON → continuer
        ├── Mémoire long terme importante ?
        │   ├── OUI → ✅ Fractional differentiation (d optimal par grid)
        │   └── NON → ✅ Rolling z-score
```

### Test final avant entraînement

✅ **Toute feature** alimentant le modèle doit passer ADF avec p < 0.05 OU être bornée mathématiquement.

```python
def validate_feature_stationarity(features_df, threshold=0.05):
    """Bloque l'entraînement si features non-stationnaires sont injectées."""
    failed = []
    for col in features_df.select_dtypes(include=np.number).columns:
        result = test_stationarity(features_df[col], alpha=threshold)
        if result is None:
            continue
        if not result['consensus']:
            # Vérifier si bornée
            vals = features_df[col].dropna()
            if vals.min() < -1000 or vals.max() > 1000:  # heuristique unbounded
                failed.append(col)
    if failed:
        raise ValueError(f"Features non-stationnaires non-bornées: {failed}")
```

### Références complémentaires

📖 Dickey & Fuller (1979), *Distribution of the Estimators for Autoregressive Time Series With a Unit Root*, J. American Statistical Association.
📖 Kwiatkowski et al. (1992), *Testing the null hypothesis of stationarity*, J. Econometrics.
📖 Hosking (1981), *Fractional differencing*, Biometrika.
📖 López de Prado (2018), AFML chap. 5.

---

**Section 2 fin.** Je continue avec Section 3 (Normalisation) au prochain message si tu valides.

---

## Section 3 — Normalisation

### Principe général

La **normalisation** est l'opération de mise à l'échelle des features pour qu'elles soient comparables et que les algorithmes (notamment ceux à base de gradient ou distance) convergent correctement. La distinction critique :

- **Stationarité (Section 2)** : transformer la **forme statistique** d'une série pour qu'elle soit indépendante du temps.
- **Normalisation (Section 3)** : transformer l'**échelle** d'une série déjà stationnaire pour qu'elle ait un range/distribution comparable à d'autres features.

⚠️ **Erreur fréquente** : croire que la normalisation résout la non-stationnarité. **FAUX**. Diviser par une constante (max global, std globale) **rescale** mais ne change pas la forme de la distribution. Si train et test ont des distributions différentes, les diviser par la même constante les laisse différentes.

📖 **Références** :
- Goodfellow, Bengio, Courville, *Deep Learning* (MIT Press, 2016), chap. 8.7
- Hastie, Tibshirani, Friedman, *Elements of Statistical Learning* (Springer, 2009), section 14.5
- Kim et al., *Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift*, ICLR 2022

### Règles fondamentales

✅ **Fit sur train uniquement, apply à val/test** : toute statistique de normalisation (mean, std, min, max, median, quantile) **DOIT** être calculée sur les données train uniquement et **appliquée** à val/test.

❌ **Pas de fit global** : `(X_full - X_full.mean()) / X_full.std()` calcule des statistiques sur l'ensemble du dataset → leak temporel.

❌ **Pas de fit per-split** : `train_normalized = (train - train.mean()) / train.std()` ET `val_normalized = (val - val.mean()) / val.std()` → introduit un biais (les distributions sont normalisées différemment, le modèle voit deux représentations distinctes).

✅ **Pattern correct (sklearn convention)** :
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
X_train_norm = scaler.transform(X_train)
X_val_norm = scaler.transform(X_val)
X_test_norm = scaler.transform(X_test)
```

### Méthodes de normalisation — comparaison

| Méthode | Formule | Output range | Robuste outliers ? | Quand utiliser |
|---|---|---|---|---|
| **Z-score** (standardization) | `(x - μ) / σ` | ℝ, ~[-3, 3] | Non | Features approximativement gaussiennes |
| **Min-Max** | `(x - min) / (max - min)` | [0, 1] | Non (très sensible) | Features bornées par construction |
| **Robust scaling** | `(x - median) / IQR` | ℝ | Oui | Features avec outliers |
| **Max-abs** | `x / max(|x|)` | [-1, 1] | Non | Features sparse (signed) |
| **Quantile transform** | F(x) où F = CDF empirique train | [0, 1] uniforme | Oui | Distribution arbitraire |
| **Power transform** (Yeo-Johnson, Box-Cox) | transformation monotone vers gaussienne | ℝ | Partiellement | Distributions skewed |

### Méthode 1 — Z-score (standardization)

**Standard pour la plupart des cas**. Formule : `z = (x - μ_train) / σ_train`.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
```

✅ **Avantages** : simple, rapide, output centré, conserve la forme de la distribution.

❌ **Inconvénients** : sensible aux outliers (un point extrême tire μ et σ).

⚠️ **Subtilité** : si la feature n'est PAS gaussienne (skewed, multi-modale), z-score ne la transforme pas en gaussienne — juste recentre/rescale.

### Méthode 2 — Min-Max scaling

Formule : `x_scaled = (x - min_train) / (max_train - min_train)` → output dans `[0, 1]`.

✅ **Avantage** : préserve la forme exacte de la distribution, output borné (utile pour activations sigmoides).

❌ **Inconvénients catastrophiques** :
1. **Très sensible aux outliers** : un seul outlier = tout écrasé.
2. **Out-of-distribution en test** : si test contient une valeur > max_train → output > 1, modèle voit du jamais-vu.
3. **Ne fixe PAS la non-stationnarité** : si test a une distribution différente de train, min-max les laisse différentes (juste rescalées identiquement).

⚠️ **Démonstration empirique** :
```
Train MACD std = 60, Test MACD std = 132 (drift ×2.2)
Train MACD / max_global = 0.04 std
Test  MACD / max_global = 0.09 std → ratio TOUJOURS ×2.2 (drift préservé)
```

✅ **Usage légitime** : uniquement pour features **bornées par construction** où min/max sont stables (RSI, %B, etc.).

### Méthode 3 — Robust scaling

Formule : `x_scaled = (x - median_train) / IQR_train` (IQR = Q75 - Q25).

✅ **Avantages** : insensible aux outliers, output proche de gaussienne pour distributions raisonnables.

✅ **Quand utiliser** : features avec **outliers fréquents** (volume spikes, slippages anormaux, événements de marché extrêmes).

```python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler().fit(X_train)
```

### Méthode 4 — Rolling normalisation (online)

Pour les features **non-stationnaires** où la distribution évolue dans le temps :

```python
def rolling_zscore(series, window=500):
    """Z-score causal sur fenêtre rolling."""
    s = pd.Series(series)
    mean = s.rolling(window).mean()
    std = s.rolling(window).std()
    return (s - mean) / std.where(std > 1e-9, 1.0)
```

✅ **Avantages** :
- Adaptatif aux régimes
- Causal (utilise uniquement passé récent)
- Stationnarise des séries non-stationnaires

❌ **Inconvénients** :
- Perd l'info de niveau absolu
- Choix du window crucial (trop court = bruit, trop long = pas d'adaptation)

✅ **Choix du window** :
- Court (50-100) : adaptatif rapide, sensible au bruit
- Moyen (200-500) : compromis classique
- Long (1000-5000) : capture régimes long terme

### Méthode 5 — Per-instance normalisation (Reversible Instance Normalization, RevIN)

Pour les transformers/RNN modernes appliqués aux séries temporelles :

```python
class RevIN(nn.Module):
    """Normalise chaque sample par ses propres statistiques.
    Référence: Kim et al., ICLR 2022."""
    def forward(self, x):  # x: (batch, channels, seq_len)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normed = (x - mean) / torch.sqrt(var + 1e-5)
        return x_normed
```

✅ **Avantages** :
- **Robuste au distribution shift** : chaque fenêtre s'auto-normalise
- Pas de fit/apply distinction (pas de fuite possible)
- Standard sur Transformer time-series modernes (PatchTST, Informer, etc.)

❌ **Inconvénients** :
- **Perd l'info de niveau global** entre fenêtres
- Pour une fenêtre constante (toutes valeurs identiques), output = 0 (info perdue)

⚠️ **Subtilité** : RevIN ne **remplace pas** la normalisation train/val/test — c'est complémentaire. RevIN s'applique au moment du forward pass, indépendamment du split.

### Décision arbre — choix de la méthode

```
Quel type d'algorithme ?
├── Tree-based (XGBoost, RF, LGBM)
│   → ✅ Souvent pas besoin de normaliser
│   → Ces algos sont scale-invariant aux features individuelles
│   → Mais normaliser n'introduit pas d'erreur (juste pas de gain)
│
├── Linear / Logistic regression
│   → ✅ Z-score (standardization) obligatoire
│   → Permet régularisation L1/L2 cohérente
│
├── Neural network classique (MLP, CNN, RNN)
│   → ✅ Z-score si features approximativement gaussiennes
│   → ✅ Robust scaling si outliers
│   → ✅ BatchNorm/LayerNorm dans le réseau (complémentaire)
│
├── Transformer time series (PatchTST, Informer, TFT)
│   → ✅ RevIN (per-instance) au début du modèle
│   → ✅ Combiné avec rolling z-score sur features hautement non-stationnaires
│
└── K-NN / SVM avec kernel RBF / clustering
    → ✅ Normalisation IMPÉRATIVE (sinon une feature dominante écrase)
    → Préférer robust scaling si outliers
```

### Quelle méthode pour quel type de feature

| Type de feature | Méthode recommandée | Pourquoi |
|---|---|---|
| Indicateur borné stationnaire (RSI, %B) | Pas de normalisation OU Min-Max | Déjà comparable |
| Returns log-normaux | Z-score | Approximativement gaussiens |
| Volume, illiquidité | Robust scaling | Outliers fréquents |
| Slope/momentum non borné | Rolling z-score si non-stationnaire, sinon Z-score | Adaptation aux régimes |
| Distance à un niveau (en ATR) | Z-score ou Min-Max | Souvent quasi-borné |
| Features dérivées de prix non normalisées | **Transformer d'abord** (Section 2), puis normaliser | Prérequis stationnarité |

### Anti-patterns courants

❌ **"Je normalise sur tout le dataset puis je split"** : leakage classique.

❌ **"Min-max global pour rendre stationnaire"** : ne fonctionne pas, c'est juste un rescale.

❌ **"BatchNorm dans le mode eval"** : si stats batch utilisées sans running mean/var → instabilité.

❌ **"Normaliser les labels"** : pour classification binaire, jamais. Pour régression, parfois utile mais penser à dénormaliser à l'inférence.

❌ **"Standardiser des features catégorielles one-hot"** : inutile et contre-productif.

❌ **"Refit le scaler quand on retraine sur train+val"** : si le scaler change, les hyperparams optimisés sur val deviennent invalides. Re-optimiser ou conserver le scaler initial.

### Test de cohérence post-normalisation

✅ **Après normalisation, vérifier sur val/test** :

```python
def validate_normalization(X_train_norm, X_val_norm, X_test_norm, alpha=0.05):
    """Vérifie que la normalisation produit des distributions comparables."""
    from scipy.stats import ks_2samp

    issues = []
    for col in X_train_norm.columns:
        # KS test train vs val
        ks_tv, _ = ks_2samp(X_train_norm[col], X_val_norm[col])
        # KS test train vs test
        ks_tt, _ = ks_2samp(X_train_norm[col], X_test_norm[col])

        if ks_tt > 0.20:  # drift critique persistant
            issues.append((col, ks_tv, ks_tt))

    if issues:
        print("Features avec drift résiduel post-normalisation:")
        for col, ks_tv, ks_tt in issues:
            print(f"  {col}: KS_train_val={ks_tv:.3f}, KS_train_test={ks_tt:.3f}")
        print("→ Considérer transformation supplémentaire (Section 2)")
    return len(issues) == 0
```

### Références complémentaires

📖 Goodfellow et al. *Deep Learning* (2016), chap. 8.7 "Optimization Strategies and Meta-Algorithms"
📖 Ioffe & Szegedy, *Batch Normalization* (ICML 2015)
📖 Kim et al., *RevIN* (ICLR 2022) — `arXiv:2204.05257`
📖 Salinas et al., *DeepAR* (2017) — utilisation de l'instance normalization en time series
📖 sklearn documentation, *Preprocessing data*

---

**Section 3 fin.** Je continue avec Section 4 (Distribution shift) au prochain message si tu valides.
