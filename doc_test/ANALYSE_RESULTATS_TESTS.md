# Analyse des Résultats de Tests

**Date d'analyse** : 2026-01-03
**Analysé par** : Claude Code
**Documents analysés** : 8 fichiers dans doc_test/

---

## Table des Matières

1. [Vue d'Ensemble](#vue-densemble)
2. [Impact du Delta sur l'Accuracy](#impact-du-delta-sur-laccuracy)
3. [Comparaison des Indicateurs](#comparaison-des-indicateurs)
4. [Impact des Périodes d'Indicateurs](#impact-des-périodes-dindicateurs)
5. [Comparaison des Filtres (Kalman vs Octave)](#comparaison-des-filtres-kalman-vs-octave)
6. [Problème de Généralisation](#problème-de-généralisation)
7. [Stacking et Ensemble Methods](#stacking-et-ensemble-methods)
8. [Résultats de Trading](#résultats-de-trading)
9. [Test de Validation : 3 Features → 1 Target](#test-de-validation--3-features--1-target-2026-01-03)
10. [Test Octave : 1 Feature → CLOSE](#test-octave--1-feature--close-2026-01-03) ⭐ **NOUVEAU**
11. [Conclusions et Recommandations](#conclusions-et-recommandations)

---

## Vue d'Ensemble

Les documents analysés couvrent une série d'expériences sur la prédiction de tendance crypto avec différentes configurations :

| Document | Contenu Principal |
|----------|-------------------|
| ResultKelmanFilfi.docx | Tests Kalman Filter avec trading sur 303 jours |
| ResultatsDesTests.docx | Comparaison complète delta 0-5, tous indicateurs |
| ResultatsDesTests0.docx | Tests KF_CLOSE, correction labels, paramètres optimaux |
| ResultatsDesTests1.docx | Filtre Octave, Stacking 1 et 2 |
| ResultatsDesTests3.docx | Suite tests Octave, analyse généralisation |
| SuiviCloseFil.docx | Stratégie CLOSE, multi-indicateurs |
| SuiviFK_Fil.docx | Comparaison visuelle filtres |
| ResultTrain.odt | Résultats d'entraînement |

**Période de données** :
- Apprentissage : ~5 ans de données historiques
- Généralisation : 10 mois de 2024 (données jamais vues)
- Cryptos principales : BTC, SOL, ETH, etc.
- Cryptos de test : LTC, LINK, XLM (jamais vues pendant l'apprentissage)

---

## Impact du Delta sur l'Accuracy

Le "delta" représente le décalage temporel entre la prédiction et la valeur cible :
- `delta=0` : Prédire la pente actuelle (t-1 → t)
- `delta=1` : Prédire avec 1 pas de retard
- `delta=n` : Prédire avec n pas de retard

### Résultats Observés

| Delta | AUC Train | AUC Test | Acc Train | Acc Test | Gap Train/Test |
|-------|-----------|----------|-----------|----------|----------------|
| 0 | 0.935-0.951 | 0.921-0.940 | 85.2-87.7% | 83.7-86.4% | ~2% |
| 1 | 0.962-0.973 | 0.953-0.966 | 89.1-91.2% | 87.7-90.1% | ~2% |
| 2 | 0.976-0.983 | 0.968-0.978 | 91.4-93.2% | 90.1-92.1% | ~1.5% |
| 3 | 0.983-0.988 | 0.977-0.983 | 93.0-94.4% | 91.6-93.3% | ~1.5% |
| 4 | 0.988-0.991 | 0.982-0.987 | 94.1-95.2% | 92.6-94.1% | ~1.5% |
| 5 | 0.990-0.993 | 0.986-0.989 | 94.8-95.7% | 93.4-94.7% | ~1% |

### Analyse

1. **Progression linéaire** : Chaque delta supplémentaire apporte ~2% d'accuracy
2. **Delta 0 est le plus difficile** : Le modèle doit prédire "en temps réel"
3. **Delta 5+ est "trop facile"** : Haute accuracy mais inutile pour le trading (trop de lag)
4. **Sweet spot** : Delta 1-2 offre le meilleur compromis

### Interprétation

Plus le delta augmente, plus la tâche devient facile car :
- Les tendances ont plus de temps pour se confirmer
- Le bruit haute fréquence est filtré
- Mais le signal devient exploitable trop tard pour le trading

**Recommandation** : Utiliser **delta = 1 ou 2** pour le trading réel.

---

## Comparaison des Indicateurs

### Performance par Indicateur (Delta = 0)

| Indicateur | AUC Train | AUC Test | Acc Train | Acc Test |
|------------|-----------|----------|-----------|----------|
| **MACD** | 0.941 | 0.933 | 86.2% | 85.4% |
| CCI | 0.951 | 0.940 | 87.7% | 86.4% |
| BOL | 0.945 | 0.933 | 87.0% | 85.4% |
| RSI | 0.935 | 0.921 | 85.2% | 83.7% |

### Performance par Indicateur (Delta = 2)

| Indicateur | AUC Train | AUC Test | Acc Train | Acc Test |
|------------|-----------|----------|-----------|----------|
| **MACD** | 0.976 | 0.972 | 91.4% | 91.0% |
| CCI | 0.983 | 0.978 | 93.2% | 92.1% |
| BOL | 0.980 | 0.973 | 92.4% | 91.2% |
| RSI | 0.976 | 0.968 | 91.5% | 90.1% |

### Performance par Indicateur (Delta = 5, avec Stacking)

| Indicateur | Acc Train Stack | Acc Test Stack |
|------------|-----------------|----------------|
| **MACD** | 96.7% | 96.2% |
| CCI | 95.7% | 94.7% |
| BOL | 95.3% | 94.0% |
| RSI | 94.8% | 93.4% |
| CLOSE | 87.2% | 86.5% |

### Analyse

1. **MACD domine** pour les deltas élevés (mesure de tendance lourde)
2. **CCI est excellent** sur tous les deltas
3. **RSI est le moins performant** (oscillateur trop nerveux)
4. **CLOSE est le plus difficile** à prédire directement
5. **BOL a été retiré** dans les tests récents (impossible à synchroniser)

### Hiérarchie Finale

```
MACD > CCI > BOL > RSI > CLOSE
```

---

## Impact des Périodes d'Indicateurs

### Tests avec Différentes Périodes RSI

| Période RSI | Acc Train | Acc Test | Différence |
|-------------|-----------|----------|------------|
| RSI(5) | 86.1% | 85.0% | - |
| RSI(9) | 86.2% | 85.0% | +0.0% |
| RSI(14) | 86.2% | 85.0% | +0.0% |

### Tests avec Différentes Périodes CCI

| Période CCI | Acc Train | Acc Test | Différence |
|-------------|-----------|----------|------------|
| CCI(9) | 85.5% | 84.8% | - |
| CCI(13) | 85.4% | 84.6% | -0.2% |
| CCI(20) | 85.3% | 84.5% | -0.3% |

### Tests avec Différentes Périodes MACD

| MACD (fast/slow) | Acc Train | Acc Test | Différence |
|------------------|-----------|----------|------------|
| MACD(9,x) | 83.2% | 81.9% | - |
| MACD(13,x) | 83.4% | 82.4% | +0.5% |
| MACD(26,x) | 82.5% | 81.6% | -0.3% |

### Conclusion Majeure

**La période des indicateurs a un impact NÉGLIGEABLE sur l'accuracy** (< 0.5%).

Cela s'explique par :
1. Les indicateurs à différentes périodes sont fortement corrélés
2. Le modèle extrait le même signal peu importe la période
3. L'information est dans la direction, pas dans la valeur exacte

**Recommandation** : Utiliser les périodes standards (RSI=14, CCI=20, MACD=12/26) et ne pas optimiser.

---

## Comparaison des Filtres (Kalman vs Octave)

### Filtre de Kalman

Paramètres testés : process_var, measure_var

| Configuration | Avantages | Inconvénients |
|---------------|-----------|---------------|
| Kalman standard | Bon lissage | Gap généralisation 4% |
| Kalman adaptatif | Réactif | Instable |

### Filtre d'Octave

Paramètres testés : 0.15, 0.20, 0.25

| Filtre | Acc Train | Acc Test | Gap | Commentaire |
|--------|-----------|----------|-----|-------------|
| 0.15 | 89.6% | 84.5% | 5.1% | Trop réactif, apprentissage difficile |
| **0.20** | 90.8% | 88.0% | 2.8% | **Bon compromis** |
| **0.25** | 91.2% | 88.7% | 2.5% | **Meilleur pour généralisation** |

### Comparaison Directe (Delta = 2, MACD)

| Filtre | Acc Test | Gap Train/Test |
|--------|----------|----------------|
| Kalman | 91.0% | 4% |
| Octave 0.20 | 93.6% | 2% |
| Octave 0.25 | 94.3% | 2% |

### Conclusion

**Le filtre Octave (0.20 ou 0.25) surpasse Kalman** :
- Meilleure accuracy test (+2-3%)
- Meilleure généralisation (gap réduit de moitié)
- Plus stable sur données non vues

---

## Problème de Généralisation

### Gaps Observés

| Scénario | Gap Train → Test | Gap Test → 10 mois |
|----------|------------------|-------------------|
| Même crypto (BTC) | 1-2% | 2-4% |
| Cryptos jamais vues | 2-3% | 4-5% |
| Avec Kalman | 2% | 4% |
| Avec Octave | 1.5% | 2% |

### Cryptos Jamais Vues

Test sur LTC, LINK, XLM (non inclus dans l'apprentissage) :

| Crypto | Acc obtenue | Perte vs Train |
|--------|-------------|----------------|
| LTC | ~85% | -4% |
| LINK | ~84% | -5% |
| XLM | ~84% | -5% |

### Analyse du Problème

1. **Overfitting temporel** : Le modèle apprend des patterns spécifiques à la période d'entraînement
2. **Régimes de marché** : 2024 peut avoir des caractéristiques différentes des 5 ans précédents
3. **Corrélations changeantes** : Les relations entre indicateurs évoluent

### Pistes d'Amélioration Identifiées

Les documents mentionnent plusieurs pistes :

1. **Réduire le pas temporel** : Passer de 15 à 10 ou 7 minutes
2. **Enlever un indicateur** : Potentiellement MACD (paradoxalement le meilleur mais peut-être overfitté)
3. **Limiter les deltas** : Utiliser seulement delta 0, 1, 2 (pas 3+)
4. **Un seul filtre** : Utiliser uniquement 0.20
5. **Indicateurs bornés** : Privilégier RSI, Williams %R, Stochastic (entre 0 et 100)

---

## Stacking et Ensemble Methods

### Stacking 1 : Combinaison Simple

```python
X_meta = np.column_stack([
    predict_proba_RSI_20,
    predict_proba_CCI_20,
    predict_proba_BOL_20,
    predict_proba_MACD_20,
    predict_proba_RSI_25,
    predict_proba_CCI_25,
    predict_proba_BOL_25,
    predict_proba_MACD_25
])
```

| Delta | Acc Sans Stack | Acc Avec Stack 1 | Gain |
|-------|----------------|------------------|------|
| 0 | 85.4% | 86.4% | +1.0% |
| 1 | 88.5% | 89.5% | +1.0% |
| 2 | 91.0% | 92.0% | +1.0% |
| 3 | 93.3% | 94.5% | +1.2% |

### Stacking 2 : Combinaison Avancée

| Delta | Acc Stack 1 | Acc Stack 2 | Gain |
|-------|-------------|-------------|------|
| 2 | 92.0% | 94.1% | +2.1% |
| 3 | 94.5% | 96.0% | +1.5% |

### Conclusion Stacking

- **Stacking 1** : Gain constant de ~1%
- **Stacking 2** : Gain supplémentaire de ~2%
- **Total** : +3% avec stacking complet
- **Attention** : Le gap de généralisation peut augmenter (5% mentionné)

---

## Résultats de Trading

### Trading avec KF_CLOSE (10 mois, BTC)

| Delta | % Gain | Trans/Jour | % Exact | Profit Factor |
|-------|--------|------------|---------|---------------|
| 0 | 982.98% | 3.25 | 67.3% | 7.45 |
| 1 | 783.38% | 2.59 | 65.0% | 6.38 |
| 2 | 637.10% | 2.10 | 62.4% | 5.19 |
| 3 | 562.09% | 1.86 | 59.7% | 4.62 |
| 6 | 404.42% | 1.34 | 55.7% | 3.72 |

### Observations Trading

1. **Delta 0 donne les meilleurs résultats de trading** malgré une accuracy plus faible
2. **Plus de transactions = plus de profit** (dans ce backtest)
3. **Profit Factor excellent** (> 4 pour tous les deltas)
4. **Attention** : Ces résultats sont sur données d'apprentissage

### Problème Temps Réel

> "Résultats excellent, mais un problème dans le temps réel pour les valeurs à t0."

Le document mentionne un problème avec les valeurs à t0 en temps réel - probablement lié au look-ahead bias ou au timing d'exécution.

---

## Test de Validation : 3 Features → 1 Target (2026-01-03)

### Protocole de Test

Test systématique avec le script `test_indicator_params.py` :
- **Input** : 3 features (RSI, CCI, MACD) avec paramètres variables
- **Output** : Kalman(indicateur) avec paramètres fixes standards
- **Dataset** : BTC uniquement, ~50k samples
- **Modèle** : CNN-LSTM mono-output

### Grilles de Paramètres Testées

| Indicateur | Paramètres testés |
|------------|-------------------|
| RSI period | 7, 10, 14, 20 |
| CCI period | 10, 14, 20, 30 |
| MACD fast/slow | (8,17), (12,26), (16,34) |

**Total** : 4 × 4 × 3 = 48 combinaisons par target

### Résultats par Target

#### Target: Kalman(RSI) - Le plus difficile

| RSI | CCI | MACD | Accuracy |
|-----|-----|------|----------|
| 14 | 30 | 16/34 | **79.10%** |
| 10 | 30 | 12/26 | 79.00% |
| 14 | 20 | 16/34 | 78.90% |
| 14 | 30 | 12/26 | 78.90% |
| 10 | 30 | 16/34 | 78.80% |

**Spread** : 0.30% (78.80% → 79.10%)
**Meilleur** : RSI=14, CCI=30, MACD=16/34

#### Target: Kalman(CCI) - Difficulté moyenne

| RSI | CCI | MACD | Accuracy |
|-----|-----|------|----------|
| 7 | 20 | 12/26 | **83.30%** |
| 7 | 20 | 16/34 | 83.30% |
| 10 | 20 | 12/26 | 83.30% |
| 10 | 20 | 16/34 | 83.30% |
| 14 | 20 | 16/34 | 83.30% |

**Spread** : 0.00% (tous à 83.30%)
**Meilleur** : Tous équivalents avec CCI=20

#### Target: Kalman(MACD) - Le plus facile

| RSI | CCI | MACD | Accuracy |
|-----|-----|------|----------|
| 10 | 30 | 8/17 | **86.40%** |
| 14 | 10 | 12/26 | 86.40% |
| 14 | 30 | 8/17 | 86.30% |
| 7 | 20 | 8/17 | 86.20% |
| 7 | 20 | 12/26 | 86.20% |

**Spread** : 0.20% (86.20% → 86.40%)
**Meilleur** : RSI=10, CCI=30, MACD=8/17

### Analyse des Résultats

#### 1. Confirmation de la hiérarchie des targets

| Target | Accuracy | Difficulté |
|--------|----------|------------|
| Kalman(MACD) | 86.40% | Facile ✅ |
| Kalman(CCI) | 83.30% | Moyen |
| Kalman(RSI) | 79.10% | Difficile ❌ |

**Écart RSI vs MACD** : 7.3 points (significatif)

#### 2. Impact négligeable des paramètres

| Target | Spread Top 5 | Conclusion |
|--------|--------------|------------|
| RSI | 0.30% | Négligeable |
| CCI | 0.00% | Aucun impact |
| MACD | 0.20% | Négligeable |

**Tous les spreads sont < 0.5%** → L'optimisation des paramètres n'améliore pas significativement les résultats.

#### 3. Patterns observés

1. **CCI=30** apparaît souvent dans les meilleurs résultats
   - Période plus longue = signal plus stable

2. **Pour target CCI** : Tous les paramètres donnent exactement 83.30%
   - Le modèle extrait le même signal peu importe les params

3. **Pour target MACD** : MACD=8/17 (court) performe bien
   - Paradoxe : paramètres courts pour un indicateur de tendance lourde

### Conclusion du Test de Validation

> **CONFIRMÉ : L'optimisation des paramètres d'indicateurs est inutile.**

Le choix de la **cible** (MACD vs CCI vs RSI) a un impact de **7.3%** sur l'accuracy.
Le choix des **paramètres** a un impact de **< 0.5%**.

**Priorité** : Choisir la bonne cible, pas les bons paramètres.

---

## Test Octave : 1 Feature → CLOSE (2026-01-03)

### Protocole de Test

Nouveau test avec filtre Octave et architecture mono-feature :
- **Input** : 1 seul indicateur (RSI, CCI, ou MACD)
- **Output** : Direction Octave(CLOSE, 0.20) = `filtered[t-1] > filtered[t-2]`
- **Dataset** : BTC, ETH, BNB, ADA, LTC (5 assets)
- **Modèle** : CNN-LSTM mono-output
- **Script** : `prepare_data_octave.py`

### Résultats Octave (1 feature → CLOSE)

| Indicateur | Accuracy | Precision | Recall | F1 | Gap Train/Test |
|------------|----------|-----------|--------|-----|----------------|
| RSI(14) | **78.5%** | 0.775 | 0.809 | 0.792 | **0.1%** ✅ |
| CCI(20) | 77.7% | 0.769 | 0.796 | 0.782 | **0.2%** ✅ |
| MACD(12/26) | 76.2% | 0.757 | 0.777 | 0.767 | **0.3%** ✅ |

### Comparaison Kalman(INDICATEUR) vs Octave(CLOSE)

| Target | Filtre | RSI | CCI | MACD | Meilleur |
|--------|--------|-----|-----|------|----------|
| Kalman(INDIC) | Kalman | 79.1% | 83.3% | **86.4%** | MACD ✅ |
| Octave(CLOSE) | Octave 0.20 | **78.5%** | 77.7% | 76.2% | RSI ✅ |

### Observation Majeure : Hiérarchie INVERSÉE !

| Avec Kalman(INDICATEUR) | Avec Octave(CLOSE) |
|-------------------------|---------------------|
| 1. MACD (86.4%) 🥇 | 1. RSI (78.5%) 🥇 |
| 2. CCI (83.3%) 🥈 | 2. CCI (77.7%) 🥈 |
| 3. RSI (79.1%) 🥉 | 3. MACD (76.2%) 🥉 |

**Interprétation** :
- Quand on prédit **l'indicateur lui-même** (Kalman) → MACD gagne (auto-corrélation forte)
- Quand on prédit le **CLOSE** → RSI gagne (meilleure corrélation prix/momentum)

### Analyse Complète

1. **Accuracy moyenne** : 77.5% (Octave) vs 82.9% (Kalman)
   - Perte de ~5% en changeant la target
   - Mais prédire CLOSE est plus utile pour le trading !

2. **Généralisation exceptionnelle** : Gap < 0.3% pour tous les indicateurs
   - Kalman avait ~2% de gap
   - Octave = filtre plus stable, moins d'overfitting

3. **RSI meilleur pour CLOSE** :
   - RSI = oscillateur de momentum = corrélé aux retournements de prix
   - MACD = indicateur de tendance = moins réactif aux pivots

### Conclusion

> **Pour prédire la direction du CLOSE filtré, utiliser RSI comme feature.**
> **Pour prédire la direction d'un indicateur, utiliser cet indicateur comme target (mais moins utile pour le trading).**

---

## Conclusions et Recommandations

### Ce qui Fonctionne

1. **MACD** comme indicateur principal (le plus prédictible)
2. **Delta 1-2** pour le trading (compromis accuracy/lag)
3. **Filtre Octave 0.20/0.25** (meilleure généralisation que Kalman)
4. **Stacking** pour +2-3% d'accuracy
5. **Prédire CLOSE** plutôt que les indicateurs individuels

### Ce qui Ne Fonctionne Pas

1. **Optimisation des périodes** : Impact négligeable (< 0.5%)
2. **Delta 0** : Difficile et problématique en temps réel
3. **Filtre 0.15** : Trop réactif, apprentissage difficile
4. **Kalman Filter seul** : Gap de généralisation trop important

### Recommandations Finales

| Paramètre | Valeur Recommandée | Justification |
|-----------|-------------------|---------------|
| Delta | 1 ou 2 | Compromis accuracy/lag |
| Filtre | Octave 0.20 | Meilleure généralisation |
| Indicateur cible | CLOSE | Plus universel |
| Features | RSI, CCI, MACD | Combinaison complémentaire |
| Périodes | Standards (14, 20, 12/26) | L'optimisation n'aide pas |
| Stacking | Oui (Stack 1 minimum) | +1-2% gain facile |

### Questions Ouvertes

1. Pourquoi le trading en temps réel diffère du backtest ?
2. Comment réduire le gap de généralisation sous 2% ?
3. Le stacking augmente-t-il l'overfitting ?
4. Faut-il re-entraîner mensuellement ?

---

## Annexe : Paramètres Optimaux Identifiés

### Configuration Finale Suggérée

```python
# Filtre
FILTER_TYPE = 'octave'
FILTER_PARAM = 0.20  # ou 0.25

# Delta
DELTA = 2  # ou 1 pour plus de réactivité

# Indicateurs (périodes standards)
RSI_PERIOD = 14
CCI_PERIOD = 20
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Cible
TARGET = 'CLOSE'  # Prédire la direction du prix filtré

# Stacking
USE_STACKING = True
STACKING_LEVEL = 1  # ou 2 si plus de données
```

---

*Document généré automatiquement par Claude Code*
*Basé sur l'analyse de 8 fichiers de tests*
*Dernière mise à jour : 2026-01-03 (ajout résultats Octave RSI→CLOSE: 78.5%)*
