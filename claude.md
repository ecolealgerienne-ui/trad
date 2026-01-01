# Crypto Trading Signal Prediction - Pipeline de Données IA

## Description du Projet

Modèle d'IA pour prédire la direction de la tendance du Bitcoin (BTC) et autres cryptomonnaies en utilisant la **reconstruction de signal** basée sur des filtres d'Octave.

**Défi Principal** : Les prix sont trop bruités pour être prédits directement. Nous utilisons des filtres mathématiques (filtfilt) pour créer un signal "propre", mais ces filtres nécessitent de connaître le futur. L'IA doit apprendre à reconstruire la pente de ce signal parfait en n'utilisant que les données présentes.

**Objectif** : Atteindre une précision (Accuracy) > 90% sur la prédiction de la pente du signal filtré.

---

## ⚠️ Configuration Importante

### GPU OBLIGATOIRE
**TOUJOURS privilégier l'utilisation du GPU pour ce projet.**

- Configurer TensorFlow/PyTorch pour utiliser le GPU
- Vérifier la disponibilité du GPU avant l'entraînement
- Optimiser les batch sizes en fonction de la mémoire GPU disponible

### Environnement Conda
Conda est activé sur la machine. Utiliser l'environnement `base` ou créer un environnement dédié.

---

## Architecture du Projet

### Structure des Dossiers

```
trad/
├── data/
│   ├── raw/                 # Lien vers ../data_trad (données 5min sources)
│   └── processed/           # Dataset 30min généré (bougie fantôme)
├── src/
│   ├── utils.py            # Fonctions communes PARTAGÉES (NO DUPLICATION)
│   ├── filters.py          # Filtre d'Octave (scipy.signal.filtfilt)
│   ├── indicators.py       # RSI, CCI, MACD, Bollinger
│   ├── normalization.py    # Z-Score, Relative Open
│   ├── labeling.py         # Calcul pente décalée (t-1 vs t-2)
│   └── data_pipeline.py    # Pipeline principal (Bougie Fantôme 30m)
├── notebooks/
│   └── 01_data_validation.ipynb
├── requirements.txt
├── claude.md               # Ce fichier
└── README.md
```

### Données Sources

**Localisation** : `../data_trad/`
- `BTCUSD_all_5m.csv` : Données Bitcoin 5 minutes
- `ETHUSD_all_5m.csv` : Données Ethereum 5 minutes

---

## Spec #1 : Pipeline de Données (Phase Actuelle)

### 1. Concept : Multi-Timeframe Intra-bar

Le modèle travaille sur des **bougies de 30 minutes**, mais "voit" ce qu'il se passe **toutes les 5 minutes** à l'intérieur.

#### La "Bougie Fantôme" (Snapshot)

À chaque intervalle de 5 minutes, générer un vecteur d'état qui agrège la bougie de 30m en cours :

```
t=0min  : Bougie 30m commence
t=5min  : [O, H, L, C] basé sur 1ère bougie 5m
t=10min : [O, H, L, C] mis à jour avec 2 premières bougies 5m
t=15min : [O, H, L, C] mis à jour avec 3 premières bougies 5m
t=20min : [O, H, L, C] mis à jour avec 4 premières bougies 5m
t=25min : [O, H, L, C] mis à jour avec 5 premières bougies 5m
t=30min : Bougie 30m complète (6 bougies 5m)
```

#### Normalisation Multi-Actifs

**JAMAIS utiliser de prix bruts** pour universalité (BTC, ETH, etc.)

1. **Z-Score Glissant** : `(price - mean) / std` sur fenêtre mobile
2. **Relative Open** : Exprimer H, L, C en % de l'Open de la bougie 30m

### 2. La Cible de Prédiction (Labeling) ⚠️ CRITIQUE

#### Signal Cible (Y)
- Appliquer filtre d'Octave (`scipy.signal.filtfilt`) avec paramètre de lissage `0.25`
- Sur l'indicateur **RSI** (ou Close)

#### Pente
- Ne PAS prédire la valeur, mais le **signe de la pente**
- Pente positive → Label = 1
- Pente négative → Label = 0

#### Décalage (Offset)
**Le modèle à l'instant t doit prédire si la pente entre t-2 et t-1 est positive.**

```python
# Pseudo-code
signal_filtered = filtfilt(signal)  # Utilise passé + futur
slope = signal_filtered[t-1] - signal_filtered[t-2]
label[t] = 1 if slope > 0 else 0
```

Note : On utilise les données de t pour prédire le point juste avant (stabilisation).

### 3. Features Engineering (Inputs)

Indicateurs à calculer et **stacker** (empiler) :
- **RSI** (Relative Strength Index)
- **CCI** (Commodity Channel Index)
- **MACD** (Moving Average Convergence Divergence)
- **Bandes de Bollinger**

Chaque indicateur sur **plusieurs fenêtres temporelles** (multi-timeframe).

---

## Règles de Développement

### 1. Partage de Code - ZÉRO DUPLICATION
- **TOUJOURS vérifier si une fonction existe déjà** avant de la réécrire
- Mettre les fonctions communes dans `utils.py`
- Réutiliser entre tous les scripts

### 2. Prévention de Data Leakage
- **JAMAIS utiliser de données futures** dans les features
- Le filtre d'Octave utilise le futur UNIQUEMENT pour le label (cible)
- Valider qu'aucune information de t+1 ne fuite dans les inputs de t

### 3. Format de Sortie
- Dataset final : **CSV/DataFrame**
- Colonnes : `[timestamp, feat1, feat2, ..., featN, label]`
- Label : `{0, 1}` (pente négative ou positive)

---

## Spec #2 : Architecture IA (Phase Future)

### Modèle Hybride CNN-LSTM ou TCN

1. **Couche Convolutionnelle (CNN)** : Extraire motifs sur 10 dernières bougies
2. **Couche Récurrente (LSTM)** : Mémoriser séquence des steps (5m, 10m, 15m...)
3. **Couche Dense (Stacking)** : Combiner probabilités → décision finale (0 ou 1)

---

## Méthodologie de Développement

### Phase 1 : Pipeline de Données ✅ (En cours)
1. Créer scripts de transformation 5m → 30m
2. Générer dataset avec bougie fantôme
3. Calculer indicateurs + normalisation
4. Calculer labels (pente filtrée décalée)
5. Valider dataset (pas de leakage, distribution labels)

### Phase 2 : Modèle IA (À venir)
1. Architecture CNN-LSTM/TCN
2. Entraînement avec GPU
3. Validation croisée temporelle
4. Backtesting

---

## Commandes Utiles

### Vérifier GPU
```python
# TensorFlow
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

# PyTorch
import torch
print(torch.cuda.is_available())
```

### Lancer Pipeline
```bash
python src/data_pipeline.py --input ../data_trad/BTCUSD_all_5m.csv --output data/processed/btc_30m_dataset.csv
```

---

## Notes pour Claude

- **GPU FIRST** : Toujours vérifier et utiliser le GPU
- **DRY Principle** : Don't Repeat Yourself - Partager le code via utils.py
- **Data Integrity** : Vérifier SYSTÉMATIQUEMENT qu'il n'y a pas de data leakage
- **Validation** : Chaque étape doit être validée avant de passer à la suivante
- Objectif Accuracy > 90% nécessite une **qualité de données parfaite**

---

## Licence

Ce projet est sous licence Unlicense - voir le fichier [LICENSE](LICENSE) pour plus de détails.
