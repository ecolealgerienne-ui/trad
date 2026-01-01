# Crypto Trading Signal Prediction - Data Pipeline

Pipeline de données pour prédire la direction de la tendance du Bitcoin et autres cryptomonnaies en utilisant la reconstruction de signal basée sur des filtres d'Octave.

## 🎯 Objectif

Atteindre une **précision > 90%** sur la prédiction de la pente du signal filtré.

## 🚀 Quick Start

### 1. Installation

```bash
# Installer les dépendances
conda activate base  # ou votre environnement conda
pip install -r requirements.txt
```

### 2. Générer le Dataset

```bash
# Transformer données 5min → 30min avec bougie fantôme
python src/data_pipeline.py \
    --input ../data_trad/BTCUSD_all_5m.csv \
    --output data/processed/btc_30m_dataset.csv \
    --label-source rsi \
    --smoothing 0.25
```

### 3. Valider le Dataset

```bash
# Lancer le notebook de validation
jupyter notebook notebooks/01_data_validation.ipynb
```

## 📁 Structure du Projet

```
trad/
├── data/
│   ├── raw/                    # Lien vers ../data_trad
│   └── processed/              # Dataset 30min généré
├── src/
│   ├── utils.py               # Fonctions communes
│   ├── filters.py             # Filtre d'Octave (filtfilt)
│   ├── indicators.py          # RSI, CCI, MACD, Bollinger
│   ├── normalization.py       # Z-Score, Relative Open
│   ├── labeling.py            # Calcul pente décalée
│   └── data_pipeline.py       # Pipeline principal
├── notebooks/
│   └── 01_data_validation.ipynb
├── requirements.txt
├── claude.md                  # Documentation détaillée
└── README.md
```

## 🔬 Concept: Bougie Fantôme (Ghost Candle)

Le modèle travaille sur des **bougies de 30 minutes**, mais "voit" ce qu'il se passe **toutes les 5 minutes** à l'intérieur.

### Exemple

Pour une bougie 30min (de 10:00 à 10:30):

```
t=10:05 → [O, H, L, C] basé sur 1ère bougie 5m
t=10:10 → [O, H, L, C] mis à jour avec 2 premières bougies 5m
t=10:15 → [O, H, L, C] mis à jour avec 3 premières bougies 5m
t=10:20 → [O, H, L, C] mis à jour avec 4 premières bougies 5m
t=10:25 → [O, H, L, C] mis à jour avec 5 premières bougies 5m
t=10:30 → Bougie 30m complète (6 bougies 5m)
```

Chaque ligne = un snapshot de la bougie 30m en formation.

## 🏷️ Labeling: Reconstruction de Signal

**Workflow:**

1. Calculer RSI sur les prix
2. Appliquer **filtre d'Octave** (filtfilt, smoothing=0.25)
3. Calculer la **pente** du signal filtré
4. **Décalage temporel**: Label[t] prédit la pente entre t-2 et t-1
5. Label = 1 si pente > 0, sinon 0

```python
# Pseudo-code
signal_filtered = octave_filter(rsi, smoothing=0.25)  # Utilise passé + futur
slope = signal_filtered[t-1] - signal_filtered[t-2]
label[t] = 1 if slope > 0 else 0
```

⚠️ **Le filtre utilise le futur UNIQUEMENT pour le label, jamais pour les features!**

## 🎛️ Pipeline Options

```bash
python src/data_pipeline.py --help
```

**Options principales:**

- `--input`: Fichier CSV 5min source
- `--output`: Fichier CSV de sortie
- `--timeframe`: Timeframe cible (défaut: 30T)
- `--label-source`: Source pour labels (rsi ou close)
- `--smoothing`: Paramètre de lissage (0.0-1.0, défaut: 0.25)
- `--no-indicators`: Ne pas calculer les indicateurs techniques
- `--add-history`: Ajouter features historiques (10 dernières bougies)

## 📊 Indicateurs Calculés

- **RSI** (14, 21)
- **CCI** (20)
- **MACD** (12/26/9)
- **Bollinger Bands** (20)
- **ATR** (14)
- **Stochastic** (14/3)

Tous normalisés avec **Z-Score glissant** (window=50).

## 🔍 Normalisation

### Bougie Fantôme

**Relative Open**: H, L, C exprimés en % de l'Open

```python
rel_high = (ghost_high - ghost_open) / ghost_open * 100
rel_low = (ghost_low - ghost_open) / ghost_open * 100
rel_close = (ghost_close - ghost_open) / ghost_open * 100
```

### Indicateurs

**Z-Score glissant** (causal):

```python
z = (x - rolling_mean) / rolling_std
```

## ⚠️ Prévention Data Leakage

Le pipeline vérifie automatiquement:

1. Aucune feature n'utilise de données futures
2. Corrélation feature[t] × label[t+1] < 0.7
3. Toutes les transformations sont causales (sauf le filtre pour labels)

## 🧪 Validation

Le notebook `01_data_validation.ipynb` vérifie:

- ✅ Intégrité OHLC des bougies fantômes
- ✅ Distribution des labels (équilibre 40-60%)
- ✅ Pas de data leakage
- ✅ Qualité des features normalisées
- ✅ Visualisation du signal filtré

## 🖥️ GPU Configuration

**TOUJOURS utiliser le GPU pour ce projet!**

### Vérifier GPU (TensorFlow)

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

### Vérifier GPU (PyTorch)

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

## 📈 Prochaines Étapes (Phase 2)

1. **Modèle CNN-LSTM/TCN**
2. **Entraînement avec GPU**
3. **Validation croisée temporelle**
4. **Backtesting**

Voir `claude.md` pour les specs complètes.

## 📝 Licence

Unlicense - Voir [LICENSE](LICENSE)

## 🤝 Contribution

Voir les règles de développement dans `claude.md`:

- **ZÉRO duplication**: Réutiliser les fonctions de `utils.py`
- **Data Integrity**: Vérifier systématiquement le data leakage
- **GPU First**: Toujours privilégier le GPU

---

**Développé avec Claude Code** 🤖
